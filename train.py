'''
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
'''

import os
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import GPTConfig, GPT
# -----------------------------------------------------------------------------

# I/O
out_dir = 'out'
eval_interval = 200
log_interval = 1
eval_iters = 200
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval
init_from = 'scratch' # 'scratch' or 'resume'

# data
dataset = 'slither-audited-smart-contracts'
gradient_accumulation_steps = 5 * 8 # used to simulate larger batch sizes
batch_size = 12 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 1024

# model
n_layer = 8
n_head = 8
n_embd = 768
dropout = 0.0 # for pretraining 0 is good, for fine-tuning dropout must be 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?

# AdamW optimizer
learning_rate = 6e-4 # max learning rate
max_iters = 600000 # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0

# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 2000 # how many steps to warm up for
lr_decay_iters = 600000 # should be ~= max_iters as per Chinchilla (scaling laws)
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 as per Chinchilla (scaling laws)

# system
device = 'cpu'

if torch.cuda.is_available():
  device = 'cuda'
  torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
  torch.backends.cuda.allow_tf32 = True # allow tf32 on cudnn

elif hasattr(torch.backends, 'mps') and torch.mps.is_available():
  device = 'mps'

print(f'Using device: {device}')

dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = True if device == 'cuda' else False # use PyTorch 2.0 to compile the model to be faster, True do not work for 'mps'
# -----------------------------------------------------------------------------

# set up DDP (Distributed Data Parallel)
# torchrun command sets the environment variables RANK, LOCAL_RANK and WORLD_SIZE
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
  if torch.cuda.is_available():
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
  
  else:
    init_process_group(backend='gloo') # for cpu/mps
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cpu:{ddp_local_rank}'
    torch.cpu.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size

else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1

tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f'tokens per iteration will be: {tokens_per_iter:,}')

if master_process:
  os.makedirs(out_dir, exist_ok=True)

torch.manual_seed(1337 + seed_offset)

ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = torch.amp.autocast(device_type=device, dtype=ptdtype) if device == 'cuda' else nullcontext()
# -----------------------------------------------------------------------------

# Data Loader
data_dir = os.path.join('data', dataset)
def get_batch(split):
  # Create a memory-map to an array stored in a binary file on disk.
  # Memory-mapped files are used for accessing small segments of large files on disk, 
  # without reading the entire file into memory. NumPy’s memmap’s are array-like objects.
  # We recreate np.memmap every batch to avoid a memory leak.
  if split == 'train':
    data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
  
  else:
    data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')

  ix = torch.randint(len(data) - block_size, (batch_size,))
  x = torch.stack([torch.from_numpy((data[i: i + block_size]).astype(np.int64)) for i in ix])
  y = torch.stack([torch.from_numpy((data[i + 1: i + 1 + block_size]).astype(np.int64)) for i in ix])

  if device == 'cuda':
    x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
  
  else:
    x, y = x.to(device), y.to(device)
  
  return x, y
# -----------------------------------------------------------------------------

