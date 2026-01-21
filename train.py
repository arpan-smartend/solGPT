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
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import SolGPTConfig, SolGPT
# -----------------------------------------------------------------------------

# I/O
out_dir = 'out'
log_dir = 'out'
eval_interval = 200
log_interval = 1
eval_iters = 200
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval
init_from = 'resume' # 'scratch' or 'resume' # TODO: take arg from command line

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
vocab_size = 50496  

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
  os.makedirs(log_dir, exist_ok=True)

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

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# model init TODO: take args from command line
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, dropout=dropout, vocab_size=50496)

if init_from == 'scratch':
  # init a new model from scratch
  print('Initializing a new model from scratch')
  solGPT_config = SolGPTConfig(**model_args)
  model = SolGPT(solGPT_config)

  if master_process:
    log_file = os.path.join(log_dir, 'log.txt')
    with open(log_file, 'w') as f: # open for writing to clear the file
      pass

elif init_from == 'resume':
  print(f'Resuming training from {out_dir}')
  # resume training from a checkpoint
  ckpt_path = os.path.join(out_dir, 'ckpt.pt')
  checkpoint = torch.load(ckpt_path, map_location=device)
  checkpoint_model_args = checkpoint['model_args']

  # force these config attributes to be equal otherwise we can't even resume training
  # the rest of the attributes (e.g. dropout) can stay as desired from command line
  for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
    model_args[k] = checkpoint_model_args[k]

  # create the model
  solGPT_config = SolGPTConfig(**model_args)
  model = SolGPT(solGPT_config)
  state_dict = checkpoint['model']

  # The _orig_mod prefix appears in a PyTorch model's state_dict when the model has been compiled using torch.compile(). 
  # The compiled model is wrapped in an OptimizedModule, and the original model is stored internally as _orig_mod, 
  # leading to key mismatches if you try to load the checkpoint with an uncompiled model. 
  unwanted_prefix = '_orig_mod.'
  for k,v in list(state_dict.items()):

    if k.startswith(unwanted_prefix):
      state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

  model.load_state_dict(state_dict)
  iter_num = checkpoint['iter_num']
  best_val_loss = checkpoint['best_val_loss']

  if master_process:
    log_file = os.path.join(log_dir, 'log.txt')

    if not os.path.exists(log_file) and not os.path.isfile(log_file):
      with open(log_file, 'w') as f: # open for writing to clear the file
        pass


else:
  raise ValueError(f'Invalid init_type: {init_from}')

model.to(device)
# -----------------------------------------------------------------------------

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.amp.GradScaler(enabled=(dtype == 'float16'))

# optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type=device)

if init_from == 'resume':
  optimizer.load_state_dict(checkpoint['optimizer'])

checkpoint = None # free up memory
# -----------------------------------------------------------------------------

# compile the model
if compile:
  print('compiling the model... (takes a ~minute)')
  model = torch.compile(model)

# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
  
raw_model = model.module if ddp else model # unwrap DDP container if needed
# -----------------------------------------------------------------------------

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
  out = {}
  model.eval()
  for split in ['train', 'val']:
    losses = torch.zeros(eval_iters)
    
    for k in range(eval_iters):
      X, Y = get_batch(split)

      with ctx:
        logits, loss = model(X, Y)
      
      losses[k] = loss.item()
    
    out[split] = losses.mean()
  # Use model.train() at the start of your training loop in PyTorch to set layers like Dropout and BatchNorm 
  # to their training behavior (e.g., updating statistics, dropping neurons), ensuring the model learns correctly,
  model.train()
  return out
# -----------------------------------------------------------------------------

# learning rate
# GPT-3 paper - B Details of Model Training
# we use cosine decay for learning rate down to 10% of its value, over 260 billion tokens 
# (after 260 billion tokens, training continues at 10% of the original learning rate). 
# learning rate decay scheduler (cosine with warmup)
def get_lr(it):

  # 1) Linear warm up for warmup_iterations
  if it < warmup_iters:
    return learning_rate * (it + 1) / (warmup_iters + 1)
  
  # 2) if it > lr_decay_iters return min learning rate
  if it > lr_decay_iters:
    return min_lr

  # 3) in between use cosine decay down to min learning rate
  decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
  assert 0 <= decay_ratio <= 1
  coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
  return min_lr + coeff * (learning_rate - min_lr)
# -----------------------------------------------------------------------------

# training loop
X, Y = get_batch('train')
t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
# In the context of Weights & Biases (W&B) and AI infrastructure,
# MFU stands for Model FLOPs Utilization. 
# It is a metric used to measure the efficiency of training large models (like Transformers/LLMs) 
# by calculating how much of the theoretical maximum compute capacity of the GPUs is actually being used for productive calculation. 
running_mfu = -1.0

while True:

  # determine and set the learning rate for this iteration
  lr = get_lr(iter_num) if decay_lr else learning_rate
  for param_group in optimizer.param_groups:
    param_group['lr'] = lr
  
  # evaluate the loss on train/val sets and write checkpoints
  if iter_num % eval_interval == 0 and master_process:
    losses = estimate_loss()
    print(f'step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}')
    
    if losses['val'] < best_val_loss or always_save_checkpoint:
      best_val_loss = losses['val']

      if iter_num > 0:
        checkpoint = {
          'model': raw_model.state_dict(),
          'optimizer': optimizer.state_dict(),
          'model_args': model_args,
          'iter_num': iter_num,
          'best_val_loss': best_val_loss,
        }
        print(f'saving checkpoint to {out_dir}')
        torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
    
    with open(log_file, 'a') as f:
      f.write(f'step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f} \n')
  
  if iter_num == 0 and eval_only:
    break

  # forward backward update, with optional gradient accumulation to simulate larger batch size
  # and using the GradScaler if data type is float16
  for micro_step in range(gradient_accumulation_steps):
    if ddp:
      # in DDP training we only need to sync gradients at the last micro step.
      # the official way to do this is with model.no_sync() context manager, but
      # I really dislike that this bloats the code and forces us to repeat code
      # looking at the source of that context manager, it just toggles this variable
      model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1) # basically we want to sync at the last step only
    
    with ctx:
      logits, loss = model(X, Y)
      # we have to scale the loss to account for gradient accumulation
      # because the gradients just add on each successive backward().
      # addition of these gradients corresponds to a SUM in the objective, but
      # instead of SUM we want a MEAN. Scale the loss here so it comes out right
      loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
    
    # immediately async prefetch next batch while model is doing the forward pass on the GPU
    X, Y = get_batch('train')

    # backward pass, with gradient scaling if training in fp16
    scaler.scale(loss).backward()
    
    if grad_clip != 0.0:
      scaler.unscale_(optimizer)
      # GPT-3 paper - B Details of Model Training
      # we clip the global norm of the gradient at 1.0
      torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()

    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1

    if iter_num % log_interval == 0 and master_process:
      # get loss as float. note: this is a CPU-GPU sync point
      # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
      lossf = loss.item() * gradient_accumulation_steps

      if local_iter_num >= 5: # let the training loop settle a bit
        mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
        running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
      
      print(f'iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%')
      with open(log_file, 'a') as f:
        f.write(f'iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%  \n')
    
    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        break
    
if ddp:
    destroy_process_group()