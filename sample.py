'''
Sample from a trained model
python3 -m sample.py
'''

import os
from contextlib import nullcontext
import torch
from model import SolGPTConfig, SolGPT
from tokenizer.tokenizer import SolTokenizer

# -----------------------------------------------------------------------------

init_from = 'resume'
out_dir = 'out' # ignored if init_from is not 'resume'
start = "pragma solidity" # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
num_samples = 10 # number of samples to draw
max_new_tokens = 512 # number of tokens generated in each sample
temperature = 0.8 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 200 # retain only the top_k most likely tokens, clamp others to have 0 probability
seed = 1337

device = 'cpu'
torch.manual_seed(seed)
if torch.cuda.is_available():
  device = 'cuda'
  torch.cuda.manual_seed(seed)
  torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
  torch.backends.cuda.allow_tf32 = True # allow tf32 on cudnn

elif hasattr(torch.backends, 'mps') and torch.mps.is_available():
  device = 'mps'
  torch.mps.manual_seed(seed)

dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
compile = False # use PyTorch 2.0 to compile the model to be faster

ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = torch.amp.autocast(device_type=device, dtype=ptdtype) if device == 'cuda' else nullcontext()

# -----------------------------------------------------------------------------

if init_from == 'resume':
  # init from a model saved in a specific directory
  ckpt_path = os.path.join(out_dir, 'ckpt.pt')
  checkpoint = torch.load(ckpt_path, map_location=device)
  solGPT_config = SolGPTConfig(**checkpoint['model_args'])
  model = SolGPT(solGPT_config)
  state_dict = checkpoint['model']
  unwanted_prefix = '_orig_mod.'

  for k,v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
      state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

  model.load_state_dict(state_dict)

model.eval()
model.to(device)

if compile:
  model = torch.compile(model)

# -----------------------------------------------------------------------------

enc = SolTokenizer()
encode = lambda s: enc.encode(s)
decode = lambda l: enc.decode(l)
start_ids = encode(start)
x = torch.tensor(start_ids, dtype=torch.long, device=device)
x = x.unsqueeze(dim=0)

# run generation
with torch.no_grad():
  with ctx:
    for k in range(num_samples):
      y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
      print(decode(y[0].tolist()))
      print('-' * 7)
# -----------------------------------------------------------------------------