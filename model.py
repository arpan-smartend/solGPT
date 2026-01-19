import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F
# -----------------------------------------------------------------------------

class LayerNorm(nn.Module):
  '''LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False '''

  def __init__(self, ndim, bias):
    super().__init__()
    self.weight = nn.Parameter(torch.ones(ndim))
    self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
  
  def forward(self, input):
    return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)
# -----------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
  '''The input consists of queries and keys of dimension dk, and values of dimension dv. We compute the dot products of the
  query with all keys, divide each by √dk, and apply a softmax function to obtain the weights on the
  values.
  In practice, we compute the attention function on a set of queries simultaneously, packed together
  into a matrix Q. The keys and values are also packed together into matrices K and V . We compute
  the matrix of outputs as:
  Attention(Q,K,V ) = softmax(QKT/√dk)V'''

  def __init__(self, config):
    super().__init__()
    assert config.n_embd % config.n_head == 0

    # key, query, value projections for all heads, but in a batch
    self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)

    # output projection
    self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

    # regularization
    self.attn_dropout = nn.Dropout(config.dropout)
    self.resid_dropout = nn.Dropout(config.dropout)

    self.n_head = config.n_head
    self.n_embd = config.n_embd
    self.dropout = config.dropout

  def forward(self, x):
    B, T, C = x.size() # batch size, sequence length and embedding dimensionality (n_embd)

    # calculate query, key, values for all heads in batch and move head forward to match the batch dim
    # nh is 'number of heads', hs is 'head size' and C (number of channels) = nh * hs
    # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=C=768 channels in the Transformer
    # We have tokens lined up in a sequence of 1024 tokens and each of them emit 3 vectors query, key and value
    # So in the line below what happens is we multiply queries keys and values to get the attention or affinities
    qkv = self.c_attn(x)
    q, k, v = qkv.split(self.n_embd, dim=2) # we split into q, k, v

    # Here we have included number of heads as a batch dimension so we do not have the explicit for loop to concat the output of each heads
    # so B and nh together now kind of act like a batch/batches
    q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
    k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
    v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

    # attention materializes the large (T, T) matrix for all queries and key
    # is_causal = True -> This ensures that a token at a given position in a sequence can only attend to tokens that 
    # appeared at or before its current position, and not to any future tokens
    y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
    y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side, actual concatenation operation is happening here

    # output projection through the residual dropout
    y = self.resid_dropout(self.c_proj(y))
    return y
# -----------------------------------------------------------------------------

class MLP(nn.Module): # feed- forward - FFN(x) = max(0,xW1 + b1)W2 + b2 
  '''In addition to attention sub-layers, each of the layers in our encoder and decoder contains a fully
  connected feed-forward network, which is applied to each position separately and identically. This
  consists of two linear transformations with a ReLU activation in between'''

  def __init__(self, config):
    super().__init__()

    # we always have the feedforward layer four times the size of the embd layer, dff = 4 * dmodel)
    self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
    self.gelu = nn.GELU(approximate='tanh') # GPT2 uses the approximate version
    self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
    self.dropout = nn.Dropout(config.dropout)
  
  def forward(self, x):
    x = self.c_fc(x)
    x = self.gelu(x)
    x = self.c_proj(x)
    x = self.dropout(x)
    return x
# -----------------------------------------------------------------------------

class Block(nn.Module):
  '''One transformer block'''
  def __init__(self, config):
    super().__init__()
    self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
    self.attn = CausalSelfAttention(config) # communication is done by self attention, tokens communicate with each other/ aggregation fn /reduce fn
    self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
    self.mlp = MLP(config) # Feed-Forward Networks - MLP works on every single token, essentially a map function
   
  def forward(self, x):
    x = x + self.attn(self.ln_1(x)) # x + -> residual connection
    x = x + self.mlp(self.ln_2(x))  # x + -> residual connection
    return x
# -----------------------------------------------------------------------------

@dataclass
class GPTConfig:
  block_size: int = 1024        # 1024   -> max sequence length
  vocab_size: int = 50496       # 50442  -> 256: raw byte tokens + 50,000 merges done by openAI + 1 special token(<|endoftext|>) + 185 solidity tokens, padded up to nearest multiple of 64 for efficiency
  n_layer: int = 8              # 12     -> number of layers
  n_head: int = 8               # 12     -> number of heads
  n_embd: int = 768             # 768    -> embedding dimension
  dropout: float = 0.0
  bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
# -----------------------------------------------------------------------------

class SolGPT(nn.Module):

  def __init__(self, config):
    super().__init__()
    assert config.vocab_size is not None
    assert config.block_size is not None

    self.config = config
    self.transformer = nn.ModuleDict(dict(
      wte = nn.Embedding(config.vocab_size, config.n_embd), # token embedding
      wpe = nn.Embedding(config.block_size, config.n_embd), # position embedding
      drop = nn.Dropout(config.dropout),
      h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]), # no of transformer blocks
      ln_f = nn.LayerNorm(config.n_embd) # GPT2-paper: final layer norm after the final self-attention block
    ))
    self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False) # n_embd to vocab size - final linear layer

    # weight sharing scheme
    self.transformer.wte.weight = self.lm_head.weight

    # init all weights
    self.apply(self._init_weights) 

    # apply special scaled init to the residual projections, per GPT-2 paper
    for pn, p in self.named_parameters():
      if pn.endswith('c_proj.weight'):
        torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer)) # residual init we are multiplying 2 because we have 2 residual connections self.attn and self.mlp

    # report number of parameters
    print('number of parameters: %.2fM' % (self.get_num_params()/1e6,))
  
  def get_num_params(self, non_embedding=True):
    '''
    Return the number of parameters in the model.
    For non-embedding count (default), the position embeddings get subtracted. The token embeddings would too, except due to the parameter 
    sharing these params are actually used as weights in the final layer, so we include them.
    '''
    n_params = sum(p.numel() for p in self.parameters())

    if non_embedding:
      n_params -= self.transformer.wpe.weight.numel()
    
    return n_params
  
  def _init_weights(self, module):

    if isinstance(module, nn.Linear):
      torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

      if module.bias is not None:
        torch.nn.init.zeros_(module.bias)
      
    elif isinstance(module, nn.Embedding):
      torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

  def forward(self, idx, targets=None):
    device = idx.device
    # idx is of shape (B, T)
    B, T = idx.size()
    assert T <= self.config.block_size, f'Cannot forward sequence of length {T}, block size is only {self.config.block_size}'

    # forward the token and positional embeddings
    pos = torch.arange(0, T, dtype=torch.long, device=device) # shape (T)
    pos_emb = self.transformer.wpe(pos) # shape (T, n_embd)
    tok_emb = self.transformer.wte(idx) # shape (B, T, n_embd)
    x = self.transformer.drop(tok_emb + pos_emb) # shape (B, T, n_embd)

    # forward to the block of the transformer
    for block in self.transformer.h:
       x = block(x)

    # forward to the final layer norm and classifier
    x = self.transformer.ln_f(x) # (B, T, vocab_size)

    if targets is not None:
      # if we are given some desired targets also calculate the loss
      logits = self.lm_head(x) # (B, T, vocab_size)
      loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1) # reduction is by default mean

    else:
      # inference-time mini-optimization: only forward the lm_head on the very last position
      logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim (B, 1, vocab_size) else it will be (B, vocab_size)
      loss = None

    return logits, loss
  
  # have to explore more on this, I cannot understand...need to read the paper in more detail
  def estimate_mfu(self, fwdbwd_per_iter, dt):
    ''' estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS '''
    # first estimate the number of flops we do per iteration.
    # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
    N = self.get_num_params()
    cfg = self.config
    L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
    flops_per_token = 6*N + 12*L*H*Q*T
    flops_per_fwdbwd = flops_per_token * T
    flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
    # express our flops throughput as ratio of A100 bfloat16 peak flops
    flops_achieved = flops_per_iter * (1.0/dt) # per second
    flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
    mfu = flops_achieved / flops_promised
    return mfu
  
  @torch.no_grad()
  def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
    '''
    Take a conditioning sequence of indices idx (LongTensor of shape (B, T)) and complete
    the sequence max_new_tokens times, feeding the predictions back into the model each time.
    Most likely you'll want to make sure to be in model.eval() mode of operation for this.
    '''

    for _ in range(max_new_tokens):
      # if the sequence context is growing too long we must crop it at block_size
      idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]

      # forward the model to get the logits for the index in the sequence
      logits, _ = self(idx_cond)

      # pluck the logits at the final step and scale by desired temperature
      logits = logits[:, -1, :] / temperature

      # optionally crop the logits to only the top k options
      if top_k is not None:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < v[:, [-1]]] = -float('Inf')

      # apply softmax to convert logits to (normalized) probabilities
      probs = F.softmax(logits, dim=-1)

      # sample from the distribution
      idx_next = torch.multinomial(probs, num_samples=1)

      # append sampled index to the running sequence and continue
      idx = torch.cat((idx, idx_next), dim=1)
    
    return idx
# -----------------------------------------------------------------------------



    

