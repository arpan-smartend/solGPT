import tiktoken
from .solidity_keywords import solidity_keywords


class SolTokenizer:
  '''This call extends the base gpt2 tokenizer by adding solidity keywords'''

  def __init__(self):
    gpt2_base = tiktoken.get_encoding('gpt2')
    special_tokens = []

    for keyword in solidity_keywords:
      encoded_keyword = gpt2_base.encode(keyword)

      # if the encoding of any keyword is > 2, it means it is not in the gpt2 vocab and we must extend it
      if len(encoded_keyword) > 1:
        special_tokens.append(keyword)

    self.special_token_set = set(special_tokens).union(set(gpt2_base.special_tokens_set))
    special_token_ids = {token: gpt2_base.n_vocab + i for i, token in enumerate(special_tokens)}
    self.tokenizer = tiktoken.Encoding(
      name='sol_gpt2',
      pat_str=gpt2_base._pat_str,
      mergeable_ranks=gpt2_base._mergeable_ranks,
      special_tokens={
        **gpt2_base._special_tokens,
        **special_token_ids
      }
    )

  def encode(self, text, num_threads=8):
    # text can be either a string or a list of strings
    if isinstance(text, str):
      ids = self.tokenizer.encode(text, allowed_special=self.special_token_set)
      ids.append(self.tokenizer.eot_token)
    
    elif isinstance(text, list):
      ids = self.tokenizer.encode_batch(text, allowed_special=self.special_token_set, num_threads=num_threads)
      ids.append([self.tokenizer.eot_token])
    
    else:
      raise ValueError(f"Invalid input type: {type(text)}")

    return ids 
  
  def decode(self, ids):
    return self.tokenizer.decode(ids)
  
  def get_vocab_size(self):
    return self.tokenizer.n_vocab

