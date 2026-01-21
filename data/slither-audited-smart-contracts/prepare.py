'''
run this file to create the training and validation files in your local machine:
python -m data.slither-audited-smart-contracts.prepare
'''

# --------------------------------------------------------------------------------------------------------
import re
import os
from tqdm import tqdm
import numpy as np
from datasets import load_dataset
from tokenizer.tokenizer import SolTokenizer
# --------------------------------------------------------------------------------------------------------

num_proc = max(1, os.cpu_count() // 2)
sol_tokenizer = SolTokenizer()

if __name__ == '__main__':
  dataset = load_dataset('mwritescode/slither-audited-smart-contracts', 'all-multilabel', num_proc=num_proc)
  dataset = dataset.select_columns(['source_code'])
  # split dataset into train and val
  split_dataset = dataset["train"].train_test_split(test_size=0.010, seed=1337, shuffle=True)
  split_dataset['val'] = split_dataset.pop('test')

  # format source code
  def format_source_code(dataset_col):
    formatted = re.sub(r'/\*[\s\S]*?\*/', '', dataset_col['source_code'])
    formatted = re.sub(r'//[^\n\r]*', '', formatted)
    formatted = re.sub(r'[\n\r]{2,}', '\n', formatted)
    return {'source_code': formatted}

  # process the dataset
  def process(dataset_col):
    ids = sol_tokenizer.encode(dataset_col['source_code'])
    out = {'ids': ids, 'len': len(ids)}
    return out
  
  # tokenize the dataset
  tokenized_dataset = split_dataset.map(
    format_source_code,
    desc="formatting the splits",
    num_proc=num_proc,
  ).map(
    process, 
    desc="tokenizing the splits",
    remove_columns=['source_code'],
    num_proc=num_proc
  )

  # concatenate all the ids in each dataset into one large file we can use for training
  for split, dset in tokenized_dataset.items():
    total_token_length = np.sum(dset['len'],  dtype=np.uint64)
    filename = os.path.join(os.path.dirname(__file__), f'{split}.bin')
    print(f'Total token length in {filename} is {total_token_length}')
    dtype = np.uint16
    arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(total_token_length,))
    total_batches = 1024

    idx = 0
    for batch_idx in tqdm(range(total_batches), desc=f'writing {filename}'):
      batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
      array_batch = np.concatenate(batch['ids'])
      arr[idx: idx + len(array_batch)] = array_batch
      idx += len(array_batch)

    arr.flush() # complete writing the file

'''
train.bin has 1269618024 tokens and the filesize is 2.54 gb
val.bin has 12534342 tokens and the filesize is 25.1 mb
'''

    