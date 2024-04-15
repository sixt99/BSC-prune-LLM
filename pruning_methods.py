import torch
import time
import numpy as np
import matplotlib.pyplot as plt
import scipy
import re
import tensorflow as tf
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset, Dataset
import random

def randomly_prune_blocks(matrix, n_blocks, block_size):
  random.seed(time.time())
  grid_size_x = matrix.shape[0]//block_size
  grid_size_y = matrix.shape[1]//block_size

  # If the blocks do not fit, do nothing
  if n_blocks > grid_size_x * grid_size_y:
    return

  # Find n_blocks unique pairs (i,j)
  pairs = set()
  while len(pairs) < n_blocks:
    i = random.randint(0, grid_size_x - 1)
    j = random.randint(0, grid_size_y - 1)
    pairs.add((i, j))

  # Remove the block associated to each unique pair (i,j)
  for pair in pairs:
    i = pair[0]
    j = pair[1]
    block = matrix[block_size * i : block_size * (i+1), block_size * j : block_size * (j+1)]
    block.fill_(0)

def randomly_prune_blocks_by_area(matrix, area_percentage, block_size):
  random.seed(time.time())
  grid_size_x = matrix.shape[0]//block_size
  grid_size_y = matrix.shape[1]//block_size

  # Find n_blocks unique pairs (i,j)
  pairs = set()
  while len(pairs)*block_size**2/(matrix.shape[0]*matrix.shape[1]) < area_percentage:
    i = random.randint(0, grid_size_x - 1)
    j = random.randint(0, grid_size_y - 1)
    pairs.add((i, j))

  # Remove the block associated to each unique pair (i,j)
  for pair in pairs:
    i = pair[0]
    j = pair[1]
    block = matrix[block_size * i : block_size * (i+1), block_size * j : block_size * (j+1)]
    block.fill_(0)

def prune_blocks_with_probabilities(matrix, n_blocks, block_size):
  random.seed(time.time())
  grid_size_x = matrix.shape[0]//block_size
  grid_size_y = matrix.shape[1]//block_size

  # If the blocks do not fit, do nothing
  if n_blocks > grid_size_x * grid_size_y:
    return

  # Compute the corresponding weight of a block being removed
  weights = []
  idxs = []
  for i in range(grid_size_x):
    for j in range(grid_size_y):
      idxs.append((i,j))
      block = matrix[block_size * i : block_size * (i+1), block_size * j : block_size * (j+1)]
      weights.append(torch.sum(block**2))

  probabilities = torch.tensor(weights)
  probabilities = -probabilities
  probabilities = torch.softmax(probabilities, dim=0)
  probabilities = probabilities.tolist()

  dictionary = dict(zip(idxs, probabilities))

  pairs = []
  while len(pairs) < n_blocks:
    pair = random.choices(list(dictionary.keys()), weights=list(dictionary.values()), k=1)[0]
    dictionary.pop(pair)
    pairs.append(pair)

  # Remove the block associated to each unique pair (i,j)
  for pair in pairs:
    i = pair[0]
    j = pair[1]
    block = matrix[block_size * i : block_size * (i+1), block_size * j : block_size * (j+1)]
    block.fill_(0)