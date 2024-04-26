import pandas as pd
import numpy as np
from utils import *

# Define parameters
layer_name = "distilbert.transformer.layer.2.ffn.lin1.weight"
dataset_path = f"outputs/{layer_name}/output_a0.3_bs128.csv"
sort_by = "eval_matthews"
n_superpositions = 23

df = pd.read_csv(dataset_path)
idxs = np.argsort(df[sort_by])
grid_shape = df['grid_size'][0][1:-1].split(',')
grid_shape = list(map(int, grid_shape))

# Best distributions
print('Best results:')
best_weights = np.zeros(grid_shape)
for iter in idxs[-n_superpositions:]:
    #print(df.loc[iter][sort_by])
    string = df.loc[iter]['pairs'][2:-2].split('),(')
    for x in string:
        i, j = list(map(int, x.split(',')))
        best_weights[i, j] += 1

# Worst distributions
print('Worst results:')
worst_weights = np.zeros(grid_shape)
for iter in idxs[:n_superpositions]:
    #print(df.loc[iter][sort_by])
    string = df.loc[iter]['pairs'][2:-2].split('),(')
    for x in string:
        i, j = list(map(int, x.split(',')))
        worst_weights[i, j] += 1

best_pattern = np.argsort(best_weights, axis=None)[-44:]
best_pattern = [(j//grid_shape[1], j%grid_shape[1]) for j in best_pattern]

model, tokenizer, tokenized_dataset, trainer = initialize()
tokenized_dataset = tokenized_dataset['validation']
tensor = model.state_dict()[layer_name]
prune_by_pairs(tensor, best_pattern, 128)
evaluation = trainer.evaluate(tokenized_dataset)
print(evaluation)

plot_matrix_analysis(tensor.cpu().detach().numpy(), visualization_mode='abs')