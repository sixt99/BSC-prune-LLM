import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from functions.make_plots import *
from functions.pruning_methods import *
from functions.initialization import *
from numpy.linalg import norm

# Define parameters
layer_name = "distilbert.transformer.layer.2.attention.k_lin.weight"
dataset_path = f"outputs/{layer_name}/output_a0.3_bs128.csv"
sort_by = "eval_matthews"
n_superpositions = 100

df = pd.read_csv(dataset_path)
idxs = np.argsort(df[sort_by])
layer =  df['layer'][0]
grid_shape = df['grid_size'][0][1:-1].split(',')
grid_shape = list(map(int, grid_shape))

# Best distributions
print('Best results:')
best_weights = np.zeros(grid_shape)
for i in idxs[-n_superpositions:]:
    print(df.loc[i][sort_by])
    string = df.loc[i]['pairs'][2:-2].split('),(')
    for x in string:
        i, j = list(map(int, x.split(',')))
        best_weights[i, j] += 1

# Worst distributions
print('Worst results:')
worst_weights = np.zeros(grid_shape)
for i in idxs[:n_superpositions]:
    print(df.loc[i][sort_by])
    string = df.loc[i]['pairs'][2:-2].split('),(')
    for x in string:
        i, j = list(map(int, x.split(',')))
        worst_weights[i, j] += 1

# Print correlation between best and worst
a = best_weights.flatten()
b = worst_weights.flatten()
print('Correlation: ', np.dot(a,b)/norm(a)/norm(b))

# Plot superposed best distributions
fig, axs = plt.subplots(2, 2, figsize=(12, 5))
bar = axs[0,0].imshow(best_weights)
fig.colorbar(bar)
axs[0,0].set_title('Superposed best configurations')

# Plot superposed worst distributions
bar = axs[0,1].imshow(worst_weights)
fig.colorbar(bar)
axs[0,1].set_title('Superposed worst configurations')

# Plot histogram of the selected column
axs[1,0].hist(df[sort_by], bins=70, color='skyblue', edgecolor='black')
axs[1,0].axvline(x=0.5294395294021531, color='red', linestyle='--')

# Plot real matrix
model = load_model()
tensor = model.state_dict()[layer_name]
axs[1,1].imshow(np.abs(tensor)>0.05)
axs[1,1].set_title('Real matrix')
plt.tight_layout()
plt.show()