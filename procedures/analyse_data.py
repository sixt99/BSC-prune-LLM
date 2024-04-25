import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from functions.make_plots import *
from functions.pruning_methods import *
from functions.initialization import *
from numpy.linalg import norm

# Define parameters
layer_name = "distilbert.transformer.layer.2.ffn.lin1.weight"
dataset_path = f"outputs/{layer_name}/output_a0.3_bs128.csv"
sort_by = "eval_matthews"
n_superpositions = 10

df = pd.read_csv(dataset_path)
idxs = np.argsort(df[sort_by])
grid_shape = df['grid_size'][0][1:-1].split(',')
grid_shape = list(map(int, grid_shape))

base = {'eval_loss': 0.8195775151252747, 'eval_accuracy': 0.8092042186001918, 'eval_precision': 0.8246268656716418, 'eval_recall': 0.9195561719833565, 'eval_f1': 0.8695081967213115, 'eval_matthews': 0.5294395294021531, 'eval_runtime': 4.6418, 'eval_samples_per_second': 224.696, 'eval_steps_per_second': 2.37}

# Best distributions
print('Best results:')
best_weights = np.zeros(grid_shape)
for iter in idxs[-n_superpositions:]:
    print(df.loc[iter][sort_by])
    string = df.loc[iter]['pairs'][2:-2].split('),(')
    for x in string:
        i, j = list(map(int, x.split(',')))
        best_weights[i, j] += 1

# Worst distributions
print('Worst results:')
worst_weights = np.zeros(grid_shape)
for iter in idxs[:n_superpositions]:
    print(df.loc[iter][sort_by])
    string = df.loc[iter]['pairs'][2:-2].split('),(')
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
axs[1,0].axvline(x=base[sort_by], color='red', linestyle='--')
axs[1,0].hist(df[sort_by], bins=50, color='skyblue', edgecolor='black', density=True)

# Plot gaussian on top of histogram
lim_a = np.min(df[sort_by])
lim_b = np.max(df[sort_by])
mu = np.mean(df[sort_by])
sigma = np.std(df[sort_by])
median = np.median(df[sort_by])

x = np.linspace(lim_a, lim_b, 1000)
gaussian = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(
    -((x - mu) ** 2) / (2 * sigma**2)
)
axs[1,0].plot(x, gaussian, color="red", label="Gaussian")

# Plot real matrix
model = load_model()
tensor = model.state_dict()[layer_name]
axs[1,1].imshow(np.abs(tensor)>0.05)
axs[1,1].set_title('Real matrix')
plt.tight_layout()
plt.show()