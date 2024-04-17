import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from functions.make_plots import *
from functions.pruning_methods import *
from functions.initialization import *

# Define parameters
dataset_path = "outputs/distilbert.transformer.layer.0.attention.q_lin.weight/output_a0.3_bs128.csv"
sort_by = "eval_matthews"
tail = 10

df = pd.read_csv(dataset_path)
idxs = np.argsort(df[sort_by])
layer =  df['layer'][0]
grid_shape = df['grid_size'][0][1:-1].split(',')
grid_shape = list(map(int, grid_shape))
weights = np.zeros(grid_shape)
for i in idxs[-tail:]:
    string = df.loc[i]['pairs'][2:-2].split('),(')
    for x in string:
        i, j = list(map(int, x.split(',')))
        weights[i, j] += 1

# Make the plot
fig, axs = plt.subplots(1, 2, figsize=(12, 5))
bar = axs[0].imshow(weights)
fig.colorbar(bar)
model = load_model()
tensor = model.state_dict()['distilbert.transformer.layer.0.attention.q_lin.weight']
axs[1].imshow(np.abs(tensor)>0.05)
plt.tight_layout()
plt.show()