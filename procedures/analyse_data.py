import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import *
import os

#67
attempt = 122
sort_by = ["eval_matthews", "pruned_area"]
attempt_path = f"/Users/sixteoriolllenassegura/prune_llm/marenostrum_layerwise/attempt_{attempt}/"

folders = [x for x in os.listdir(attempt_path) if 'transformer' in x]
idxs = [int(x.split('_')[0]) for x in folders]
idxs = np.argsort(idxs)
folders = [folders[x] for x in idxs]

x_values = []
y_values = []

for layer in folders:
    layer_path = attempt_path + f"{layer}"
    n_generations = len(os.listdir(layer_path)) - 1
    for iter in range(n_generations):
        generation_path = layer_path + f"/generation_{iter}.csv"
        df_it = pd.read_csv(generation_path)
        if iter == 0:
            df = df_it
            continue
        df = pd.concat([df, df_it], axis = 0).reset_index(drop=True)

    x = df[sort_by[0]]
    y = df[sort_by[1]]
    hist, x_edges, y_edges, _ = plt.hist2d(x, y, bins=200, cmap='viridis', range=[[0.4, 0.85], [0, 1]])
    max_idx = np.argmax(hist)
    max_idx_2d = np.unravel_index(max_idx, hist.shape)

    plt.xlabel(sort_by[0])
    plt.ylabel(sort_by[1])
    plt.colorbar(label='Frequency')

    plt.title(layer)

    x_values.append(x_edges[max_idx_2d[0]])
    y_values.append(y_edges[max_idx_2d[1]])

    # Draw a line between the two points
    plt.plot(x_values, y_values, color='red', linewidth=0.5)

    plt.show()
