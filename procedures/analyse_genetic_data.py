import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import *
for i in range(10):
    # Define parameters
    path = f"/Users/sixteoriolllenassegura/prune_llm/procedures/genetic_outputs/attempt_22/generation_{i}.csv"
    sort_by = "eval_matthews"

    df = pd.read_csv(path)
    idxs = np.argsort(df[sort_by])

    # Plot histogram of the selected column
    plt.hist(df[sort_by], bins=4, color='skyblue', edgecolor='black', density=True)

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
    plt.plot(x, gaussian, color="red", label="Gaussian")
    plt.show()