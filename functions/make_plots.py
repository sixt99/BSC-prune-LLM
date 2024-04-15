import numpy as np
import matplotlib.pyplot as plt


def plot_matrix_analysis(
    matrix, visualization_mode=None, show_gaussian=True, ignore_zeros=False
):
    # Comupte mean, standard deviation and limits
    if not ignore_zeros:
        mu = np.mean(matrix)
        sigma = np.std(matrix)
        median = np.median(matrix)
        lim_a = mu - sigma * 4
        lim_b = mu + sigma * 4
    else:
        mu = np.mean(matrix[matrix != 0])
        sigma = np.std(matrix[matrix != 0])
        median = np.median(matrix[matrix != 0])
        lim_a = mu - sigma * 4
        lim_b = mu + sigma * 4

    # Print some info about the matrix
    print(f"Min: {np.min(matrix)} Max: {np.max(matrix)}")
    print(f"Mean: {mu} Std: {sigma} Median: {median}")

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # Plot matrix
    axs[0].set_title("Matrix heatmap")
    if visualization_mode == "binary":
        axs[0].imshow(matrix != 0)
    elif visualization_mode == "std":
        heatmap = axs[0].imshow(matrix)
        cbar = fig.colorbar(heatmap, ax=axs[0])
        heatmap.set_clim(vmin=mu - sigma * 2, vmax=mu + sigma * 2)
    elif visualization_mode == "abs":
        axs[0].imshow(np.abs(matrix) > 0.05)
    elif visualization_mode is None:
        heatmap = axs[0].imshow(matrix)
        cbar = fig.colorbar(heatmap, ax=axs[0])

    # Plot histogram
    if not ignore_zeros:
        axs[1].hist(
            matrix.flatten(), bins=200, density=True, alpha=0.7, range=(lim_a, lim_b)
        )
    else:
        flat = matrix.flatten()
        axs[1].hist(
            flat[flat != 0], bins=200, density=True, alpha=0.7, range=(lim_a, lim_b)
        )
    axs[1].set_title("Histogram")

    # Compute Gaussian
    if show_gaussian and sigma != 0:
        x = np.linspace(lim_a, lim_b, 1000)
        gaussian = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(
            -((x - mu) ** 2) / (2 * sigma**2)
        )
        axs[1].plot(x, gaussian, color="red", label="Gaussian")

    plt.show()


def print_weight_matrices(
    model, visualization_mode=None, show_gaussian=True, ignore_zeros=False
):
    for x in model.state_dict().keys():
        # Retrieve weight matrix
        matrix = model.state_dict()[x].detach().numpy()

        # Print matrices only
        if len(matrix.shape) > 1:
            print(x)
            plot_matrix_analysis(
                matrix, visualization_mode, show_gaussian, ignore_zeros
            )
