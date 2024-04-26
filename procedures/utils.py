import torch
import time
import random
import numpy as np
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer
)
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset

def load_model(model_path = "model"):
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    return model

def load_trainer(model):
    training_args = TrainingArguments(
        per_device_eval_batch_size=100,
        output_dir="./results"
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics
    )
    return trainer

def load_tokenizer(tokenizer_path = "model"):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    return tokenizer

def load_tokenized_data(tokenizer, dataset_name = "cola"):
    dataset = load_dataset("glue", dataset_name)
    def preprocess_function(examples):
        return tokenizer(examples["sentence"], truncation=True, padding=True)
    tokenized_dataset = dataset.map(preprocess_function, batched=True)
    return tokenized_dataset

def initialize(model_path = "model", tokenizer_path = "model", dataset_name = "cola"):
    model = load_model(model_path)
    trainer = load_trainer(model)
    tokenizer = load_tokenizer(tokenizer_path)
    tokenized_dataset = load_tokenized_data(tokenizer, dataset_name)
    return model, trainer, tokenizer, tokenized_dataset

def string2pairs(string):
    string = string[2:-2].split('),(')
    return [list(map(int, x.split(','))) for x in string]

def block_pruning2string(evaluation, area_percentage, block_size, output, layer):
    string = ''
    for x in evaluation.keys():
        string += str(evaluation[x]) + ","
    string += str(area_percentage) + ","
    string += str(block_size) + ","
    string += '"' + str(output['grid_size']) + '"' + ","
    string += '"' + str(output['pairs']) + '"' + ","
    string += '"' + layer + '"'
    string = string.replace(' ', '')

    return string

def global_block_pruning2string(evaluation, area_percentage, block_size, pairs):
    string = ''
    for x in evaluation.keys():
        string += str(evaluation[x]) + ","
    string += str(area_percentage) + ","
    string += str(block_size) + ","
    string += '"' + str(pairs) + '"'
    string = string.replace(' ', '')

    return string

def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    tp = np.sum(np.logical_and(preds, labels))
    tn = np.sum(np.logical_and(preds == 0, labels == 0))
    fp = np.sum(np.logical_and(preds, labels == 0))
    fn = np.sum(np.logical_and(preds == 0, labels))
    acc = np.sum(labels == preds) / len(labels)
    precision = 0 if tp + fp == 0 else tp / (tp + fp)
    recall = 0 if tp + fn == 0 else tp / (tp + fn)
    f1 = 0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
    mcc = 0 if (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) == 0 else (tp * tn - fp * fn) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "matthews": mcc,
    }

def create_mask(pairs, size, block_size):
    mask = torch.zeros(size, dtype=torch.bool)
    for pair in pairs:
        i, j = pair
        mask[i*block_size:(i+1)*block_size,j*block_size:(j+1)*block_size].fill_(1)
    return mask


def randomly_prune_blocks(matrix, n_blocks, block_size):
    random.seed(time.time())
    grid_size_x = matrix.shape[0] // block_size
    grid_size_y = matrix.shape[1] // block_size

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
        block = matrix[
            block_size * i : block_size * (i + 1), block_size * j : block_size * (j + 1)
        ]
        block.fill_(0)

def randomly_prune_blocks_with_probabilities(matrix, n_blocks, block_size):
    random.seed(time.time())
    grid_size_x = matrix.shape[0] // block_size
    grid_size_y = matrix.shape[1] // block_size

    # If the blocks do not fit, do nothing
    if n_blocks > grid_size_x * grid_size_y:
        return

    # Compute the corresponding weight of a block being removed
    weights = []
    idxs = []
    for i in range(grid_size_x):
        for j in range(grid_size_y):
            idxs.append((i, j))
            block = matrix[
                block_size * i : block_size * (i + 1),
                block_size * j : block_size * (j + 1),
            ]
            weights.append(torch.sum(block**2))

    probabilities = torch.tensor(weights)
    probabilities = -probabilities
    probabilities = torch.softmax(probabilities, dim=0)
    probabilities = probabilities.tolist()

    dictionary = dict(zip(idxs, probabilities))

    pairs = []
    while len(pairs) < n_blocks:
        pair = random.choices(
            list(dictionary.keys()), weights=list(dictionary.values()), k=1
        )[0]
        dictionary.pop(pair)
        pairs.append(pair)

    # Remove the block associated to each unique pair (i,j)
    for pair in pairs:
        i = pair[0]
        j = pair[1]
        block = matrix[
            block_size * i : block_size * (i + 1), block_size * j : block_size * (j + 1)
        ]
        block.fill_(0)

# TODO this can be greatly improved by considering random_numbers = random.sample(range(0, 11), 5)
def randomly_prune_blocks_by_area(matrix, area_percentage, block_size, verbose = False):
    random.seed(time.time())
    grid_size_x = matrix.shape[0] // block_size
    grid_size_y = matrix.shape[1] // block_size

    # Find n_blocks unique pairs (i,j)
    pairs = set()
    while (len(pairs) * block_size**2 / (matrix.shape[0] * matrix.shape[1]) < area_percentage):
        i = random.randint(0, grid_size_x - 1)
        j = random.randint(0, grid_size_y - 1)
        pairs.add((i, j))

    # Remove the block associated to each unique pair (i,j)
    for pair in pairs:
        i, j = pair
        block = matrix[
            block_size * i : block_size * (i + 1), block_size * j : block_size * (j + 1)
        ]
        block.fill_(0)

    if verbose:
        output = {}
        output['original_size'] = matrix.shape
        output['grid_size'] = (grid_size_x, grid_size_y)
        output['pairs'] = pairs
        return output

def prune_by_pairs(matrix, pairs, block_size, verbose = False):
    grid_size_x = matrix.shape[0] // block_size
    grid_size_y = matrix.shape[1] // block_size

    # Remove the block associated to each unique pair (i,j)
    for pair in pairs:
        i, j = pair
        block = matrix[
            block_size * i : block_size * (i + 1), block_size * j : block_size * (j + 1)
        ]
        block.fill_(0)

    if verbose:
        output = {}
        output['original_size'] = matrix.shape
        output['grid_size'] = (grid_size_x, grid_size_y)
        output['pairs'] = pairs
        return output

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


def print_weight_matrices(model, visualization_mode='abs', show_gaussian=True, ignore_zeros=False):
    for x in model.state_dict().keys():
        # Retrieve weight matrix
        matrix = model.state_dict()[x].detach().numpy()

        # Print matrices only
        if len(matrix.shape) > 1:
            print(x)
            plot_matrix_analysis(
                matrix, visualization_mode, show_gaussian, ignore_zeros
            )