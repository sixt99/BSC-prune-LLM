import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from functions.make_plots import *
from functions.pruning_methods import *
from functions.initialization import *
from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments
)
from datasets import load_dataset

def preprocess_function(examples):
    return tokenizer(examples["sentence"], truncation=True, padding=True)

# Tokenizer and dataset
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
dataset = load_dataset("glue", "cola")
tokenized_dataset = dataset.map(preprocess_function, batched=True)['validation']

training_args = TrainingArguments(
    per_device_eval_batch_size=100,
    output_dir="./results",
)

# Define parameters
layer_name = "distilbert.transformer.layer.2.ffn.lin1.weight"
dataset_path = f"outputs/{layer_name}/output_a0.3_bs128.csv"
sort_by = "eval_matthews"

df = pd.read_csv(dataset_path)
idxs = np.argsort(df[sort_by])
grid_shape = df['grid_size'][0][1:-1].split(',')
grid_shape = list(map(int, grid_shape))

base = {'eval_loss': 0.8195775151252747, 'eval_accuracy': 0.8092042186001918, 'eval_precision': 0.8246268656716418, 'eval_recall': 0.9195561719833565, 'eval_f1': 0.8695081967213115, 'eval_matthews': 0.5294395294021531, 'eval_runtime': 4.6418, 'eval_samples_per_second': 224.696, 'eval_steps_per_second': 2.37}

evaluations = []
for n_superpositions in [23]:
    print(n_superpositions)
    # Best distributions
    #print('Best results:')
    best_weights = np.zeros(grid_shape)
    for iter in idxs[-n_superpositions:]:
        #print(df.loc[iter][sort_by])
        string = df.loc[iter]['pairs'][2:-2].split('),(')
        for x in string:
            i, j = list(map(int, x.split(',')))
            best_weights[i, j] += 1

    # Worst distributions
    #print('Worst results:')
    worst_weights = np.zeros(grid_shape)
    for iter in idxs[:n_superpositions]:
        #print(df.loc[iter][sort_by])
        string = df.loc[iter]['pairs'][2:-2].split('),(')
        for x in string:
            i, j = list(map(int, x.split(',')))
            worst_weights[i, j] += 1

    best_pattern = np.argsort(best_weights, axis=None)[-44:]
    best_pattern = [(j//grid_shape[1], j%grid_shape[1]) for j in best_pattern]

    model = load_model()
    trainer = Trainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
    )

    tensor = model.state_dict()[layer_name]
    prune_by_pairs(tensor, best_pattern, 128)
    evaluation = trainer.evaluate(tokenized_dataset)
    print(evaluation)
    evaluations.append(evaluation[sort_by])

    #plot_matrix_analysis(tensor.cpu().detach().numpy(), visualization_mode='abs')

for x in evaluations:
    print(evaluations)

plot_matrix_analysis(tensor.cpu().detach().numpy(), visualization_mode='abs')