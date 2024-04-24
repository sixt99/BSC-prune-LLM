from functions.make_plots import *
from functions.pruning_methods import *
from functions.initialization import *
from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments
)
from datasets import load_dataset
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import pandas as pd
import os

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

# Parameters
area_percentage = 0.3
block_size = 128
model = load_model()
sort_by = "eval_matthews"

trainer = Trainer(
    model=model,
    args=training_args,
    compute_metrics=compute_metrics,
)

# Iterate over the layers
for layer in model.state_dict():
    matrix = model.state_dict()[layer]
    if len(matrix.shape) == 2: # If the weights are matrices
        file_name = f"outputs/{layer}/output_a{area_percentage}_bs{block_size}.csv"
        if not os.path.exists(file_name): # If we don't have information about this layer, don't prune it
            continue

        if "ffn" in layer: # We do not modify these matrices
            continue
        
        # Get the best distribution of blocks according to sort_by metric
        df = pd.read_csv(file_name)
        best_idx = np.argmax(df[sort_by])
        string = df.loc[best_idx]['pairs'][2:-2].split('),(')        
        pairs = [list(map(int, x.split(','))) for x in string]

        # Prune the matrix with the given blocks
        output = prune_by_pairs(matrix, pairs, block_size, verbose=True)

evaluation = trainer.evaluate(tokenized_dataset)
print(evaluation)

print_weight_matrices(model.cpu(), visualization_mode='abs')

output_dir = "/Users/sixteoriolllenassegura/prune_llm/trainings/take_best_matrices_a0.3_bz128"
model.save_pretrained(output_dir)