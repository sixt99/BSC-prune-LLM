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

model = load_model()

# Parameters
max_n_rows = 100
area_percentage = 0.3
block_size = 128

for layer in model.state_dict():
    matrix = model.state_dict()[layer].cpu().detach().numpy()
    if len(matrix.shape) == 2 and layer.startswith("distilbert.transformer.layer"):
        file_name = f"outputs/{layer}/output_a{area_percentage}_bs{block_size}.csv"
        print(file_name)

        # If the folder is not found, create it
        if not os.path.exists(f"outputs/{layer}/"):
            os.makedirs(f"outputs/{layer}/")

        # If the file is not found, create it and write the features in a csv style
        if not os.path.exists(file_name):
            n_rows_so_far = 0 # There is no data so  far
            with open(file_name, 'w') as file:
                features = "eval_loss,eval_accuracy,eval_precision,eval_recall,eval_f1,eval_matthews,eval_runtime,eval_samples_per_second,eval_steps_per_second,area_percentage,block_size,grid_size,pairs,layer\n"
                file.write(features)
        # If the file is found, count the number of data rows we have so far
        else:
            with open(file_name, "rb") as f:
                n_rows_so_far = sum(1 for _ in f) - 1 # Do not count the row where the features are written

        # Iterate as many times as needed to reach max_n_rows of data
        for _ in range(n_rows_so_far, max_n_rows):
            model = load_model()
            trainer = Trainer(
                model=model,
                args=training_args,
                compute_metrics=compute_metrics,
            )

            tensor = model.state_dict()[layer]
            output = randomly_prune_blocks_by_area(tensor, area_percentage, block_size, verbose=True)
            evaluation = trainer.evaluate(tokenized_dataset)

            string = ''
            for x in evaluation.keys():
                string += str(evaluation[x]) + ","
            string += str(area_percentage) + ","
            string += str(block_size) + ","
            string += '"' + str(output['grid_size']) + '"' + ","
            string += '"' + str(output['pairs']) + '"' + ","
            string += '"' + layer + '"'
            string = string.replace(' ', '')

            with open(file_name, 'a') as f:
                print(string, file=f)