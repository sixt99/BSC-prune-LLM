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
repetitions = 500
area_percentage = 0.3
block_size = 128
layer = 'distilbert.transformer.layer.3.ffn.lin1.weight'

file_name = f"outputs/{layer}/output_a{area_percentage}_bs{block_size}.csv"
for iter in range(repetitions):
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