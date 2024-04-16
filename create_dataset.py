from functions.make_plots import *
from functions.pruning_methods import *
from functions.initialization import *
from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments
)
import pickle
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

def preprocess_function(examples):
    return tokenizer(examples["sentence"], truncation=True, padding=True)

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
with open("data/cola.pkl", 'rb') as file:
    dataset = pickle.load(file)
tokenized_dataset = dataset.map(preprocess_function, batched=True)["validation"]

training_args = TrainingArguments(
    per_device_eval_batch_size=100,
    output_dir="./results",
)

# Parameters
repetitions = 5
area_percentage = 0.1
block_size = 128
layer = 'distilbert.transformer.layer.0.attention.q_lin.weight'

file_name = f"outputs/output_a{area_percentage}_bs{block_size}.csv"
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