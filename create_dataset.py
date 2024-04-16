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

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
dataset = load_dataset("glue", "cola")
tokenized_dataset = dataset.map(preprocess_function, batched=True)["validation"]

training_args = TrainingArguments(
    per_device_eval_batch_size=100,
    output_dir="./results",
)

area_percentage = 0.1
block_size = 128
file_name = f"outputs/output_a{area_percentage}_bs{block_size}.csv"
for iter in range(400):
    model = load_model()
    trainer = Trainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
    )

    tensor = model.state_dict()['distilbert.transformer.layer.0.attention.q_lin.weight']

    output = randomly_prune_blocks_by_area(tensor, area_percentage, block_size, verbose=True)
    evaluation = trainer.evaluate(tokenized_dataset)
    evaluation['area_percentage'] = area_percentage
    evaluation['block_size'] = block_size
    evaluation['grid_size'] = output['grid_size']
    evaluation['pairs'] = output['pairs']

    with open(file_name, 'a') as f:
        print(evaluation, file=f)