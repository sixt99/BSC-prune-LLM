from functions.make_plots import *
from functions.pruning_methods import *
from functions.initialization import *
from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments
)
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from datasets import load_dataset

def preprocess_function(examples):
    return tokenizer(examples["sentence"], truncation=True, padding=True)

path = "model"
model = AutoModelForSequenceClassification.from_pretrained(path, num_labels=2)
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
dataset = load_dataset("glue", "cola")
encoded_dataset = dataset.map(preprocess_function, batched=True)['train']

training_args = TrainingArguments(
    per_device_eval_batch_size=100,
    output_dir="./results",
)

trainer = Trainer(
    model=model,
    args=training_args,
    compute_metrics=compute_metrics,
)

evaluation = trainer.evaluate(encoded_dataset)
print(evaluation)

print_weight_matrices(model.cpu(), visualization_mode='abs')