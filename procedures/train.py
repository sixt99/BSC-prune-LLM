from utils import *
from transformers import (
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    AutoTokenizer
)
from datasets import load_dataset

def preprocess_function(examples):
    return tokenizer(examples["sentence"], truncation=True)

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

# Prune the model
# for x in model.state_dict().keys():
#   tensor = model.state_dict()[x]
#   if ".layer." in x and len(tensor.size()) == 2:
#     output = randomly_prune_blocks_by_area(tensor, area_percentage = 0.3, block_size = 128, verbose = True)
# pairs = output['pairs']

dataset = load_dataset("glue", "cola")
encoded_dataset = dataset.map(preprocess_function, batched=True)

args = TrainingArguments(
    evaluation_strategy = "epoch",
    save_strategy = "epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=15,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="matthews",
    #push_to_hub=True,
    output_dir="./trainings/prune_and_train"
)

trainer = Trainer(
    model,
    args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["validation"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()