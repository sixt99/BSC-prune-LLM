from utils import *
from transformers import (
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    AutoTokenizer,
    DataCollatorWithPadding,
    get_scheduler
)
import torch
import evaluate
from tqdm.auto import tqdm
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.optim import AdamW

def tokenize_function(examples):
   return tokenizer(examples["sentence1"], examples["sentence2"], truncation=True)

device = torch.device("mps")

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=2)
model.to(device)

# Datasets
raw_datasets = load_dataset("glue", "mrpc")
tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.remove_columns(["sentence1", "sentence2", "idx"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")
train_data = tokenized_datasets["train"].shuffle(seed=42).select(range(10))
eval_data = tokenized_datasets["validation"].shuffle(seed=42).select(range(10))

# Data_collator
data_collator = DataCollatorWithPadding(tokenizer)

# Data_loaders
train_dataloader = DataLoader(train_data, shuffle=True, batch_size=8, collate_fn=data_collator)
eval_dataloader = DataLoader(eval_data, batch_size=8, collate_fn=data_collator)

# Training
num_epochs = 3
optimizer = AdamW(model.parameters(), lr=5e-5)
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
)
progress_bar = tqdm(range(num_training_steps))
model.train()
for epoch in range(num_epochs):
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)

metric = evaluate.load("accuracy")
model.eval()
for batch in eval_dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    metric.add_batch(predictions=predictions, references=batch["labels"])

print(metric.compute())