import torch.utils
from functions.make_plots import *
from functions.pruning_methods import *
from functions.initialization import *
from transformers import (
    AutoModelForSequenceClassification,
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
import os
import pandas as pd
def tokenize_function(examples):
   return tokenizer(examples["sentence"], truncation=True)

# Define the device
device = torch.device("mps")

# Tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
#model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
model = AutoModelForSequenceClassification.from_pretrained("model", num_labels=2)
model.to(device)

# Prune the model
area_percentage = 0.3
block_size = 128
sort_by = "eval_matthews"
masks = {}
# Randomly prune and save the masks
# for x in model.state_dict().keys():
#     tensor = model.state_dict()[x]
#     if ".layer." in x and len(tensor.size()) == 2:
#         output = randomly_prune_blocks_by_area(tensor, area_percentage = area_percentage, block_size = block_size, verbose = True)
#         masks[x] = create_mask(output['pairs'], output['original_size'], block_size = block_size)

# Prune according to the best configurations
# Iterate over the layers
for layer in model.state_dict():
    matrix = model.state_dict()[layer]
    if len(matrix.shape) == 2: # If the weights are matrices
        file_name = f"outputs/{layer}/output_a{area_percentage}_bs{block_size}.csv"
        if not os.path.exists(file_name): # If we don't have information about this layer, don't prune it
            continue

        if "ffn.lin1" in layer: # We do not modify these matrices
            continue
        
        # Get the best distribution of blocks according to sort_by metric
        df = pd.read_csv(file_name)
        best_idx = np.argmax(df[sort_by])
        string = df.loc[best_idx]['pairs'][2:-2].split('),(')        
        pairs = [list(map(int, x.split(','))) for x in string]

        # Prune the matrix with the given blocks
        output = prune_by_pairs(matrix, pairs, block_size, verbose=True)

#print_weight_matrices(model.cpu(), visualization_mode='abs')

# Obtain datasets
raw_datasets = load_dataset("glue", "cola")
tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.remove_columns(["sentence", "idx"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")
train_data = tokenized_datasets["train"]#.shuffle(seed=42).select(range(10))
eval_data = tokenized_datasets["validation"]#.shuffle(seed=42).select(range(10))

# Data_collator and data_loaders
data_collator = DataCollatorWithPadding(tokenizer)
train_dataloader = DataLoader(train_data, shuffle=True, batch_size=8, collate_fn=data_collator)
eval_dataloader = DataLoader(eval_data, batch_size=8, collate_fn=data_collator)

# Training
num_epochs = 5
optimizer = AdamW(model.parameters(), lr=5e-5)
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
)
progress_bar = tqdm(range(num_training_steps))
metric_names = ['accuracy', 'precision', 'recall', 'f1', 'matthews_correlation']
metrics = {}
results = {}
for name in metric_names:
    metrics[name] = evaluate.load(name)
model.train()
for epoch in range(num_epochs):
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        for name, param in model.named_parameters():
            if len(param.shape) == 2:
                param.grad[param.cpu().detach().numpy()==0] = 0
                #plot_matrix_analysis(param.grad.cpu().detach().numpy(), visualization_mode='std')

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)

    # Evaluate after every epoch
    model.eval()
    for batch in eval_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        for name in metric_names:
            metrics[name].add_batch(predictions=predictions, references=batch["labels"])

    for name in metric_names:
        results.update(metrics[name].compute())

    print(results)
    model.train()

output_dir = "/Users/sixteoriolllenassegura/prune_llm/trainings/take_best_matrices_a0.3_bz128"
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)