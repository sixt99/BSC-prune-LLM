import torch.utils
from utils import *
from transformers import (
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    get_scheduler
)
from datasets import load_metric
import torch
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from torch.optim import AdamW
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

def prune_model_by_genes(model, genes, metadata):
    if not isinstance(genes, np.ndarray):
        genes = np.array(genes)
    gene_count = 0
    n_blocks_per_layer = []
    block_size = metadata['block_size']
    for layer_name, (grid_size_x, grid_size_y) in zip(metadata['pruned_layer_names'], metadata['grid_shapes']):
        idxs = np.where(genes[gene_count : gene_count + grid_size_x * grid_size_y] == 1)[0]
        n_blocks_per_layer.append(len(idxs))
        for idx in idxs:
            i = idx // grid_size_y
            j = idx % grid_size_y
            model.state_dict()[layer_name][block_size * i : block_size * (i + 1), block_size * j : block_size * (j + 1)].fill_(0)
        gene_count += grid_size_x * grid_size_y

    return n_blocks_per_layer, genes

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Tokenizer and model
model = AutoModelForSequenceClassification.from_pretrained("./trainings/before", num_labels=2)
model = model.to(device)
tokenizer = load_tokenizer()

# Obtain datasets
tokenized_datasets = load_tokenized_data(tokenizer)
tokenized_datasets = tokenized_datasets.remove_columns(["sentence", "idx"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")
train_data = tokenized_datasets["train"]#.shuffle(seed=42).select(range(100))
eval_data = tokenized_datasets["validation"]#.shuffle(seed=42).select(range(100))

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
    metrics[name] = load_metric(f'./metrics/{name}')

model.train()

# BIG TODO
#for param in model.bert.parameters():
#    param.requires_grad = False

for epoch in range(num_epochs):
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        # Set grads of blocks to zero
        for name, param in model.named_parameters():
            if len(param.shape) == 2:
                param.grad[param.cpu().detach().numpy()==0] = 0

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

output_dir = "./trainings/after"
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)