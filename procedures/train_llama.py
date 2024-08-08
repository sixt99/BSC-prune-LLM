from transformers import DataCollatorForSeq2Seq
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
import numpy as np
from datasets import load_from_disk
import evaluate
from accelerate import Accelerator
from tqdm.auto import tqdm
import torch
from torch.utils.data import DataLoader
from transformers import get_scheduler
import uuid
from peft import LoraConfig, TaskType, get_peft_model
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, set_seed
import copy
import torch.nn.functional as F
import json 
import os     

# Useful pages: 
# https://www.youtube.com/watch?time_continue=3938&v=Pb_RGAl75VE&embeds_referring_euri=https%3A%2F%2Fwww.google.com%2F&source_ve_path=Mjg2NjMsMjg2NjY&feature=emb_logo
# https://github.com/Maykeye/BTLM-peft-test-4bit/blob/main/test-btlm.ipynb
# https://github.com/tcapelle/llm_recipes/blob/main/nbs/Alpaca_finetunning_with_WandB.ipynb

show_examples = True
batch_size = 2
evaluate_after_each_epoch = True
seed_index = 0

accelerator = Accelerator()

@accelerator.on_main_process
def create_seeds(path, n_seeds = 10000):
    if os.path.exists(path):
        os.remove(path)
    with open(path, 'a') as file:
        for _ in range(n_seeds):
            file.write(str(uuid.uuid4().int % 2**32) + "\n")

def get_seed(path):
    global seed_index
    with open(path, "r") as file:
        seed_list = [line.strip() for line in file.readlines()]
    seed = int(seed_list[seed_index])
    seed_index += 1
    return seed

def prune_model_by_genes(model, genes, layer_names, grid_shapes, block_size = 64):
    if not isinstance(genes, np.ndarray):
        genes = np.array(genes)
    
    gene_count = 0
    n_blocks_per_layer = []
    # Iterate over the layers to be pruned
    for layer_name, (x, y) in zip(layer_names, grid_shapes):
        idxs = np.where(genes[gene_count : gene_count + x * y] == 1)[0]
        n_blocks_per_layer.append(len(idxs)) # Keep record of how many blocks there are in each layer
        for idx in idxs:
            # Block position within the grid
            i = idx // y
            j = idx % y

            # Coordinates of the block's vertices to prune
            a = block_size * i
            b = block_size * (i + 1)
            c = block_size * j
            d = block_size * (j + 1)

            # In some cases, 'module.' is appended at the beggining of the layer's name
            real_name = layer_name
            for _ in range(100):
                if real_name in model.state_dict().keys():
                    break
                else:
                    real_name = 'module.' + real_name

            # Replace layer portion by zeros
            model.state_dict()[real_name][a:b, c:d].fill_(0)
        gene_count += x * y

    return n_blocks_per_layer


def get_pruning_info(model, block_size = 256):
    layer_names = []
    for layer_name in model.state_dict().keys():
        dimensions = model.state_dict()[layer_name].shape
        if (
            len(dimensions) == 2 and
            dimensions[0] >= block_size and
            dimensions[1] >= block_size and
            (
                ".layer." in layer_name or
                ".layers." in layer_name or
                ".albert_layers." in layer_name or
                ".h." in layer_name
            )
        ):
            layer_names.append(layer_name)
    grid_shapes = []
    for layer_name in layer_names:
        shape = tuple(np.array(model.state_dict()[layer_name].shape) // block_size)
        grid_shapes.append(shape)
    blocks_per_layer = [i * j for i, j in grid_shapes]
    total_n_blocks = np.sum(blocks_per_layer).astype(int)
    np.random.seed(get_seed("./seeds.txt"))
    genes = np.random.binomial(1, 0.2, total_n_blocks)

    return layer_names, grid_shapes, blocks_per_layer, total_n_blocks, genes, block_size




def check_n_zeros(model, layer_names):
    zeros = 0
    elements = 0
    for layer_name in layer_names:
        matrix = model.state_dict()[layer_name].cpu().detach().numpy()
        zeros += np.sum(matrix == 0)
        elements += matrix.shape[0] * matrix.shape[1]
    return zeros / elements





# Evaluate how the model performs using metric
def evaluate_model(model, tokenizer, dataloader, metric, device):
    model.eval()
    for batch in dataloader:
        new = batch['labels'].clone()
        # Remove the tokens corresponding to the response
        # Place these tokens at the end of the row, with padding tokens on the left
        new = torch.where(batch['labels'] == -100, batch['input_ids'], 2)
        for i in range(len(new)):
            idxs = torch.where(new[i] != 2)[0]
            new[i][-len(idxs):] = new[i][idxs]
            new[i][:-len(idxs)] = 2
        min = torch.sum(new == 2, dim = -1).min()
        new = new[:,min:] # Trim as many '2' tokens as possible

        sentences = {}
        sentences['input_ids'] = new
        sentences['attention_mask'] = torch.ones(new.size()).int().to(device)
        sentences['attention_mask'][new == 2] = 0

        # Generate text
        with torch.no_grad():
            generated_tokens = accelerator.unwrap_model(model).generate(
                **sentences,
                max_new_tokens = 300,
                pad_token_id=tokenizer.pad_token_id
            )

        # Necessary to pad predictions and labels for being gathered
        generated_tokens = accelerator.pad_across_processes(
            generated_tokens, dim=1, pad_index=tokenizer.pad_token_id
        )
        labels = accelerator.pad_across_processes(batch['labels'], dim=1, pad_index=2)
        generated_tokens = accelerator.gather(generated_tokens)
        labels = accelerator.gather(labels)

        # Turn tokens into text and select the response
        decoded_sentences = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        decoded_sentences = [x.split('In this conversation,')[1].strip() for x in decoded_sentences]
        decoded_labels = torch.where(labels == -100, 2, labels)
        decoded_labels = tokenizer.batch_decode(decoded_labels, skip_special_tokens=True)

        if show_examples:
            accelerator.print('---------------------', flush = True)
            accelerator.print('Decoded sentences:')
            accelerator.print(json.dumps(decoded_sentences, indent = 4), flush = True)
            accelerator.print('Decoded labels:')
            accelerator.print(json.dumps(decoded_labels, indent = 4), flush = True)

        # Compute metric
        metric.add_batch(predictions=decoded_sentences, references=decoded_labels)

    return metric.compute()

def load_some_data(amount, tokenized_datasets, data_collator, dataset = "train"):
    if dataset == "train":
        train_dataloader = DataLoader(
            tokenized_datasets["train"].shuffle(seed = get_seed('./seeds.txt')).select(range(amount)),
            shuffle=True,
            batch_size=batch_size,
            collate_fn=data_collator
        )
        return train_dataloader

    elif dataset == "validation":
        eval_dataloader = DataLoader(
            tokenized_datasets["validation"].shuffle(seed = get_seed('./seeds.txt')).select(range(amount)),
            batch_size=20,
            collate_fn=data_collator
        )
        return eval_dataloader

def main():
    create_seeds('./seeds.txt', n_seeds = 10000)
    
    # Define special tokens to indicate instructions
    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

    def preprocess_function(examples):
        # Define prompts. Example:
        # [INST]<<SYS>>
        # Summarize the following conversation.
        # <</SYS>>
        # 
        # Hello! Alex here. Are you coming to the party?[/INST]
        # In this conversation, Alex asks if he's coming to the party.
        instructions = f"{B_INST}{B_SYS}Summarize the following conversation.{E_SYS}"
        prompts = [instructions + ex + E_INST + "\nIn this conversation, " for ex, sum in zip(examples['dialogue'], examples['summary'])]
        inputs = [prompt + sum + tokenizer.eos_token for prompt, sum in zip(prompts, examples['summary'])]
        targets = copy.deepcopy(inputs)
        model_inputs = tokenizer(inputs, text_target = targets, padding='max_length')

        # Set tokens to -100 if they are not generated by the model.
        # For example, consider the sentence "Hello, how are you? I am fine, thank you", encoded as:
        # [1, 15043, 29892, 920, 526, 366, 29973, 306, 626, 2691, 29892, 6452, 366, 2]
        # To only keep the tokens from "I am fine, thank you" set all previous tokens to -100:
        # [-100, -100, -100, -100, -100, -100, -100, 306, 626, 2691, 29892, 6452, 366, 2]

        # Tokens corresponding to "In this conversation,"
        split_tokens = torch.tensor([29914, 25580, 29962, 13, 797, 445, 14983, 29892])
        for i in range(len(model_inputs['labels'])):
            counter = 0
            for j, x in enumerate(model_inputs['labels'][i]):
                if counter == len(split_tokens):
                    break
                if x == split_tokens[counter]:
                    counter += 1
                else:
                    counter = 0
                model_inputs['labels'][i][j] = -100

        return model_inputs
    
    path = "/gpfs/projects/bsc03/bsc03268/tasks/meta-llama.Llama-2-7b-chat-hf-samsum"
    dataset = load_from_disk(path + "/dataset")
    tokenizer = AutoTokenizer.from_pretrained(path + "/tokenizer")
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=path + "/model")
    metric = evaluate.load('./metrics/rouge', experiment_id = get_seed('./seeds.txt'))
    device = accelerator.device
    model = AutoModelForCausalLM.from_pretrained(path + "/model").to(device)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    # Decoder-based models require "left" as padding-side
    # Otherwise, the model would generate text after a big amount of padding tokens
    # The model is not trained to manage this
    tokenizer.padding_side = "left"

    # Tokenize the whole dataset
    tokenized_datasets = dataset.map(
        preprocess_function,
        batched=True,
        batch_size=batch_size,
        remove_columns=dataset["train"].column_names,
    )
    tokenized_datasets.set_format("torch")

    # Create dataloaders we can iterate on
    train_dataloader = load_some_data(10, tokenized_datasets, data_collator, dataset = "train")
    eval_dataloader = load_some_data(10, tokenized_datasets, data_collator, dataset = "validation")

    ########## LORA ##########
    lora_config = LoraConfig(
        r=8,
        target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    ########## LORA ##########

    num_train_epochs = 50
    num_training_steps = num_train_epochs * len(train_dataloader)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    ########## PRUNING ##########
    layer_names, grid_shapes, blocks_per_layer, total_n_blocks, genes, block_size = get_pruning_info(model)
    before = check_n_zeros(model, layer_names)
    prune_model_by_genes(model, genes, layer_names, grid_shapes, block_size)
    after = check_n_zeros(model, layer_names)
    accelerator.print(f'Before: {before}\nAfter: {after}')
    ########## PRUNING ##########

    model, optimizer, lr_scheduler, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, lr_scheduler, train_dataloader, eval_dataloader
    )

    # Save loss and metric scores evolution
    losses = []
    train_metrics = []
    eval_metrics = []

    # Start training
    progress_bar = tqdm(range(len(train_dataloader) * num_train_epochs), disable=not accelerator.is_local_main_process)
    os.remove(f'cuda:{accelerator.process_index}.txt')
    for epoch in range(num_train_epochs):
        accelerator.print(f"\nEPOCH:{epoch}\n", flush=True)

        ########## EVALUATE ##########
        if evaluate_after_each_epoch:
            train_metric = evaluate_model(model, tokenizer, train_dataloader, metric, device)
            train_metrics.append(train_metric)
            eval_metric = evaluate_model(model, tokenizer, eval_dataloader, metric, device)
            eval_metrics.append(eval_metric)
        ########## EVALUATE ##########
        
        ########## TRAINING LOOP ##########
        model.train()
        # total_loss = 0
        for batch in train_dataloader:
            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss
            # total_loss += torch.tensor(loss.item())
            accelerator.backward(loss)

            # Set gradients of blocks to zero
            # for _, param in model.named_parameters():
            #     if len(param.shape) == 2 and param.requires_grad:  # Check if the parameter is a 2D tensor and requires gradients
            #         param.grad[param.cpu().detach().numpy() == 0] = 0

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)

        # train_loss = total_loss / len(train_dataloader)
        # print(train_loss)
        # train_loss = accelerator.gather(tensor=train_loss)
        ########## TRAINING LOOP ##########
 
        with open(f"cuda:{accelerator.process_index}.txt", "a") as file:
            file.write(f'{loss}, ')

        # accelerator.print(f'Train evaluation: {train_metric}', flush=True)
        # accelerator.print(f'Eval evaluation: {eval_metric}', flush=True)
        losses.append(loss.item())

    accelerator.print(json.dumps(losses))
    if evaluate_after_each_epoch:
        accelerator.print('TRAIN METRICS')
        accelerator.print(json.dumps(train_metrics))
        accelerator.print('EVAL METRICS')
        accelerator.print(json.dumps(eval_metrics))

if __name__ == "__main__":
    main()