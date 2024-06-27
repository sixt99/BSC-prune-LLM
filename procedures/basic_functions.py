import numpy as np
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    LlamaForCausalLM,
    DataCollatorWithPadding
)
from datasets import load_from_disk, load_metric
import uuid
import evaluate
from tqdm.auto import tqdm
from accelerate import Accelerator
import torch

# ------ TEXT CLASSIFICATION ------
def get_columns_classification():
    return [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "matthews",
        "gm"
    ]

# def compute_metrics_classification(pred):
#     labels = pred.label_ids
#     preds = np.argmax(pred.predictions, axis=1)
#     tp = np.sum(np.logical_and(preds, labels))
#     tn = np.sum(np.logical_and(preds == 0, labels == 0))
#     fp = np.sum(np.logical_and(preds, labels == 0))
#     fn = np.sum(np.logical_and(preds == 0, labels))
#     acc = np.sum(labels == preds) / len(labels)
#     precision = 0 if tp + fp == 0 else tp / (tp + fp)
#     recall = 0 if tp + fn == 0 else tp / (tp + fn)
#     f1 = 0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
#     mcc = 0 if (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) == 0 else (tp * tn - fp * fn) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
#     return {
#         "accuracy": acc,
#         "precision": precision,
#         "recall": recall,
#         "f1": f1,
#         "matthews": mcc,
#     }

def load_model_classification(model_path):
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    return model

def load_tokenizer_classification(tokenizer_path):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    return tokenizer

def load_tokenized_data_classification(tokenizer, dataset_path, task_name):
    dataset = load_from_disk(dataset_path)

    def preprocess_function(examples):
        if 'cola' in task_name.lower():
            return tokenizer(examples["sentence"], truncation=True, padding=True)
        elif 'mrpc' in task_name.lower():
            return tokenizer(examples["sentence1"], examples["sentence2"], truncation=True, padding=True)
        
    tokenized_dataset = dataset.map(preprocess_function, batched=True)

    # Specific preprocessing for different tasks
    if 'cola' in task_name.lower():
        tokenized_dataset = tokenized_dataset.remove_columns(["sentence", "idx"])
        tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
        tokenized_dataset.set_format("torch")

    elif 'mrpc' in  task_name.lower():
        tokenized_dataset = tokenized_dataset.remove_columns(["sentence1", "sentence2", "idx"])
        tokenized_dataset = tokenized_dataset.rename_column("label", "labels")    
        tokenized_dataset.set_format("torch")

    return tokenized_dataset

def load_data_collator_classification(tokenizer):
    data_collator = DataCollatorWithPadding(tokenizer)
    return data_collator

def load_metric_names_classification():
    return ['accuracy', 'precision', 'f1', 'matthews_correlation']

def evaluate_model_classification(
    model,
    dataloader,
    tokenizer,
    device,
    metric_names,
    metric_instances,
    accelerator
):
    
    model.eval() # Set the model to evaluation mode
    for batch in tqdm(dataloader):
        # Move the batch data to the specified device (CPU, MPS or GPU)
        batch = {k: v.to(device) for k, v in batch.items()}
        
        # Disable gradient calculation for evaluation
        with torch.no_grad():
            outputs = model(**batch) # Forward pass: compute model outputs
        
        # Compute predicted labels by taking the argmax over the logits
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        
        # Update each metric with the current batch of predictions and references (true labels)
        for name in metric_names:
            metric_instances[name].add_batch(predictions=predictions, references=batch["labels"])

    evaluation = {}
    for name in metric_names:
        evaluation.update(metric_instances[name].compute())

    return evaluation

def train_step_classification(
    model,
    train_dataloader,
    device,
    optimizer,
    lr_scheduler,
    progress_bar
):
    model.train() # Set the model to training mode
    for batch in train_dataloader:
        # Move the batch data to the specified device (CPU, MPS, or GPU)
        batch = {k: v.to(device) for k, v in batch.items()}
        
        outputs = model(**batch) # Forward pass: compute model outputs
        loss = outputs.loss # Extract the loss from the model outputs
        loss.backward() # Backward pass: compute gradients
        
        # Set gradients of blocks to zero
        for name, param in model.named_parameters():
            if len(param.shape) == 2 and param.requires_grad:  # Check if the parameter is a 2D tensor and requires gradients
                param.grad[param.cpu().detach().numpy() == 0] = 0

        optimizer.step() # Update model parameters based on gradients
        lr_scheduler.step() # Update learning rate based on the scheduler
        optimizer.zero_grad() # Reset gradients for the next iteration
        progress_bar.update(1) # Update the progress bar


# ------ MACHINE TRANSLATION ------
def get_columns_translation():
    return [
        "bleu",
    ]

# def compute_metrics_translation(eval_preds):
#     # TODO is there a better way to do this?
#     tokenizer = load_tokenizer_translation("/gpfs/projects/bsc03/bsc03268/tasks/google-t5.t5-small-opus/tokenizer")
#     metric = evaluate.load(f'./metrics/bleu', experiment_id = uuid.uuid4().int % 2**32)
#     preds, labels = eval_preds
#     preds = np.where(preds != -100, preds, tokenizer.pad_token_id)

#     if isinstance(preds, tuple):
#         preds = preds[0]
#     decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

#     labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
#     decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

#     def postprocess_text(preds, labels):
#         preds = [pred.strip() for pred in preds]
#         labels = [[label.strip()] for label in labels]
#         return preds, labels

#     decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

#     result = metric.compute(predictions=decoded_preds, references=decoded_labels)
#     result = {"bleu": result["bleu"]}

#     prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
#     result["gen_len"] = np.mean(prediction_lens)
#     result = {k: round(v, 4) for k, v in result.items()}
#     return result

def load_model_translation(model_path):
    if "llama" in model_path.lower():
        model = LlamaForCausalLM.from_pretrained(model_path)
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    return model

def load_tokenizer_translation(tokenizer_path):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    return tokenizer

def load_tokenized_data_translation(tokenizer, dataset_path, task_name = None):
    dataset = load_from_disk(dataset_path)

    max_length = 128
    def preprocess_function(examples):
        inputs = ["translate English to French: " + ex["en"] for ex in examples["translation"]]
        targets = [ex["fr"] for ex in examples["translation"]]
        model_inputs = tokenizer(
            inputs, text_target=targets, max_length=max_length, truncation=True
        )
        return model_inputs

    tokenized_datasets = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset["train"].column_names,
    )
    tokenized_datasets.set_format("torch")

    return tokenized_datasets

# TODO ATENCIÓ, això ho hem de cridar cada cop que fem pruning del model perquè depèn del model oi? :(
def load_data_collator_translation(tokenizer):
    data_collator = DataCollatorForSeq2Seq(tokenizer)
    return data_collator

def load_metric_names_translation():
    return ['bleu']

def evaluate_model_translation(
    model,
    dataloader,
    tokenizer,
    device,
    metric_names,
    metric_instances,
    accelerator
):
    def postprocess(predictions, labels):
        predictions = predictions.cpu().numpy()
        labels = labels.cpu().numpy()

        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Some simple post-processing
        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [[label.strip()] for label in decoded_labels]
        return decoded_preds, decoded_labels

    model.eval()
    for batch in tqdm(dataloader, disable=not accelerator.is_local_main_process):
        with torch.no_grad():
            generated_tokens = accelerator.unwrap_model(model).generate(
                batch["input_ids"],
                attention_mask=batch["attention_mask"],
                max_length=128,
            )
        labels = batch["labels"]

        # Necessary to pad predictions and labels for being gathered
        generated_tokens = accelerator.pad_across_processes(
            generated_tokens, dim=1, pad_index=tokenizer.pad_token_id
        )
        labels = accelerator.pad_across_processes(labels, dim=1, pad_index=-100)

        predictions_gathered = accelerator.gather(generated_tokens)
        labels_gathered = accelerator.gather(labels)

        decoded_preds, decoded_labels = postprocess(predictions_gathered, labels_gathered)
        for name in metric_names:
            metric_instances[name].add_batch(predictions=decoded_preds, references=decoded_labels)

    evaluation = {}
    for name in metric_names:
        evaluation.update(metric_instances[name].compute())

    return evaluation

# TODO adapt this to translation
def train_step_translation(
    model,
    train_dataloader,
    device,
    optimizer,
    lr_scheduler,
    progress_bar
):
    model.train() # Set the model to training mode
    for batch in train_dataloader:
        outputs = model(**batch) # Forward pass: compute model outputs
        loss = outputs.loss # Extract the loss from the model outputs
        accelerator.backward(loss) # Backward pass: compute gradients
        
        # Set gradients of blocks to zero
        for name, param in model.named_parameters():
            if len(param.shape) == 2 and param.requires_grad:  # Check if the parameter is a 2D tensor and requires gradients
                param.grad[param.cpu().detach().numpy() == 0] = 0

        optimizer.step() # Update model parameters based on gradients
        lr_scheduler.step() # Update learning rate based on the scheduler
        optimizer.zero_grad() # Reset gradients for the next iteration
        progress_bar.update(1) # Update the progress bar