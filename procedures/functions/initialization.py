import numpy as np
from transformers import (
    AutoModelForSequenceClassification,
)
from functions.pruning_methods import *

def load_model():
    model = AutoModelForSequenceClassification.from_pretrained("model")
    return model


def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    tp = np.sum(np.logical_and(preds, labels))
    tn = np.sum(np.logical_and(preds == 0, labels == 0))
    fp = np.sum(np.logical_and(preds, labels == 0))
    fn = np.sum(np.logical_and(preds == 0, labels))
    acc = np.sum(labels == preds) / len(labels)
    precision = 0 if tp + fp == 0 else tp / (tp + fp)
    recall = 0 if tp + fn == 0 else tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    mcc = (tp * tn - fp * fn) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "matthews": mcc,
    }