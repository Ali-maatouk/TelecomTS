import torch
import importlib
import numpy as np
import torch.nn as nn
import torch.optim as optim
from types import SimpleNamespace
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, mean_absolute_error
)


def prepare(config, X_train, y_train) -> tuple:
    """
    Prepare the model, head, dataset, dataloader, optimizer, and loss function.
    """
    encoder_type = config["encoder_type"]
    type = config["task_type"]

    criterion = (
        nn.CrossEntropyLoss()
        if type not in ["anomaly duration", "forecasting"]
        else nn.MSELoss()
    )

    encoder_type = config["encoder_type"]
    module = importlib.import_module(f"encoders.{encoder_type}")
    EncoderClass = getattr(module, "Model")

    model = EncoderClass(SimpleNamespace(**config[f"{encoder_type}_model"])).float()
    d_model = config[f"{encoder_type}_model"]["d_model"]

    if type == "anomaly detection":
        head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 2),
        )
    elif type == "root-cause analysis":
        head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(0.2),
            nn.Linear(d_model, 11),
        )
    elif type == "anomaly duration":
        head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, config[f"{encoder_type}_model"]["seq_len"]),
            nn.Sigmoid(),
        )
    elif type == "forecasting":
        head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, config[f"{encoder_type}_model"]["enc_in"]),
        )

    train_dataset = torch.utils.data.TensorDataset(
        torch.tensor(X_train), torch.tensor(y_train)
    )

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=config["train"]["batch_size"],
        shuffle=True,
        drop_last=False,
    )

    optimizer = optim.Adam(
        list(model.parameters()) + list(head.parameters()),
        lr=config["train"]["optim"]["lr"],
        weight_decay=config["train"]["optim"]["weight_decay"],
        betas=config["train"]["optim"]["betas"],
    )
    return model, head, train_dataset, train_dataloader, optimizer, criterion


def _prevalence_weights(y_true, p_target=0.05, pos_label=1, eps=1e-6) -> np.ndarray:
    """
    Reweight samples so eval reflects a target prevalence (e.g., 5% anomalies)
    """
    p_curr = float((y_true == pos_label).mean())
    p_curr = min(max(p_curr, eps), 1 - eps)
    w_pos = p_target / p_curr
    w_neg = (1 - p_target) / (1 - p_curr)
    return np.where(y_true == pos_label, w_pos, w_neg)


def evaluate(model, head, dataset, task: str) -> dict:
    """
    task ∈ {'anomaly detection', 'root-cause analysis', 'anomaly duration', 'forecasting'}
    """
    model.eval(); head.eval()
    X, y = dataset.tensors

    with torch.no_grad():
        Xs = X.permute(0, 2, 1).float()
        out = model(Xs)
        logits = head(out)

    metrics = {}

    if task == 'forecasting':
        y_true = y.cpu().numpy()
        y_pred = logits.cpu().numpy()
        mae = mean_absolute_error(y_true, y_pred)
        rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
        metrics.update({'mae': mae, 'rmse': rmse})
        return metrics

    if task == 'anomaly duration':
        y_pred = np.round(logits).cpu().numpy().astype(np.int64)
        y_true = y.cpu().numpy().astype(np.int64)

        y_true_flat = y_true.reshape(-1)
        y_pred_flat = y_pred.reshape(-1)

        acc = accuracy_score(y_true_flat, y_pred_flat)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_true_flat, y_pred_flat, average='binary', pos_label=1, zero_division=0
        )
        metrics.update({'accuracy': acc, 'precision_anom': prec, 'recall_anom': rec, 'f1_anom': f1})
        return metrics

    # Classification (binary or multi-class)
    y_true = y.cpu().numpy().astype(np.int64)
    y_logits = logits.cpu().numpy()
    y_pred = y_logits.argmax(axis=1)

    # --- plain (unweighted) summary ---
    acc = accuracy_score(y_true, y_pred)
    cm  = confusion_matrix(y_true, y_pred, labels=np.unique(y_true))
    prec_m, rec_m, f1_m, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )

    metrics.update({
        'accuracy': acc,
        'macro_precision': prec_m,
        'macro_recall': rec_m,
        'macro_f1': f1_m,
        'confusion_matrix': cm,
    })

    # --- prevalence-adjusted (only for anomaly detection) ---
    if task == 'anomaly detection':
        sw = _prevalence_weights(y_true, p_target=0.05, pos_label=1)
        prec_adj, rec_adj, f1_adj, _ = precision_recall_fscore_support(
            y_true, y_pred, average='binary', pos_label=1, sample_weight=sw, zero_division=0
        )
        cm_adj = confusion_matrix(y_true, y_pred, labels=[0, 1], sample_weight=sw)

        metrics.update({
            'anomaly_precision_prevalence_adjusted': prec_adj,
            'anomaly_recall_prevalence_adjusted':    rec_adj,
            'anomaly_f1_prevalence_adjusted':        f1_adj,
            'confusion_matrix_prevalence_adjusted':  cm_adj,
        })

    return metrics
