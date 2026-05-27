import torch
import importlib
import numpy as np
import torch.nn as nn
import torch.optim as optim
from types import SimpleNamespace
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix


def prepare(config, X_train, y_train):
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
            nn.Linear(d_model, 10),
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


def evaluate(model, head, dataset, gt, type, batch_size=64):
    """Run inference in fixed-size batches and aggregate predictions.

    Batching is important for foundation-style encoders (Chronos / Toto / Mantis)
    that allocate large per-channel intermediate tensors; running the whole
    dataset in a single forward pass would OOM on CPU.
    """
    model.eval()
    head.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    logits_chunks = []
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.permute(0, 2, 1)
            logits_chunks.append(head(model(xb)).cpu().numpy())
    logits = np.concatenate(logits_chunks, axis=0)

    if type == "anomaly duration":
        train_accuracy = (np.round(logits) == gt).mean()
        print(f"Accuracy: {train_accuracy * 100:.2f}%")
    elif type == "forecasting":
        train_mae = np.mean(np.abs(logits - gt))
        train_RMSE = np.sqrt(((logits - gt) ** 2).mean())  # RMSE
        print(f"MAE: {train_mae:.4f}")
        print(f"RMSE: {train_RMSE:.4f}")
    else:
        y_pred = np.argmax(logits, axis=1)
        train_accuracy = (y_pred == gt).mean()
        print(f"Accuracy: {train_accuracy * 100:.2f}%")

        cm = confusion_matrix(gt, y_pred, labels=np.unique(gt))
        print("Confusion matrix (rows=true, cols=pred):\n", cm)
