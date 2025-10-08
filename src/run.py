import yaml
import torch
import random
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from utils.data_utils import preprocess
from utils.train_utils import evaluate, prepare

with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)

task_type = config["task_type"]

seed = config["seed"]
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

data = load_dataset(
    "AliMaatouk/TelecomTS",
    data_files={"full": "**/chunked.jsonl"}
)

splits = data["full"].train_test_split(test_size=config["split_ratio"], shuffle=True)
train_data = splits["train"]
test_data = splits["test"] 

# ======== Train ========

seq_len = config[f"{config['encoder_type']}_model"]["seq_len"]
X_train, y_train = preprocess(train_data, task_type, seq_len)

model, head, train_dataset, train_dataloader, optimizer, criterion = prepare(
    config, X_train, y_train
)

tqdm.write("Training...")

model.train()
head.train()
for epoch in range(config["train"]["epochs"]):
    train_losses = []
    for batch in tqdm(
        train_dataloader, desc=f"Epoch {epoch+1}/{config['train']['epochs']}"
    ):
        optimizer.zero_grad()
        outputs = model(batch[0].permute(0, 2, 1))
        logits = head(outputs)
        loss = criterion(logits, batch[1])
        train_losses.append(loss.item())
        loss.backward()
        optimizer.step()

    tqdm.write(
        f"Epoch {epoch+1}/{config['train']['epochs']}, Loss: {np.mean(train_losses):.4f}"
    )

train_metrics = evaluate(model, head, train_dataset, task_type)
for k, v in train_metrics.items():
    if isinstance(v, (int, float)):
        tqdm.write(f"Train {k}: {v:.4f}")
    else:
        v = [np.round(x, 4).tolist() for x in v.tolist()]
        tqdm.write(f"Train {k}:\n{v}")

# ======== Evaluate ========

tqdm.write("Evaluating...")

X_test, y_test = preprocess(test_data, task_type, seq_len)

test_dataset = torch.utils.data.TensorDataset(
    torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.int64)
)
test_metrics = evaluate(model, head, test_dataset, task_type)
for k, v in test_metrics.items():
    if isinstance(v, (int, float)):
        tqdm.write(f"Test {k}: {v:.4f}")
    else:
        v = [np.round(x, 4).tolist() for x in v.tolist()]
        tqdm.write(f"Test {k}:\n{v}")
