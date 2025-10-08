from torch.utils.data import Dataset
import torch
import json
import numpy as np
from sklearn.preprocessing import StandardScaler

class TimeSeriesDataset(Dataset):
    def __init__(self, json_paths, features=None, seq_len=128, transform=None, scale=None):
        super().__init__()
        self.seq_len = seq_len
        self.transform = transform
        self.scale = scale

        if isinstance(json_paths, str):
            json_paths = [json_paths]

        items = []
        for path in json_paths:
            with open(path, "r") as f:
                items.extend(json.load(f))
                
        if features is None:
            first_df = items[0]["dataframe"][0]
            exclude = []
            features = [k for k in first_df.keys() if k not in exclude]
        self.features = features

        self.proto_vocab = {"": 0, "None": 0, None: 0, "UDP": 1, "TCP": 2}

        valid = []
        for item in items:
            df = item["dataframe"]
            if len(df) == seq_len and all(feat in df[0] for feat in self.features):
                valid.append(item)
        print(f"Loaded {len(items)} → {len(valid)} valid samples")
        self.data = valid

        self.scalers = {}
        if self.scale == "zscore":
            for feat in self.features:
                if feat in ("DL_Protocol", "UL_Protocol"):
                    continue
                all_vals = []
                for item in self.data:
                    arr = [float(d[feat]) for d in item["dataframe"]]
                    all_vals.extend(arr)
                all_vals = np.array(all_vals, dtype=float).reshape(-1, 1)
                self.scalers[feat] = StandardScaler().fit(all_vals)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        df = item["dataframe"]

        series = []
        for d in df:
            row = []
            for feat in self.features:
                val = d[feat]
                if feat in ("DL_Protocol", "UL_Protocol"):
                    val = self.proto_vocab.get(str(val).strip() if val is not None else None, 0)
                else:
                    val = float(val)
                    if self.scale == "zscore" and feat in self.scalers:
                        val = self.scalers[feat].transform([[val]])[0, 0]
                row.append(val)
            series.append(row)

        # (L, C)
        x = torch.tensor(series, dtype=torch.float)

        if self.transform is not None:
            x = self.transform(x)

        anomaly_arr = np.array(item.get("anomaly_array", [0] * self.seq_len))
        label = 1 if anomaly_arr.sum() > 0 else 0

        return x, torch.tensor(label, dtype=torch.float)

    def get_features(self):
        return self.features


def get_transform(spec):
    if spec == "no":
        return None
    elif spec == "yes":
        return lambda x: x.log1p()
    else:
        raise ValueError(f"Unknown transform `{spec}` in config")
