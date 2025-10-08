import numpy as np

protocol_map = {"TCP": 0, "UDP": 1, "None": 2}


def _anomaly_type_2_id(anomaly_type: str) -> int:
    """
    Map anomaly type strings to integer IDs.
    """
    mapping = {
        "Jamming": 0,
        "Antenna Failure": 1,
        "Co-Channel Interference (Mild)": 2,
        "Co-Channel Interference (Severe)": 3,
        "Faulty RF Filters (Temporal)": 4,
        "Doppler Shift (Severe)": 5,
        "Faulty Handover Algorithm (Too Frequent)": 6,
        "Buffer Overflow (Gradual Buildup)": 7,
        "Resource Allocation Bugs": 8,
        "High Network Congestion (Gradual Buildup)": 9,
        "High Network Congestion (Sudden Spike)": 10,
    }
    return mapping[anomaly_type]


def _make_sliding_windows(X_np: np.ndarray, window_size: int = 8) -> (np.ndarray, np.ndarray):
    """
    X_np: (N, C, T)
    Returns:
        X_out: (batch, window_size, C)
        y_out: (batch, C)
    """
    N, C, T = X_np.shape
    assert T > window_size, "Sequence too short for given window_size"

    X_list, y_list = [], []
    for n in range(N):
        for t in range(T - window_size):
            X_list.append(X_np[n, :, t : t + window_size].T)  # (window_size, C)
            y_list.append(X_np[n, :, t + window_size])  # (C,)

    X_out = np.stack(X_list)  # (batch, window_size, C)
    X_out = X_out.transpose(0, 2, 1)  # (batch, C, window_size)
    y_out = np.stack(y_list)  # (batch, C)

    return X_out.astype(np.float32), y_out.astype(np.float32)


def _balance_data(X: np.ndarray, y: np.ndarray) -> (tuple):
    """
    Downsample to balance classes in y
    """
    # Find the indices of each class
    class_indices = {}
    for idx, label in enumerate(y):
        if label not in class_indices:
            class_indices[label] = []
        class_indices[label].append(idx)

    # Find the minimum class count
    min_count = min(len(indices) for indices in class_indices.values())

    # Reduce each class to the minimum count
    balanced_indices = []
    for indices in class_indices.values():
        balanced_indices.extend(indices[:min_count])

    # Select only the balanced indices
    X_balanced = X[balanced_indices]
    y_balanced = y[balanced_indices]

    return X_balanced, y_balanced


def _kpis_to_seq(item, protocol_map):
    """KPIs dict -> (T, C) array with protocols encoded."""
    rows = []
    # preserve insertion order of keys (JSON -> dict keeps it)
    for key in list(item["KPIs"].keys()):
        vals = item["KPIs"][key]
        if key in ["UL_Protocol", "DL_Protocol"]:
            vals = [protocol_map.get(v, 2) for v in vals]
        rows.append(vals)
    return np.asarray(rows, dtype=np.float32).T  # (T, C)


def preprocess(data, task, window_size=8):
    """
    task ∈ {'anomaly detection', 'root-cause analysis', 'anomaly duration', 'forecasting'}
    Returns:
      - detection / root-cause: (X: (N,C,T), y: (N,))
      - duration:               (X: (N,C,T), y: (N,T))
      - forecasting:            (X: (B,C,window), y: (B,C)) after sliding windows
    """
    if task == "anomaly detection":
        X_list = [_kpis_to_seq(item, protocol_map) for item in data]
        y_list = [1 if item["anomalies"]["exists"] else 0 for item in data]

        X = np.asarray(X_list, dtype=np.float32).transpose(0, 2, 1)  # (N,C,T)
        y = np.asarray(y_list, dtype=np.int64)

        # Balance the dataset (uses your existing helper)
        X, y = _balance_data(X, y)
        return X, y

    elif task == "root-cause analysis":
        X_list, y_list = [], []
        for item in data:
            if not item["anomalies"]["exists"]:
                continue
            X_list.append(_kpis_to_seq(item, protocol_map))
            y_list.append(_anomaly_type_2_id(item["anomalies"]["type"]))

        X = np.asarray(X_list, dtype=np.float32).transpose(0, 2, 1)  # (N,C,T)
        y = np.asarray(y_list, dtype=np.int64)

        # Per-channel z-score normalization
        X = (X - X.mean(axis=(0, 2), keepdims=True)) / (X.std(axis=(0, 2), keepdims=True) + 1e-6)

        return X, y

    elif task == "anomaly duration":
        X_list, y_list = [], []
        for item in data:
            if not item["anomalies"]["exists"]:
                continue
            seq = _kpis_to_seq(item, protocol_map)  # (T,C)
            X_list.append(seq)

            T = seq.shape[0]
            s = item["anomalies"]["anomaly_duration"]["start"]
            e = item["anomalies"]["anomaly_duration"]["end"]
            arr = np.zeros(T, dtype=np.float32)
            arr[s:e+1] = 1.0
            y_list.append(arr)

        X = np.asarray(X_list, dtype=np.float32).transpose(0, 2, 1)    # (N,C,T)
        y = np.asarray(y_list, dtype=np.float32)                       # (N,T)
        return X, y

    elif task == "forecasting":
        # only consider anomalous sequences
        X_list = [
            _kpis_to_seq(item, protocol_map)
            for item in data
            if item["anomalies"]["exists"]
        ]
        X = np.asarray(X_list, dtype=np.float32).transpose(0, 2, 1)    # (N,C,T)

        Xw, yw = _make_sliding_windows(X, window_size=window_size)     # (B,C,w), (B,C)

        # Per-channel z-score (match your original)
        mu = Xw.mean(axis=(0, 2), keepdims=True)
        sd = Xw.std(axis=(0, 2), keepdims=True)
        sd[sd == 0] = 1.0
        Xw = (Xw - mu) / sd
        yw = (yw - mu[:, :, 0]) / sd[:, :, 0]

        return Xw.astype(np.float32), yw.astype(np.float32)

    else:
        raise ValueError(f"Unknown task: {task}")
