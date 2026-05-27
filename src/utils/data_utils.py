import numpy as np

protocol_map = {"TCP": 0, "UDP": 1, None: 2, "None": 2}


def anomaly_type_2_id(anomaly_type):
    mapping = {
        "Antenna Failure": 0,
        "Co-Channel Interference (Mild)": 1,
        "Co-Channel Interference (Severe)": 2,
        "Faulty RF Filters (Temporal)": 3,
        "Doppler Shift (Severe)": 4,
        "Faulty Handover Algorithm (Too Frequent)": 5,
        "Buffer Overflow (Gradual Buildup)": 6,
        "Resource Allocation Bugs": 7,
        "High Network Congestion (Gradual Buildup)": 8,
        "High Network Congestion (Sudden Spike)": 9,
    }
    return mapping[anomaly_type]


def make_sliding_windows(X_np, window_size=8):
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


def preprocess(data, type, window_size=8):
    if type == "anomaly detection":
        X_list = []
        for item in data:
            seq, row = [], []
            for key, values in item["KPIs"].items():
                if key in ["UL_Protocol", "DL_Protocol"]:
                    encoded_values = [protocol_map.get(v, 2) for v in values]
                    row.append(encoded_values)
                else:
                    row.append(values)
            seq = np.array(row).T  # (seq_len, n_channels)
            X_list.append(seq)

        y_list = [1 if item["anomalies"]["exists"] else 0 for item in data]

        X_np = np.array(X_list, dtype=np.float32)  # (n_samples, seq_len, n_channels)
        X_np = np.transpose(X_np, (0, 2, 1))  # (n_samples, n_channels, seq_len)
        y_np = np.array(y_list, dtype=np.int64)  # (n_samples,)
        return X_np, y_np

    elif type == "root-cause analysis":
        X_list, y_list = [], []
        for item in data:
            if (
                not item["anomalies"]["exists"]
                or item["anomalies"]["type"] == "Jamming"
            ):
                continue  # skip non-anomalous samples
            seq, row = [], []
            for key, values in item["KPIs"].items():
                if key in ["UL_Protocol", "DL_Protocol"]:
                    encoded_values = [protocol_map.get(v, 2) for v in values]
                    row.append(encoded_values)
                else:
                    row.append(values)
            seq = np.array(row).T  # (seq_len, n_channels)
            X_list.append(seq)
            y_list.append(anomaly_type_2_id(item["anomalies"]["type"]))

        X_np = np.array(X_list, dtype=np.float32)  # (n_samples, seq_len, n_channels)
        X_np = np.transpose(X_np, (0, 2, 1))  # (n_samples, n_channels, seq_len)
        y_np = np.array(y_list, dtype=np.int64)  # (n_samples,)

        # Z-score normalization per channel
        means = X_np.mean(axis=(0, 2), keepdims=True)  # (1, C, 1)
        stds = X_np.std(axis=(0, 2), keepdims=True)  # (1, C, 1)
        stds[stds == 0] = 1.0
        X_np = (X_np - means) / stds

        return X_np, y_np

    elif type == "anomaly duration":
        X_list, y_list = [], []
        for item in data:
            if (
                not item["anomalies"]["exists"]
                or item["anomalies"]["type"] == "Jamming"
            ):
                continue  # skip non-anomalous samples
            seq, row = [], []
            for key, values in item["KPIs"].items():
                if key in ["UL_Protocol", "DL_Protocol"]:
                    encoded_values = [protocol_map.get(v, 2) for v in values]
                    row.append(encoded_values)
                else:
                    row.append(values)
            seq = np.array(row).T  # (seq_len, n_channels)
            X_list.append(seq)

            anomaly_start_idx = item["anomalies"]["anomaly_duration"]["start"]
            anomaly_end_idx = item["anomalies"]["anomaly_duration"]["end"]
            anomaly_array = np.zeros(seq.shape[0], dtype=np.float32)
            anomaly_array[anomaly_start_idx : anomaly_end_idx + 1] = 1
            y_list.append(anomaly_array)

        X_np = np.array(X_list, dtype=np.float32)  # (n_samples, seq_len, n_channels)
        X_np = np.transpose(X_np, (0, 2, 1))  # (n_samples, n_channels, seq_len)
        y_np = np.array(y_list, dtype=np.float32)  # (n_samples, seq_len)
        return X_np, y_np

    elif type == "forecasting":
        X_list = []
        for item in data:
            if (
                not item["anomalies"]["exists"]
                or item["anomalies"]["type"] == "Jamming"
            ):
                continue  # skip non-anomalous samples
            seq, row = [], []
            for key, values in item["KPIs"].items():
                if key in ["UL_Protocol", "DL_Protocol"]:
                    encoded_values = [protocol_map.get(v, 2) for v in values]
                    row.append(encoded_values)
                else:
                    row.append(values)
            seq = np.array(row).T  # (seq_len, n_channels)
            X_list.append(seq)

        X_np = np.array(X_list, dtype=np.float32)  # (n_samples, seq_len, n_channels)
        X_np = np.transpose(X_np, (0, 2, 1))  # (n_samples, n_channels, seq_len)

        X_out, y_out = make_sliding_windows(X_np, window_size=window_size)

        # Z-score normalization per channel
        means = X_out.mean(axis=(0, 2), keepdims=True)  # (1, C, 1)
        stds = X_out.std(axis=(0, 2), keepdims=True)  # (1, C, 1)
        stds[stds == 0] = 1.0
        X_out = (X_out - means) / stds
        y_out = (y_out - means[:, :, 0]) / stds[:, :, 0]

        return X_out.astype(np.float32), y_out.astype(np.float32)

    else:
        raise ValueError(f"Unknown task type: {type}")
