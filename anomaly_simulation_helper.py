import pandas as pd
import glob
import os
import re
import json
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from datetime import datetime, timedelta

columns = [
    'RSRP', 'DL_BLER', 'DL_MCS', 'UL_BLER', 'UL_MCS', 'UL_NPRB', 'UL_SNR',
    'TX_Bytes', 'RX_Bytes', 'Estimated_UL_Buffer', 'PRBs_DL_Current', 'PRBs_UL_Current',
    'PRB_Utilization_DL', 'PRB_Utilization_UL', 'UL_Protocol', 'UL_NumberOfPackets',
    'DL_Protocol', 'DL_NumberOfPackets'
]

dtypes = {
    'RSRP': 'float64', 'DL_BLER': 'float64', 'DL_MCS': 'float64',
    'UL_BLER': 'float64', 'UL_MCS': 'float64', 'UL_NPRB': 'float64',
    'UL_SNR': 'float64', 'TX_Bytes': 'float64', 'RX_Bytes': 'float64',
    'Estimated_UL_Buffer': 'float64', 'PRBs_DL_Current': 'float64',
    'PRBs_UL_Current': 'float64', 'PRB_Utilization_DL': 'float64',
    'PRB_Utilization_UL': 'float64',
    'UL_Protocol': 'object', 'UL_NumberOfPackets': 'int64',
    'DL_Protocol': 'object', 'DL_NumberOfPackets': 'int64'
}

protocol_map = {
    "001": "None",
    "010": "TCP",
    "100": "UDP"
}

def validate_sequence(df):
    timestamps = df["timestamp"]
    timestamps = pd.to_datetime(timestamps, format='%Y-%m-%d %H:%M:%S.%f')
    diffs = timestamps.diff().dt.total_seconds()[1:]
    if (abs(diffs - 0.1) < 1e-6).all():
        return True
    else:
        print(f"Validation failed")
        return False

def format_sig_figs(df, copy = False):
    if copy:
        df = df.copy(deep=True)

    protocol_map = {
        "001": "None",
        "010": "TCP",
        "100": "UDP"
    }

    for col in df.columns:
        if col in ['RSRP', 'UL_NPRB', 'UL_SNR', 'PRBs_DL_Current', 'PRBs_UL_Current']:
            df[col] = df[col].round(0).astype(int)
        if col in ['DL_MCS', 'UL_MCS', 'PRB_Utilization_DL', 'PRB_Utilization_UL']:
            df[col] = df[col].round(1)
        if col == "DL_BLER" or col == "UL_BLER":
            df[col] = df[col].round(2)
        if col in ['TX_Bytes', 'RX_Bytes', 'Estimated_UL_Buffer', 'UL_NumberOfPackets', 'DL_NumberOfPackets']:
            df[col] = df[col].round(0).astype(int)
        
        if col in ['UL_Protocol', 'DL_Protocol']:
            df[col] = df[col].fillna("None").astype(str).replace(protocol_map)
        
        df[col] = df[col].astype(str)
    return df

def extract_sequences(df, anomaly_indicators = None, anomaly_identifiers = None, seq_len = 256, step = 32):
    df_sequences = []
    for i in range(0, len(df) - seq_len + 1, step):
        seq = df.iloc[i:i + seq_len]
        if validate_sequence(seq) and len(seq) == seq_len:
            first_timestamp = seq.iloc[0]["timestamp"]
            last_timestamp = seq.iloc[-1]["timestamp"]
            seq = seq.drop(columns=["timestamp"])
            if anomaly_indicators is None:
                df_sequences.append([seq, (first_timestamp, last_timestamp)])
            else:
                anomaly_indicators_seq = anomaly_indicators[i:i + seq_len]
                if len(anomaly_indicators_seq) == seq_len:
                    if anomaly_identifiers is None:
                        df_sequences.append([seq, (first_timestamp, last_timestamp), anomaly_indicators_seq])
                    else: 
                        unique_anomalies = set(anomaly_identifiers[i:i + seq_len])
                        unique_anomalies.discard(-1)
                        df_sequences.append([seq, (first_timestamp, last_timestamp), anomaly_indicators_seq, list(unique_anomalies)])
                else:
                    print(f"Invalid anomaly indicators length at index {i} for {df_name}")
        else:
            print(f"Invalid sequence at index {i} for {df_name}")
    return df_sequences

def extract_all_sequences(df_dfs, anomaly_indicators = None, anomaly_identifiers = None, seq_len = 256, step = 32):
    all_sequences = []
    for df_name, df in df_dfs.items():
        sequences = extract_sequences(df, anomaly_indicators, anomaly_identifiers, seq_len, step)
        all_sequences.extend(sequences)
    return all_sequences

def convert_intervals_to_binary(intervals, time_index):
    binary = np.zeros(len(time_index), dtype=int)
    for start, end in intervals:
        start = pd.to_datetime(start)
        end = pd.to_datetime(end)
        mask = (time_index >= start) & (time_index <= end)
        binary[mask] = 1
    return binary

def convert_intervals_to_anomalies(intervals, time_index):
    anomalies = np.full(len(time_index), -1, dtype=int)
    for anomaly, (start, end) in intervals:
        start = pd.to_datetime(start)
        end = pd.to_datetime(end)
        mask = (time_index >= start) & (time_index <= end)
        anomalies[mask] = anomaly_to_index[anomaly]
    return anomalies

import importlib
import anomaly_files.anomaly_timestamps_sim as sim
from anomaly_files.anomaly_timestamps_sim import anomaly_dict
importlib.reload(sim)

anomaly_to_index = {key: idx for idx, key in enumerate(anomaly_dict.keys())}

def process_anomalies(input_df, lambda_occurrence=60, lambda_resolution=150, test_anomaly=None, seq_len=128, step=32):
    if "date" not in input_df.columns:
        input_df = input_df.rename(columns={"timestamp": "date"})
    input_df['date'] = pd.to_datetime(input_df['date'], format='%Y-%m-%d %H:%M:%S.%f')
    start_time = pd.to_datetime(input_df['date'].iloc[0], format='%Y-%m-%d %H:%M:%S.%f')
    end_time = pd.to_datetime(input_df['date'].iloc[-1], format='%Y-%m-%d %H:%M:%S.%f')

    output_wireless_df, anomaly_occurrence_list, anomaly_list = sim.generate_anomalies(start_time, end_time, anomaly_dict, wireless_dataframe=input_df, lambda_occurrence=lambda_occurrence, lambda_resolution=lambda_resolution, test_anomaly=test_anomaly)
    output_wireless_df = output_wireless_df.rename(columns={"date": "timestamp"})
    time_range = pd.date_range(start=start_time, end=end_time, freq="100ms")
    anomaly_indicator = convert_intervals_to_binary(anomaly_occurrence_list, time_range)
    anomaly_identifiers = convert_intervals_to_anomalies(anomaly_list, time_range)
    return extract_sequences(output_wireless_df, anomaly_indicators=anomaly_indicator, anomaly_identifiers=anomaly_identifiers, seq_len=seq_len, step=step)


from anomaly_files.anomaly_timestamps_sim import anomaly_dict
import copy
anomaly_dict_copy = copy.deepcopy(anomaly_dict)
affected_kpi_dict = {}
for key, value in anomaly_dict_copy.items():
    value.get_instance()
    affected_kpi_dict[key] = list(value.metric_instance_dict.keys())

index_to_anomaly = {value: key for key, value in anomaly_to_index.items()}

def serialize_list(entry, single_anomaly=False):
    if single_anomaly:
        df, (start, end), arr, anomaly, affected_kpis = entry
        return {
            "dataframe": df.to_dict(orient="records"),
            "time_range": [str(start), str(end)],
            "anomaly_array": arr.tolist(),
            "anomaly": anomaly,
            "affected_kpis": affected_kpis
        }
    else: 
        df, (start, end), arr, anomaly = entry
        if anomaly == None:
            return {
                "dataframe": df.to_dict(orient="records"),
                "time_range": [str(start), str(end)],
                "anomaly_array": arr.tolist(),
                "anomaly": None
            }
        return {
            "dataframe": df.to_dict(orient="records"),
            "time_range": [str(start), str(end)],
            "anomaly_array": arr.tolist(),
            "anomaly": [int(x) for x in anomaly]
        }

def save_anomaly_data(data, name, folder="anomaly_data", single_anomaly=False):
    os.makedirs(folder, exist_ok=True)
    json_path = os.path.join(folder, f"{name}.json")

    serialized_data = [serialize_list(entry, single_anomaly) for entry in data]

    with open(json_path, "w") as f:
        json.dump(serialized_data, f, indent=2)

    print(f"Saved to {json_path}")

#apply this too all dataframes
def enforce_dtypes(df):
    dtype_map = {
        'RSRP': float,
        'DL_BLER': float,
        'DL_MCS': float,
        'UL_BLER': float,
        'UL_MCS': float,
        'UL_NPRB': float,
        'UL_SNR': float,
        'TX_Bytes': float,
        'RX_Bytes': float,
        'Estimated_UL_Buffer': float,
        'PRBs_DL_Current': float,
        'PRBs_UL_Current': float,
        'PRB_Utilization_DL': float,
        'PRB_Utilization_UL': float,
        'UL_Protocol': str,
        'UL_NumberOfPackets': int,
        'DL_Protocol': str,
        'DL_NumberOfPackets': int
    }

    for col, dtype in dtype_map.items():
        if dtype in [float, int]:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype(dtype)
        else:
            df[col] = df[col].astype(dtype)

    return df

def enforce_dtypes_across_sequences(sequences):
    for sequence in sequences:
        sequence[0] = enforce_dtypes(sequence[0])
    return sequences

def load_anomaly_data(name, folder="anomaly_data", anomaly = True):
    json_path = os.path.join(folder, f"{name}.json")

    with open(json_path, "r") as f:
        serialized_data = json.load(f)

    data = []
    for entry in serialized_data:
        df = pd.DataFrame(entry["dataframe"])
        df = enforce_dtypes(df)
        start, end = pd.to_datetime(entry["time_range"][0]), pd.to_datetime(entry["time_range"][1])
        arr = np.array(entry["anomaly_array"], dtype=float)
        anomaly = entry["anomaly"]
        if anomaly is not None:
            affected_kpis = entry.get("affected_kpis", []) 
            data.append((df, (start, end), arr, anomaly, affected_kpis)) #use tuple to fit my implemented functions, used list before to edit the anomaly entry
        else:
            data.append((df, (start, end), arr, anomaly))

    return data

def plot_six_columns_grid(dataframes, column_names, i, save_plot = False, seq_len=None, seed=42, anomaly_mask=None):
    # Load Arial directly from file
    font_path = "/usr/share/fonts/truetype/msttcorefonts/Arial.ttf"
    arial_prop = fm.FontProperties(fname=font_path)

    # Set default font globally (optional fallback)
    plt.rcParams['font.family'] = arial_prop.get_name()
    plt.rcParams['font.size'] = 9  
    plt.rcParams['axes.titlesize'] = 11 
    plt.rcParams['axes.labelsize'] = 9

    if i < 0 or i >= len(dataframes):
        raise IndexError("Index i is out of bounds for the dataframe list.")
    if len(column_names) != 6:
        raise ValueError("You must provide exactly 6 column names.")

    df = dataframes[i]
    mask = anomaly_mask[i]
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 3.8))
    axes = axes.flatten()  # make indexing easier

    protocol_order = {"none": 0, "UDP": 1, "TCP": 2}
    panel_labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']

    for idx, col in enumerate(column_names):
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame[{i}].")

        ax = axes[idx]
        if col in ["UL_Protocol", "DL_Protocol"]:
            numeric_values = df[col].map(protocol_order)
            ax.plot(numeric_values, linewidth=1)
            ax.set_yticks([0, 1, 2])
            ax.set_yticklabels(["none", "UDP", "TCP"], fontsize=7, fontproperties=arial_prop)
        else:
            ax.plot(df[col], linewidth=1)
            ax.tick_params(axis='y', labelsize=7)

        if mask is not None:
            is_anomaly = mask.values.astype(bool)
            spans = []
            in_span = False
            for idx2, val in enumerate(is_anomaly):
                if val and not in_span:
                    span_start = idx2 + df.index[0]
                    in_span = True
                elif not val and in_span:
                    ax.axvspan(span_start, idx2 + df.index[0], color='red', alpha=0.2)
                    in_span = False
            if in_span:
                ax.axvspan(span_start, len(is_anomaly) + df.index[0], color='red', alpha=0.2)

        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.8)
        title = f"{panel_labels[idx]} {col}"  
        ax.set_title(title, fontsize=10, pad=7, fontproperties=arial_prop)
        ax.set_facecolor('white')

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(0.8)
        ax.spines['bottom'].set_linewidth(0.8)
        ax.grid(False)
        ax.set_xticks([])

    plt.tight_layout(pad=0.8)
    plt.subplots_adjust(wspace=0.27, hspace=0.23)
    if save_plot:
        plt.savefig("six_panel_plot.png", dpi=300, bbox_inches='tight', transparent=True)
    plt.show()