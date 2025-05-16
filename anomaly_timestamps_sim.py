import numpy as np
import pandas as pd
import random
from datetime import datetime, timedelta
import openai
import os 
from dotenv import load_dotenv

from anomaly_files.anomalies import Anomaly, Metric
import matplotlib.pyplot as plt

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = openai.Client(api_key=api_key)

columns = [
    "date", "UL_NumberOfPackets", "DL_NumberOfPackets", "UL_BLER", "DL_BLER", "RSRP", "UL_SNR",
    "Estimated_UL_Buffer", "TX_Bytes", "RX_Bytes", "PRB_Utilization_DL", "PRB_Utilization_UL",
    "PRBs_DL_Current", "PRBs_UL_Current", "DL_MCS", "UL_MCS", "UL_NPRB"
]

def generate_anomalies(start_time, end_time, anomaly_dict, wireless_dataframe = None, lambda_occurrence=40, lambda_resolution=60, test_anomaly=None):
    """
    Simulate anomalies occurring in a wireless network within a given time range.
    
    Parameters:
    start_time (datetime): The start of the simulation period.
    end_time (datetime): The end of the simulation period.
    lambda_occurrence (float): Rate parameter for time between anomaly resolution and next anomaly detected (per hour).
    lambda_resolution (float): Rate parameter for maintenance time (per hour).
    """
    anomaly_types = list(anomaly_dict.keys())
    current_time = start_time
    old_df = wireless_dataframe.copy(deep=True)
    anomaly_occurrence_list = []
    anomaly_list = []
    while current_time < end_time:
        # Time until the next anomaly from resolution (exponential inter-arrival time)
        time_to_next_anomaly = np.random.exponential(scale=1/lambda_occurrence)
        anomaly_time = current_time + timedelta(hours=time_to_next_anomaly)
        
        if anomaly_time >= end_time:
            break
        
        # Time to resolve the anomaly (exponentially distributed)
        resolution_time = np.random.exponential(scale=1/lambda_resolution)
        resolved_time = anomaly_time + timedelta(hours=resolution_time)

        anomaly_type = random.choice(anomaly_types)
        if test_anomaly:
            anomaly_type = test_anomaly

        anomaly_dict[anomaly_type].get_instance()
        anomaly_dict[anomaly_type].apply_transformation(wireless_dataframe, anomaly_time, resolved_time)
        
        anomaly_occurrence_list.append((anomaly_time, resolved_time))
        anomaly_list.append((anomaly_type, (anomaly_time, resolved_time)))
        current_time = resolved_time
 
    if test_anomaly:
        save_plots(wireless_dataframe, old_df, anomaly_occurrence_list, save_dir = "presentation_plots", anomaly=test_anomaly)
    else:
        save_plots(wireless_dataframe, old_df, anomaly_occurrence_list, save_dir = "presentation_plots")
    return wireless_dataframe, anomaly_occurrence_list, anomaly_list

def get_unique_filename(file_path):
    if not os.path.exists(file_path):
        return file_path
    base, ext = os.path.splitext(file_path)
    i = 1
    while True:
        new_path = f"{base}_v{i}{ext}"
        if not os.path.exists(new_path):
            return new_path
        i += 1

def save_plots(wireless_dataframe, old_df, anomaly_occurence_list, anomaly=None, save_dir="metric_plots_sim"):
    metrics = [
        "UL_NumberOfPackets", "DL_NumberOfPackets", "UL_BLER", "DL_BLER", "RSRP", "UL_SNR", #"DL_SNR",
        "Estimated_UL_Buffer", "TX_Bytes", "RX_Bytes", "PRB_Utilization_DL", "PRB_Utilization_UL",
        "PRBs_DL_Current", "PRBs_UL_Current", "DL_MCS", "UL_MCS", "UL_NPRB"
    ]
    os.makedirs(save_dir, exist_ok=True)
    wireless_dataframe['date'] = pd.to_datetime(wireless_dataframe['date'])
    
    wireless_dataframe=wireless_dataframe.iloc[:2000]
    old_df = old_df.iloc[:2000]

    if anomaly != None:
        metrics = [metric for metric in metrics if metric in anomaly_dict[anomaly].metric_attributes.keys()]
    
    for metric in metrics:
        plt.figure(figsize=(10, 4))

        plt.plot(
            wireless_dataframe['date'], wireless_dataframe[metric],
            label=f"Transformed - {metric}", color='blue', alpha = 0.9, linestyle='-'
        )
        plt.plot(
            old_df['date'], old_df[metric],
            label=f"Original - {metric}", color='red', alpha = 0.9, linestyle='--'
        )
        plt.xlabel("Time")
        plt.ylabel(metric)

        if anomaly != None: 
            plt.title(f"{anomaly} {metric} Over Time")
        else:
            plt.title(f"{metric} Over Time")

        plt.grid(True)
        plt.xticks(rotation=45)

        time_min, time_max = wireless_dataframe['date'].min(), wireless_dataframe['date'].max()
        for start, end in anomaly_occurence_list:
            start = pd.to_datetime(start)
            end = pd.to_datetime(end)
            if end >= time_min and start <= time_max:
                clipped_start = max(start, time_min)
                clipped_end = min(end, time_max)
                plt.axvspan(clipped_start, clipped_end, color='red', alpha=0.2)
    
        if anomaly != None:
            file_path = os.path.join(save_dir, f"{anomaly}/{anomaly} {metric}.png")
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
        else:
            file_path = os.path.join(save_dir, f"{metric}.png")
        unique_path = get_unique_filename(file_path)
        plt.savefig(unique_path, bbox_inches='tight')
        plt.close()

anomaly_dict = {
    "Antenna Failure": Anomaly("Antenna Failure", {
        "RSRP": ("c_add", (-10, -20)),  
        "UL_SNR": ("c_add", (-5, -15)),  
        "DL_BLER": ("exp_growth", (0.035, 0.05)), 
        "UL_BLER": ("exp_growth", (0.035, 0.045)),  
        "DL_MCS": ("c_add", (-5, -15)),  
        "UL_MCS": ("c_add", (-5, -15)),  
        "PRBs_DL_Current": ("c_multiply", (0.2, 0.7)), 
        "PRBs_UL_Current": ("c_multiply", (0.2, 0.5)),  
        "TX_Bytes": ("log_decay", (0.2, 0.3)), 
        "RX_Bytes": ("log_decay", (0.2, 0.3)),  
        "Estimated_UL_Buffer": ("exp_growth", (0.06, 0.075)), 
        "UL_NumberOfPackets": ("c_multiply", (1.5, 3)),
        "DL_NumberOfPackets": ("c_multiply", (1.3, 2)), 
    }),
    "Co-Channel Interference (Mild)": Anomaly("Co-Channel Interference (Mild)", {
        "RSRP": ("c_add", (-3, -8)), 
        "UL_SNR": ("c_add", (-1, -3)),
        "DL_BLER": ("c_multiply", (1.1, 1.4)), 
        "UL_BLER": ("c_multiply", (1.1, 1.4)),  
        "TX_Bytes": ("c_multiply", (0.85, 0.95)), 
        "RX_Bytes": ("c_multiply", (0.85, 0.95)),  
        "PRB_Utilization_DL": ("c_multiply", (1.05, 1.2)), 
        "PRB_Utilization_UL": ("c_multiply", (1.05, 1.2)),  
        "UL_NPRB": ("c_multiply", (1.05, 1.2)) 
    }),
    "Co-Channel Interference (Severe)": Anomaly("Co-Channel Interference (Severe)", {
        "RSRP": ("c_add", (-8, -15)), 
        "UL_SNR": ("c_add", (-8, -15)), 
        "DL_BLER": ("c_multiply", (1.5, 3)), 
        "UL_BLER": ("c_multiply", (1.5, 3)),  
        "TX_Bytes": ("c_multiply", (0.2, 0.7)), 
        "RX_Bytes": ("c_multiply", (0.2, 0.7)),  
        "PRB_Utilization_DL": ("c_multiply", (1.3, 1.8)), 
        "PRB_Utilization_UL": ("c_multiply", (1.3, 1.8)),  
        "UL_NumberOfPackets": ("c_multiply", (0.3, 0.6)), 
        "DL_NumberOfPackets": ("c_multiply", (0.3, 0.6)), 
        "UL_NPRB": ("c_multiply", (1.2, 1.7))
    }),
    "Faulty RF Filters (Temporal)": Anomaly("Faulty RF Filters (Temporal)", {
        "RSRP": ("linear", (-0.2, -0.5)), 
        "UL_SNR": ("linear", (-0.2, -0.35)), 
        "DL_BLER": ("exp_growth", (0.025, 0.035)), 
        "UL_BLER": ("exp_growth", (0.02, 0.03)),  
        "TX_Bytes": ("log_decay", (0.012, 0.2)), 
        "RX_Bytes": ("log_decay", (0.012, 0.2)),  
        "PRB_Utilization_DL": ("exp_growth", (0.006, 0.009)), 
        "PRB_Utilization_UL": ("exp_growth", (0.006, 0.009)), 
        "UL_NPRB": ("linear", (0.02, 0.05)) 
    }),
    "Doppler Shift (Severe)": Anomaly("Doppler Shift (Severe)", {
        "RSRP": ("sinusoidal_add", (-8, -3), (2, 3), (-2, -4)),
        "UL_SNR": ("sinusoidal_add", (-5, -2), (2, 3), (-1, -3)), 
        "DL_MCS": ("c_add", (-1, -2)), 
        "UL_MCS": ("c_add", (-1, -2)),  
        "TX_Bytes": ("sinusoidal_multiply", (0.2, 0.4), (2, 3)), 
        "RX_Bytes": ("sinusoidal_multiply", (0.2, 0.4), (2, 3)),  
        "UL_NPRB": ("sinusoidal_multiply", (0.3, 0.6), (2, 3))
    }),
    "Faulty Handover Algorithm (Too Frequent)": Anomaly("Faulty Handover Algorithm (Too Frequent)", {
        "RSRP": ("sinusoidal_add", (-5, -2), (0.1, 0.3)), 
        "UL_SNR": ("sinusoidal_add", (-3, -1), (0.1, 0.3)),  
        "DL_BLER": ("c_multiply", (1.3, 2.5)), 
        "UL_BLER": ("c_multiply", (1.3, 2.5)),  
        "TX_Bytes": ("c_multiply", (0.5, 0.9)), 
        "RX_Bytes": ("c_multiply", (0.5, 0.9)),  
        "Estimated_UL_Buffer": ("c_multiply", (1.5, 3.0)), 
        "UL_NumberOfPackets": ("c_multiply", (1.5, 3.0)), 
        "DL_NumberOfPackets": ("c_multiply", (1.5, 3.0)),  
    }),
    "Buffer Overflow (Gradual Buildup)": Anomaly("Buffer Overflow (Gradual Buildup)", {
        "Estimated_UL_Buffer": ("exp_growth", (0.12, 0.2)), 
        "TX_Bytes": ("log_decay", (0.17, 0.2)), 
        "RX_Bytes": ("log_decay", (0.15, 0.18)), 
        "UL_NumberOfPackets": ("linear", (5, 10)), 
        "DL_NumberOfPackets": ("linear", (5, 10)),
        "UL_BLER": ("exp_growth", (0.03, 0.04)), 
        "DL_BLER": ("exp_growth", (0.03, 0.04)),  
        "PRB_Utilization_DL": ("logistic_growth", (0.07, 0.1)), 
        "PRB_Utilization_UL": ("logistic_growth", (0.06, 0.08)),
        "UL_NPRB": ("logistic_growth", (0.05, 0.08)),  
    }),
    "Resource Allocation Bugs": Anomaly("Resource Allocation Bugs", {
        "PRBs_DL_Current": ("sinusoidal_multiply", (0.5, 1), (0.3, 1)),
        "PRBs_UL_Current": ("sinusoidal_multiply", (0.5, 1), (0.3, 1)),  
        "PRB_Utilization_DL": ("sinusoidal_multiply", (0.3, 1), (0.3, 1)),
        "PRB_Utilization_UL": ("sinusoidal_multiply", (0.3, 1), (0.3, 1)),  
        "UL_BLER": ("c_multiply", (1.2, 2.0)),
        "DL_BLER": ("c_multiply", (1.2, 2.0)),  
        "TX_Bytes": ("sinusoidal_multiply", (0.4, 0.7), (0.3, 1), (0.1, 0.4)),
        "RX_Bytes": ("sinusoidal_multiply", (0.4, 0.7), (0.3, 1), (0.1, 0.4)), 
        "UL_NPRB": ("sinusoidal_multiply", (0.4, 0.9), (0.3, 1.0)),
    }),
    "High Network Congestion (Gradual Buildup)": Anomaly("High Network Congestion (Gradual Buildup)", {
        "Estimated_UL_Buffer": ("exp_growth", (0.05, 0.13)), 
        "PRB_Utilization_DL": ("logistic_growth", (0.04, 0.07)), 
        "PRB_Utilization_UL": ("logistic_growth", (0.03, 0.06)),  
        "UL_BLER": ("exp_growth", (0.017, 0.025)), 
        "DL_BLER": ("exp_growth", (0.017, 0.025)),  
        "TX_Bytes": ("log_decay", (0.11, 0.14)), 
        "RX_Bytes": ("log_decay", (0.1, 0.12)),  
        "UL_NumberOfPackets": ("exp_growth", (0.08, 0.13)), 
        "DL_NumberOfPackets": ("exp_growth", (0.08, 0.13)),  
        "UL_NPRB": ("logistic_growth", (0.03, 0.07)),
    }),
    "High Network Congestion (Sudden Spike)": Anomaly("High Network Congestion (Sudden Spike)", {
        "Estimated_UL_Buffer": ("c_multiply", (2.0, 5.0)), 
        "PRB_Utilization_DL": ("c_multiply", (1.3, 1.8)), 
        "PRB_Utilization_UL": ("c_multiply", (1.3, 1.8)),  
        "UL_BLER": ("c_multiply", (1.5, 2.5)), 
        "DL_BLER": ("c_multiply", (1.5, 2.5)),  
        "TX_Bytes": ("c_multiply", (0.5, 0.8)), 
        "RX_Bytes": ("c_multiply", (0.5, 0.8)),  
        "UL_NumberOfPackets": ("c_multiply", (0.4, 0.6)), 
        "DL_NumberOfPackets": ("c_multiply", (0.4, 0.6)),  
        "UL_NPRB": ("c_multiply", (1.4, 2.0)),
    })
}
