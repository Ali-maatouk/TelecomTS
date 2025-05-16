import pandas as pd
import glob
import os
import random
import re
import json
import math
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from typing import Dict, Tuple, List
from matplotlib import font_manager as fm
from anomaly_files.anomalies import Anomaly
from anomaly_files.anomaly_timestamps_sim import anomaly_dict
from anomaly_simulation_helper import load_anomaly_data, anomaly_to_index

import google.generativeai as genai
from openai import OpenAI, OpenAIError, APITimeoutError
import time
import anthropic
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY_2")
GOOGLE_GEMINI_API = ""
DEEPSEEK_API = ""

anomaly_evaluate_load = load_anomaly_data("anomaly_evaluate")
no_anomaly_evaluate_load = load_anomaly_data("no_anomaly_evaluate", anomaly=False)

random.seed(42)

accuracy_dataset = anomaly_evaluate_load
random.shuffle(accuracy_dataset)

anomaly_detect_dict = {anomaly: [sequence for sequence in anomaly_evaluate_load if sequence[3] == np.int64(anomaly_to_index[anomaly])][0:10] for anomaly in anomaly_dict.keys()}
anomaly_detect = []
for key, value in anomaly_detect_dict.items():
    anomaly_detect.extend(value)

no_anomaly_detect = no_anomaly_evaluate_load[0:100]
detect_dataset = anomaly_detect + no_anomaly_detect
random.shuffle(detect_dataset)

def anomaly_detection_prompt_generation(sequence, context=False):
    start_time, end_time = sequence[1]
    df = sequence[0]
    df = df.astype(str)
    metric_lines = [f"{col}:" + ' ' + ' '.join(df[col].tolist()) for col in df.columns]
    time_series_data = '\n'.join(metric_lines)
    seq_len = len(df)

    prompt = (
        "You are an AI assistant tasked with analyzing time-series data for anomalies in a wireless network. "
        "You will be provided with a time-series dataset containing various metrics and a specific time range to analyze. "
        "The time-series is sampled every 0.1 seconds (i.e. timestamps are a decisecond apart), and contains a total of {} time steps. "
        "Your goal is to detect any anomalies within this range and identify the timestamps where they occur. "
    ).format(seq_len)

    if context:
        prompt += (
            "Note: Wireless network data is naturally **noisy and erratic**, even under normal conditions. "
            "Sporadic spikes, sharp drops, or momentary fluctuations can appear **without indicating any true anomaly**. "
            "This sequence is only {:.1f} seconds long, so be especially cautious in interpreting short-term changes as significant. "
            "Only mark something as anomalous if there is **clear and sustained evidence** of abnormal behavior across multiple metrics."
        ).format(seq_len / 10)

    prompt += "\n\nFirst, review the time-series data provided from time {} to {}:\n\n".format(start_time, end_time)
    prompt += time_series_data

    if context:
        prompt += "\n\nTo detect anomalies, follow these steps:\n\n"
        prompt += "1. Begin by scanning the time-series for any unusual behavior: sharp spikes or drops, sustained deviations, or values inconsistent with the expected range.\n"
        prompt += "2. Consider inter-metric relationships — for example, whether high buffer utilization coincides with low throughput or high BLER.\n"
        prompt += "3. All anomalies occur at the same timestamp range, so you should identify a single set of timestamps for the anomaly event and attribute affected metrics to that period.\n"

    prompt += (
        "\n\nSummarize your conclusion as follows:\n\n"
        "```\n"
        "<conclusion>\n"
        "Anomaly Detected: [Yes/No]\n\n"
        "[If yes, include the following strictly formatted line:]\n"
        "Anomaly Timestamps: [(start_time1, end_time1), (start_time2, end_time2), ...]\n"
        "</conclusion>\n"
        "```\n\n"
        "Only base your analysis on the provided time range. If no anomaly is detected, write:\n"
        "```\n"
        "<conclusion>\n"
        "Anomaly Detected: No\n"
        "</conclusion>\n"
        "```\n"
        "Do not include additional comments or summaries outside this format."
    )

    return prompt

def anomaly_boundary_prompt(sequence):
    start_time, end_time = sequence[1]
    df = sequence[0]
    df = df.astype(str)
    metric_lines = [f"{col}:" + ' ' + ' '.join(df[col].tolist()) for col in df.columns]
    time_series_data = '\n'.join(metric_lines)
    seq_len = len(df)

    prompt = (
        "You are an AI assistant tasked with analyzing time-series data for anomalies in a wireless network. "
        "You will be provided with a time-series dataset containing various metrics and a specific time range to analyze. "
        "The time-series is sampled every 0.1 seconds (i.e. timestamps are a decisecond apart), and contains a total of {} time steps. "
        "Your goal is to identify a **single contiguous time interval** during which an anomaly occurs. "
        "There is exactly one anomaly in the data, and it may span the entire sequence or just a sub-segment.\n\n"

    ).format(seq_len, seq_len / 10)
    prompt += "First, review the time-series data provided from time {} to {}:\n\n".format(start_time, end_time)
    prompt += time_series_data

    prompt += (
        "\n\nSummarize your conclusion as follows:\n\n"
        "```\n"
        "<conclusion>\n"
        "Anomaly Timestamps: (YYYY-MM-DD HH:MM:SS.sss, YYYY-MM-DD HH:MM:SS.sss)\n"
        "</conclusion>\n"
        "```\n\n"
        "Do not include any additional commentary or explanation outside the specified format. Respond with *only* the <conclusion> block and nothing else."
    )

    return prompt

anomaly_descriptions = "Antenna Failure: RSRP decreases by 10.00 to 20.00. UL_SNR decreases by 5.00 to 15.00. DL_BLER experiences exponential growth with rate 0.035 to 0.050. UL_BLER experiences exponential growth with rate 0.035 to 0.045. DL_MCS decreases by 5.00 to 15.00. UL_MCS decrease by 5.00 to 15.00. PRBs_DL_Current decreases by 30.00% to 80.00%. PRBs_UL_Current decreases by 50.00% to 80.00%. TX_Bytes follows a logarithmic decay pattern with factor 0.200 to 0.30. RX_Bytes follows a logarithmic decay pattern with factor 0.200 to 0.30. Estimated_UL_Buffer experiences exponential growth with rate 0.060 to 0.075. UL_NumberOfPackets increases by 50.00% to 200.00%. DL_NumberOfPackets increases by 30.00% to 100.00%. \nCo-Channel Interference (Mild): RSRP decreases by 3.00 to 8.00. UL_SNR decreases by 1.00 to 3.00. DL_BLER increases by 10.00% to 40.00%. UL_BLER increases by 10.00% to 40.00%. TX_Bytes decreases by 5.00% to 15.00%. RX_Bytes decreases by 5.00% to 15.00%. PRB_Utilization_DL increases by 5.00% to 20.00%. PRB_Utilization_UL increases by 5.00% to 20.00%. UL_NPRB increases by 5.00% to 20.00%.\nCo-Channel Interference (Severe): RSRP decreases by 8.00 to 15.00. UL_SNR decreases by 8.00 to 15.00. DL_BLER increases by 50.00% to 200.00%. UL_BLER increases by 50.00% to 200.00%. TX_Bytes decreases by 30.00% to 80.00%. RX_Bytes decreases by 30.00% to 80.00%. PRB_Utilization_DL increases by 30.00% to 80.00%. PRB_Utilization_UL increases by 30.00% to 80.00%. UL_NumberOfPackets decreases by 40.00% to 70.00%. DL_NumberOfPackets decreases by 40.00% to 70.00%. UL_NPRB increases by 20.00% to 70.00%.\nFaulty RF Filters (Temporal): RSRP decreases linearly by 0.20 to 0.50 per time step. UL_SNR decreases linearly by 0.20 to 0.35 per time step. DL_BLER experiences exponential growth with rate 0.025 to 0.035. UL_BLER experiences exponential growth with rate 0.020 to 0.030. TX_Bytes follows a logarithmic decay pattern with factor 0.012 to 0.20. RX_Bytes follows a logarithmic decay pattern with factor 0.012 to 0.20. PRB_Utilization_DL experiences exponential growth with rate 0.006 to 0.009. PRB_Utilization_UL experiences exponential growth with rate 0.006 to 0.009. UL_NPRB decreases linearly by 0.02 to 0.05 per time step.\nDoppler Shift (Severe): RSRP fluctuates periodically with amplitude 3.00 to 8.00 and frequency 2.0000 to 3.0000 Hz. UL_SNR fluctuates periodically with amplitude 2.00 to 5.00 and frequency 2.0000 to 3.0000 Hz. DL_MCS decreases by 1.00 to 2.00. UL_MCS decreases by 1.00 to 2.00. TX_Bytes oscillates multiplicatively with amplitude factor 0.20 to 0.40 and frequency 2.0000 to 3.0000 Hz. RX_Bytes oscillates multiplicatively with amplitude factor 0.20 to 0.40 and frequency 2.0000 to 3.0000 Hz. UL_NPRB oscillates multiplicatively with amplitude factor 0.30 to 0.60 and frequency 2.0000 to 3.0000 Hz.\nFaulty Handover Algorithm (Too Frequent): RSRP fluctuates periodically with amplitude 2.00 to 5.00 and frequency 0.1000 to 0.3000 Hz. UL_SNR fluctuates periodically with amplitude 1.00 to 3.00 and frequency 0.1000 to 0.3000 Hz. DL_BLER increases by 30.00% to 150.00%. UL_BLER increases by 30.00% to 150.00%. TX_Bytes decreases by 10.00% to 50.00%. RX_Bytes decreases by 10.00% to 50.00%. Estimated_UL_Buffer increases by 50.00% to 200.00%. UL_NumberOfPackets increases by 50.00% to 200.00%. DL_NumberOfPackets increases by 50.00% to 200.00%.\nBuffer Overflow (Gradual Buildup): Estimated_UL_Buffer experiences exponential growth with rate 0.120 to 0.200. TX_Bytes follows a logarithmic decay pattern with factor 0.170 to 0.20. RX_Bytes follows a logarithmic decay pattern with factor 0.150 to 0.18. UL_NumberOfPackets decreases linearly by 5.00 to 10.00 per time step. DL_NumberOfPackets decreases linearly by 5.00 to 10.00 per time step. UL_BLER experiences exponential growth with rate 0.030 to 0.040. DL_BLER experiences exponential growth with rate 0.030 to 0.040. PRB_Utilization_DL experiences logistic growth with rate 0.070 to 0.100. PRB_Utilization_UL experiences logistic growth with rate 0.060 to 0.080. UL_NPRB experiences logistic growth with rate 0.050 to 0.080.\nResource Allocation Bugs: PRBs_DL_Current oscillates multiplicatively with amplitude factor 0.50 to 1.00 and frequency 0.3000 to 1.0000 Hz. PRBs_UL_Current oscillates multiplicatively with amplitude factor 0.50 to 1.00 and frequency 0.3000 to 1.0000 Hz. PRB_Utilization_DL oscillates multiplicatively with amplitude factor 0.30 to 1.00 and frequency 0.3000 to 1.0000 Hz. PRB_Utilization_UL oscillates multiplicatively with amplitude factor 0.30 to 1.00 and frequency 0.3000 to 1.0000 Hz. UL_BLER increases by 20.00% to 100.00%. DL_BLER increases by 20.00% to 100.00%. TX_Bytes oscillates multiplicatively with amplitude factor 0.40 to 0.70 and frequency 0.3000 to 1.0000 Hz. RX_Bytes oscillates multiplicatively with amplitude factor 0.40 to 0.70 and frequency 0.3000 to 1.0000 Hz. UL_NPRB oscillates multiplicatively with amplitude factor 0.40 to 0.90 and frequency 0.3000 to 1.0000 Hz.\nHigh Network Congestion (Gradual Buildup): Estimated_UL_Buffer experiences exponential growth with rate 0.050 to 0.130. PRB_Utilization_DL experiences logistic growth with rate 0.040 to 0.070. PRB_Utilization_UL experiences logistic growth with rate 0.030 to 0.060. UL_BLER experiences exponential growth with rate 0.017 to 0.025. DL_BLER experiences exponential growth with rate 0.017 to 0.025. TX_Bytes follows a logarithmic decay pattern with factor 0.110 to 0.14. RX_Bytes follows a logarithmic decay pattern with factor 0.100 to 0.12. UL_NumberOfPackets experiences exponential growth with rate 0.080 to 0.130. DL_NumberOfPackets experiences exponential growth with rate 0.080 to 0.130. UL_NPRB experiences logistic growth with rate 0.030 to 0.070.\nHigh Network Congestion (Sudden Spike): Estimated_UL_Buffer increases by 100.00% to 400.00%. PRB_Utilization_DL increases by 30.00% to 80.00%. PRB_Utilization_UL increases by 30.00% to 80.00%. UL_BLER increases by 50.00% to 150.00%. DL_BLER increases by 50.00% to 150.00%. TX_Bytes decreases by 20.00% to 50.00%. RX_Bytes decreases by 20.00% to 50.00%. UL_NumberOfPackets decreases by 40.00% to 60.00%. DL_NumberOfPackets decreases by 40.00% to 60.00%. UL_NPRB increases by 40.00% to 100.00%.\n\n"

def root_cause_analysis(sequence, descriptions = False, uniform_prior=False):
    start_time, end_time = sequence[1]
    df = sequence[0]
    df = df.astype(str)
    metric_lines = [f"{col}:" + ' ' + ' '.join(df[col].tolist()) for col in df.columns]
    time_series_data = '\n'.join(metric_lines)
    seq_len = len(df)

    anomaly_list = [
        "Antenna Failure", "Co-Channel Interference (Mild)",
        "Co-Channel Interference (Severe)", "Faulty RF Filters (Temporal)", "Doppler Shift (Severe)",
        "Faulty Handover Algorithm (Too Frequent)", "Buffer Overflow (Gradual Buildup)", 
        "High Network Congestion (Gradual Buildup)", "High Network Congestion (Sudden Spike)", 
        "Resource Allocation Bugs"
    ]

    prompt = (
        "You are an AI assistant tasked with diagnosing a known anomaly in wireless network time-series data. "
        "You will be provided with a short time-series segment sampled every 0.1 seconds, covering {:.1f} seconds and {} time steps. "
        "This sequence ranges from {} to {} and **is confirmed to contain an anomaly**.\n\n"
    ).format(seq_len / 10, seq_len, start_time, end_time)

    if uniform_prior:
        prompt += (
            "The anomaly is known to be **one of the following**, and each is equally likely to occur in this dataset. "
            "**Do not assume any anomaly is more common or more likely than another.**\n\n"
        )

    prompt += (
        "Your task is to identify the most plausible anomaly type **from the following list**:\n\n"
        "{}\n\n"
        "Please analyze the metrics below and select the **single most likely anomaly**.\n\n"
    ).format(', '.join(anomaly_list))

    if descriptions == True:
        prompt += "Here is a summary on how the provided anomalies generally behave:\n\n"
        prompt += anomaly_descriptions

    prompt += "Here is the time-series data:\n\n" + time_series_data + "\n\n"
    prompt += (
        "Summarize your conclusions as follows:\n\n"
        "```\n"
        "<conclusion>\n"
        "Anomaly Type: [One exact string from the predefined anomaly list.]\n"
        "</conclusion>\n"
        "```\n\n"
        "Do not include any additional commentary or explanation outside the specified format. Respond with *only* the <conclusion> block and nothing else."
    )

    return prompt

def root_cause_analysis(sequence, descriptions = False, uniform_prior=False):
    start_time, end_time = sequence[1]
    df = sequence[0]
    df = df.astype(str)
    metric_lines = [f"{col}:" + ' ' + ' '.join(df[col].tolist()) for col in df.columns]
    time_series_data = '\n'.join(metric_lines)
    seq_len = len(df)

    anomaly_list = [
        "Antenna Failure", "Co-Channel Interference (Mild)",
        "Co-Channel Interference (Severe)", "Faulty RF Filters (Temporal)", "Doppler Shift (Severe)",
        "Faulty Handover Algorithm (Too Frequent)", "Buffer Overflow (Gradual Buildup)", 
        "High Network Congestion (Gradual Buildup)", "High Network Congestion (Sudden Spike)", 
        "Resource Allocation Bugs"
    ]

    prompt = (
        "You are an AI assistant tasked with diagnosing a known anomaly in wireless network time-series data. "
        "You will be provided with a short time-series segment sampled every 0.1 seconds, covering {:.1f} seconds and {} time steps. "
        "This sequence ranges from {} to {} and **is confirmed to contain an anomaly**.\n\n"
    ).format(seq_len / 10, seq_len, start_time, end_time)

    if uniform_prior:
        prompt += (
            "The anomaly is known to be **one of the following**, and each is equally likely to occur in this dataset. "
            "**Do not assume any anomaly is more common or more likely than another.**\n\n"
        )

    prompt += (
        "Your task is to identify the most plausible anomaly type **from the following list**:\n\n"
        "{}\n\n"
        "Please analyze the metrics below and select the **single most likely anomaly**.\n\n"
    ).format(', '.join(anomaly_list))

    if descriptions == True:
        prompt += "Here is a summary on how the provided anomalies generally behave:\n\n"
        prompt += anomaly_descriptions

    prompt += "Here is the time-series data:\n\n" + time_series_data + "\n\n"
    prompt += (
        "Summarize your conclusions as follows:\n\n"
        "```\n"
        "<conclusion>\n"
        "Anomaly Type: [One exact string from the predefined anomaly list.]\n"
        "</conclusion>\n"
        "```\n\n"
        "Do not include any additional commentary or explanation outside the specified format. Respond with *only* the <conclusion> block and nothing else."
    )

    return prompt

def send_to_openai_chatgpt(content, model="gpt-4.1", max_tokens=2000, temperature=0.7, timeout=30, open_router=False):
    if open_router:
        client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY)
    else:
        client = OpenAI(api_key=OPENAI_API_KEY)

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": content}],
        max_tokens=max_tokens,
        temperature=temperature,
        timeout=timeout
    )
    return response.choices[0].message

def send_to_openai_reasoning(content, model="o4-mini", max_completion_tokens=20000, timeout=90, retries=3, open_router=False):
    if open_router:
        client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY)
    else:
        client = OpenAI(api_key=OPENAI_API_KEY)

    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": content}],
                max_completion_tokens=max_completion_tokens,
                timeout=timeout
            )
            return response.choices[0].message
        except APITimeoutError as e:
            print(f"[Timeout] Attempt {attempt+1}/{retries}: {e}")
            time.sleep(2 ** attempt)
        except OpenAIError as e:
            print(f"[OpenAIError] {e}")
            break
        except Exception as e:
            print(f"[Unexpected Error] {e}")
            break
    return None

def send_to_anthropic_claude(content, model = "claude-3-7-sonnet-latest"):
    client = anthropic.Anthropic(
        api_key=ANTHROPIC_API_KEY,
    )
    message = client.messages.create(
        model=model,
        max_tokens=4096,
        messages=[
            {"role": "user", "content": content}
        ]
    )
    return message.content[0].text

def send_to_google_gemini(content):
            # Configure the API key
    client = genai.Client(api_key=GOOGLE_GEMINI_API)

    response = client.models.generate_content(
        model='gemini-2.0-flash',
        contents=content,
    )
    return response.candidates[0].content.parts[0].text  # Extract the text output

def send_to_deepseek(content):
    client = OpenAI(api_key=DEEPSEEK_API, base_url='https://api.siliconflow.cn/v1/')
    response = client.chat.completions.create(
        model="deepseek-ai/DeepSeek-V3",
        messages=[
            {"role": "user", "content": content},
    ],
        max_tokens=4096,
        stream=False
    )

    return response.choices[0].message.content

def send_to_deepseek_r1(content, open_router=False):
    if open_router:
        client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY)
        model_name = "deepseek/deepseek-r1"
    else:
        client = OpenAI(api_key=DEEPSEEK_API, base_url='https://api.siliconflow.cn/v1/')
        model_name = "deepseek-ai/DeepSeek-R1"

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "user", "content": content},
    ],
        max_tokens=4096,
        stream=False
    )

    return response.choices[0].message.content

def send_to_openai_o1(content):
    client = OpenAI(api_key=OPENAI_API_KEY)
    # Prepare the request
    response = client.chat.completions.create(
        model="o1",
        messages=[{"role": "user", "content": content}],
        timeout=30
    )
    return response.choices[0].message

def send_to_llama(content, pipeline = None):
    messages = [
        {"role": "user", "content": content},
    ]
    outputs = pipeline(
        messages,
        max_new_tokens=1024,
    )

    return outputs[0]["generated_text"][-1]["content"].content

def parse_conclusion_section(full_text):
    match = re.search(r"<conclusion>(.*?)</conclusion>", full_text, re.DOTALL)
    if not match:
        return (False, [])

    conclusion_block = match.group(1)

    detected_match = re.search(r"Anomaly Detected:\s*(Yes|No)", conclusion_block)
    detected = detected_match and detected_match.group(1).strip().lower() == "yes"

    if detected:
        timestamp_pairs = re.findall(
            r"\(\s*(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{1,10})\s*,\s*(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{1,10})\s*\)",
            conclusion_block
        )
        return True, timestamp_pairs
    else:
        return False, []
    
from datetime import datetime
def parse_anomaly_datetime_conclusion(full_text):
    match = re.search(r"<conclusion>(.*?)</conclusion>", full_text, re.DOTALL)
    if not match:
        return False, "No <conclusion> block found."

    conclusion_block = match.group(1).strip()

    timestamp_match = re.search(
        r"Anomaly Timestamps:\s*\[?\(\s*(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+)\s*,\s*(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+)\s*\)\]?",
        conclusion_block
    )

    if timestamp_match:
        try:
            start_time = datetime.strptime(timestamp_match.group(1), "%Y-%m-%d %H:%M:%S.%f")
            end_time = datetime.strptime(timestamp_match.group(2), "%Y-%m-%d %H:%M:%S.%f")
            return True, [(start_time, end_time)]
        except ValueError as ve:
            print(full_text)
            return False, f"Datetime parsing error: {ve}"
    else:
        print(full_text)
        return False, "No valid datetime timestamp pair found in conclusion block."

valid_causes = {
    'Antenna Failure',
    'Co-Channel Interference (Mild)',
    'Co-Channel Interference (Severe)',
    'Faulty RF Filters (Temporal)',
    'Doppler Shift (Severe)',
    'Faulty Handover Algorithm (Too Frequent)',
    'Buffer Overflow (Gradual Buildup)',
    'Resource Allocation Bugs',
    'High Network Congestion (Gradual Buildup)',
    'High Network Congestion (Sudden Spike)'
}

def parse_conclusion_to_tuple(text):
    match = re.search(r"<conclusion>(.*?)</conclusion>", text, re.DOTALL)
    if not match:
        return (False, math.nan, math.nan)

    block = match.group(1)

    cause_match = re.search(r"Anomaly Type:\s*(.+)", block)
    solution_match = re.search(r"Recommended Solution:\s*(.+)", block)

    if cause_match:
        cause = cause_match.group(1).strip()
    else:
        # fallback: use entire block as cause if label is missing
        lines = block.strip().split("\n")
        cause = lines[1].strip() if len(lines) > 1 else lines[0].strip()

    if cause not in valid_causes:
        for valid in valid_causes:
            if valid in text:
                cause = valid
                break

    return cause

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

def timestep_based_f_score(anomaly_timestamps, predicted_timestamps, start_end_range, processed_anomalies = False, freq='100ms'):
    #confirm that each range length matches the number of timestamps in provided lists
    if len(anomaly_timestamps) != len(predicted_timestamps) or len(anomaly_timestamps) != len(start_end_range):
        raise ValueError("Mismatch in lengths of anomaly timestamps, predicted timestamps, and start-end ranges.")
    
    all_actual = []
    all_pred = []

    for i, (gt_intervals, pred_intervals) in enumerate(zip(anomaly_timestamps, predicted_timestamps)):
        start, end = start_end_range[i]
        start = pd.to_datetime(start)
        end = pd.to_datetime(end)
        time_index = pd.date_range(start=start, end=end, freq=freq)

        if processed_anomalies == False:
            actual = convert_intervals_to_binary(gt_intervals, time_index)
        else:
            actual = gt_intervals
        pred = convert_intervals_to_binary(pred_intervals, time_index)
        all_actual.append(actual)
        all_pred.append(pred)

    actual = np.concatenate(all_actual)
    pred = np.concatenate(all_pred)

    TP = np.sum(pred * actual)
    FP = np.sum(pred * (1 - actual))
    FN = np.sum((1 - pred) * actual)
    print("TP ", TP, " FP ", FP, " FN ", FN)

    precision = TP / (TP + FP + 1e-5)
    recall = TP / (TP + FN + 1e-5)
    f1 = 2 * precision * recall / (precision + recall + 1e-5)

    return f1, precision, recall

def send_to_api(prompt, provider):
    if provider == "openai":
        return send_to_openai_chatgpt(prompt, model="gpt-4.1").content
    elif provider == "openai_reasoning":
        return send_to_openai_reasoning(prompt, model="o4-mini").content
    elif provider == "anthropic":
        return send_to_anthropic_claude(prompt, model="claude-3-7-sonnet-latest")
    elif provider == "deepseek":
        return send_to_deepseek_r1(prompt, open_router=True)
    elif provider == "openrouter_openai":
        return send_to_openai_chatgpt(prompt, open_router=True).content
    elif provider == "openrouter_o4":
        return send_to_openai_reasoning(prompt, open_router=True).content
    
#evaluate f1 on sequences with confirmed anomalies
def evaluate_detection_f1(sequence, provider="openai"):
    if (np.all(sequence[2] == 0)):
        print("No anomalies detected in the sequence.")
        return None
    
    if provider == "openai":
        llm_response = send_to_openai_chatgpt(anomaly_detection_prompt_generation(sequence), model="gpt-4.1")
    parsed_response = parse_conclusion_section(llm_response.content)
    start_time = sequence[1][0]
    end_time = sequence[1][1]
    return timestep_based_f_score([sequence[2]], [parsed_response[1]], [(start_time, end_time)], processed_anomalies=True, freq='100ms')

def evaluate_detection_f1_all(sequences, provider="openai"):
    ground_truth_anomalies = []
    predicted_anomalies = []
    start_end_range = []
    anomalies_not_detected = []
    for i, sequence in enumerate(tqdm(sequences, desc="Evaluating sequences")):
        print("ID: ", sequence[4])
        if (np.all(sequence[2] == 0)):
            anomalies_not_detected.append(i)
            continue
        prompt = anomaly_boundary_prompt(sequence)
        llm_response = send_to_api(prompt, provider)
        parsed_response = parse_anomaly_datetime_conclusion(llm_response)
        if parsed_response[0] == False or parsed_response[0] == "False":
            print("Faulty Response")
            print(llm_response)
            continue
        ground_truth_anomalies.append(sequence[2])
        predicted_anomalies.append(parsed_response[1])
        start_end_range.append(sequence[1])
        f1, precision, recall = timestep_based_f_score(
                ground_truth_anomalies, predicted_anomalies, start_end_range,
                processed_anomalies=True, freq='100ms'
            )
        
        if (i + 1) % 10 == 0:
            start_end = sequence[1]
            start = pd.to_datetime(start_end[0])
            end = pd.to_datetime(start_end[1])
            time_index = pd.date_range(start=start, end=end, freq="100ms")
            f1, precision, recall = timestep_based_f_score(
                ground_truth_anomalies, predicted_anomalies, start_end_range,
                processed_anomalies=True, freq='100ms'
            )
            print(f"[Checkpoint {i+1}] F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")

    f1, precision, recall = timestep_based_f_score(ground_truth_anomalies, predicted_anomalies, start_end_range, processed_anomalies=True, freq='100ms')
    print(f"F1: {f1}, Precision: {precision}, Recall: {recall}")

    return f1, precision, recall

def evaluate_detection_accuracy(sequence, provider="openai"):
    prompt = anomaly_detection_prompt_generation(sequence)
    llm_response = send_to_api(prompt, provider)

    parsed_response = parse_conclusion_section(llm_response)
    if (np.all(sequence[2] == 0)):
        if (parsed_response[0] == False):
            return "TN"
        else:
            return "FP"
    else:
        if (parsed_response[0] == True):
            return "TP"
        else:
            return "FN"

def evaluation_detection_accuracy_all(sequences, provider="openai"):
    total_predictions = len(sequences)
    true_pos = 0
    false_pos = 0
    true_neg = 0
    false_neg = 0

    for i, sequence in enumerate(tqdm(sequences, desc="Evaluating sequences")):
        print("ID: ", sequence[4])
        result = evaluate_detection_accuracy(sequence, provider=provider)
        if (result == "TP"):
            true_pos += 1
        elif (result == "FP"):
            false_pos += 1
        elif (result == "TN"):
            true_neg += 1
        elif (result == "FN"):
            false_neg += 1
        precision = true_pos / (true_pos + false_pos + 1e-5)
        recall = true_pos / (true_pos + false_neg + 1e-5)
        f1 = 2 * precision * recall / (precision + recall + 1e-5)
        accuracy = (true_pos + true_neg) / total_predictions
        print(f"F1: {f1}, Precision: {precision}, Recall: {recall}, Accuracy: {accuracy}")

    print("TP ", true_pos, " FP ", false_pos, " TN ", true_neg, " FN ", false_neg)

    precision = true_pos / (true_pos + false_pos + 1e-5)
    recall = true_pos / (true_pos + false_neg + 1e-5)
    f1 = 2 * precision * recall / (precision + recall + 1e-5)
    accuracy = (true_pos + true_neg) / total_predictions
    print(f"F1: {f1}, Precision: {precision}, Recall: {recall}, Accuracy: {accuracy}")
    return accuracy

def evaluate_anomaly_accuracy(sequence, provider="openai", descriptions=False, uniform_prior=False):
    if (np.all(sequence[2] == 0)):
        print("No anomalies detected in the sequence.")
        return None

    prompt = root_cause_analysis(sequence, descriptions=descriptions, uniform_prior=uniform_prior)
    llm_response = send_to_api(prompt, provider)
    parsed_response = parse_conclusion_to_tuple(llm_response)

    if parsed_response not in anomaly_to_index.keys():
        return None
    detected_anomaly = anomaly_to_index[parsed_response]
    true_anomaly = sequence[3]
    return detected_anomaly, true_anomaly

def evaluate_anomaly_accuracy_all(sequences, provider="openai", descriptions=False, uniform_prior=False):
    correct_predictions = 0
    total_predictions = 0

    detected_counter = Counter()
    true_counter = Counter()

    for i, sequence in enumerate(tqdm(sequences, desc="Evaluating sequences")):
        result = evaluate_anomaly_accuracy(sequence, provider, descriptions, uniform_prior)
        if result is None:
            print("Incomplete Detection")
            continue
        
        detected_anomaly, true_anomaly = result

        if math.isnan(detected_anomaly) or math.isnan(true_anomaly):
            print("Incomplete Detection")
            continue
    
        detected_counter[detected_anomaly] += 1
        true_counter[true_anomaly] += 1

        total_predictions += 1
        if detected_anomaly == true_anomaly:
            correct_predictions += 1

    if total_predictions == 0:
        print("No valid predictions made.")
        return None
    accuracy = correct_predictions / total_predictions

    index_to_anomaly = {v: k for k, v in anomaly_to_index.items()}
    detected_dict = {index_to_anomaly[k]: v for k, v in detected_counter.items()}
    true_dict = {index_to_anomaly[k]: v for k, v in true_counter.items()}

    print(f"Accuracy: {accuracy:.4f}")
    print("Detected Anomaly Counts:", detected_dict)
    print("True Anomaly Counts:", true_dict)

    return accuracy