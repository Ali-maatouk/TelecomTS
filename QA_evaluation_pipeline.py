import json
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import os
from scipy.fft import rfft, rfftfreq
from openai import OpenAI, OpenAIError, APITimeoutError
import time
import anthropic
import re

#number of series to truncate analysis to
num_series_analyze = 2

#llm to evaluate on
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

provider = "openai_gpt"

#Load in data
file_name = 'anomaly_data/no_anomaly_evaluate_formatted.json'

with open(file_name, "r") as file:
    data = json.load(file)

#llm evaluation helper functions
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
    return response.choices[0].message.content

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
            return response.choices[0].message.content
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

def send_to_deepseek_r1(content, open_router=False):
    if open_router:
        client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY)
    else:
        client = OpenAI(api_key=DEEPSEEK_API, base_url='https://api.siliconflow.cn/v1/')

    response = client.chat.completions.create(
        model="deepseek/deepseek-r1",
        messages=[
            {"role": "user", "content": content},
    ],
        max_tokens=4096,
        stream=False
    )

    return response.choices[0].message.content

def send_to_llm(content, provider):
    if provider == "openai_gpt":
        return send_to_openai_chatgpt(content)
    elif provider == "openai_o4_mini":
        return send_to_openai_reasoning(content)
    elif provider == "anthropic":
        return send_to_anthropic_claude(content)
    elif provider == "deepseek":
        return send_to_deepseek_r1(content, open_router=True)
    
def extract_number(response):
    match = re.search(r"[-+]?(?:\d*\.\d+|\d+)", response)
    return np.float64(match.group()) if match else None

#Initialize baseline dataframes 
means_by_channel = []
variances_by_channel = []

channels = [ch for ch in data[0]['dataframe'][0].keys() if ch not in ["UL_Protocol", "DL_Protocol"]]
trend_by_channel = {ch: [] for ch in data[0]['dataframe'][0].keys() if ch not in ["UL_Protocol", "DL_Protocol"]}
periodicity_by_channel = {ch: [] for ch in data[0]['dataframe'][0].keys() if ch not in ["UL_Protocol", "DL_Protocol"]}

#Periodicity calculation
def periodicity_calculation(y):
    yf = np.abs(rfft(y - y.mean())) #generates frequencies of ith component of yf
    xf = rfftfreq(len(y), d=1)  # d=1 assuming unit sample spacing, find dominant frequency index, skipping the zero component (representing the mean), adding 1 since we skipped the first component
    peak_freq_idx = np.argmax(yf[1:]) + 1  # ignore the zero-frequency component, using this index, find the dominant period
    dominant_freq = xf[peak_freq_idx] #TODO Look into periodicity distribution 
    if dominant_freq > 0:
        period = round(1 / dominant_freq, 2)
        periodicity = period #strong periodicity detected every period periods
    else:
        periodicity = 0
    return periodicity

#Establishing baseline loop
to_print = False
for entry in data:
    df = pd.DataFrame(entry['dataframe'])
    
    #Parse None values
    df.replace("None", np.nan, inplace=True)

    #Convert string to numeric
    df = df.apply(pd.to_numeric, errors='coerce')
    means_by_channel.append(df.mean())
    variances_by_channel.append(df.var())

    x = np.arange(len(df))
    #get slopes 
    for channel in df.columns:
        if channel in ["UL_Protocol", "DL_Protocol"]:
            continue
        y = df[channel].values

        if np.all(np.isnan(y)):
            trend_by_channel[channel].append(np.nan)
            periodicity_by_channel[channel].append(np.nan)
            continue

        slope, _ = np.polyfit(x, y, 1)
        trend_by_channel[channel].append(slope)
        periodicity = periodicity_calculation(y)
        periodicity_by_channel[channel].append(periodicity)
    if not to_print:
        print(entry)
        print(df.head(40))
        print(means_by_channel)
        to_print=True

#Convert to dataframes
means_df = pd.DataFrame(means_by_channel, columns=channels)
variances_df = pd.DataFrame(variances_by_channel, columns=channels)
trend_df = pd.DataFrame(trend_by_channel)
periodicity_df = pd.DataFrame(periodicity_by_channel)

mean_slopes_by_channel = trend_df.mean()
std_slopes_by_channel = trend_df.std()

for channel in df.columns:
    if channel in ["UL_Protocol", "DL_Protocol"]:
        continue
    mean_slope=mean_slopes_by_channel[channel]
    std_slope=std_slopes_by_channel[channel]
    trend_df[channel] = trend_df[channel].apply(lambda x: 1 if x > mean_slope + std_slope else (-1 if x < mean_slope - std_slope else 0))

data = data[:num_series_analyze]
num_series = len(data)

#Ask ChatGPT
gpt_mean_by_channel = {ch: [np.nan]* num_series  for ch in data[0]['dataframe'][0].keys() if ch not in ["UL_Protocol", "DL_Protocol"]}
gpt_var_by_channel = {ch: [np.nan]* num_series for ch in data[0]['dataframe'][0].keys() if ch not in ["UL_Protocol", "DL_Protocol"]}
gpt_trend_by_channel = {ch: [np.nan]* num_series  for ch in data[0]['dataframe'][0].keys() if ch not in ["UL_Protocol", "DL_Protocol"]}
gpt_per_by_channel = {ch: [np.nan]* num_series  for ch in data[0]['dataframe'][0].keys() if ch not in ["UL_Protocol", "DL_Protocol"]}
#for each series(dataframe), and each channel, give chatgpt the series and ask the following quesiotns
#extract responses and store in dataframe 

trend_dict = {
    (channel, i): 0
    for channel in df.keys()
    if channel not in ['UL_Protocol', 'DL_Protocol']
    for i in (-1, 0, 1)
}

for i, entry in enumerate(data):
    print("Entry: ", i)
    df = pd.DataFrame(entry['dataframe'])
    #Parse None values
    df.replace("None", np.nan, inplace=True)

    #Convert string to numeric
    df = df.apply(pd.to_numeric, errors='coerce')

    for channel in df.columns:
        if channel not in ['UL_Protocol', "DL_Protocol"]:
            if trend_dict[(channel, int(trend_df[channel][i]))] != 5:
                trend_message = (f"Consider the following series: {y.tolist()}. "
                f"Please describe the average trend of the series, ignoring any NaN values. "
                f"If the series is decreasing on average, respond with a value of -1."
                "If it is increasing, respond with a value of 1. If there doesn't appear "
                "to be a strong trend in any direction, please respond with a value of 0. Note that wireless data can be noisy, so look at global changes to determine trend. "
                "Do not include any other numbers in your response whether in the form of "
                "intermediate calculations or steps. ONLY RESPOND WITH -1, 0, or 1. Please DO NOT include any other analysis or explanations. ")
                try:
                    response = send_to_llm(trend_message, provider=provider)
                    print("Trend")
                    print(response)
                    response = extract_number(response)
                    # print(response)
                except:
                    response = np.nan
                gpt_trend_by_channel[channel][i] = response
                trend_dict[(channel, int(trend_df[channel][i]))] += 1

        if channel in ["UL_Protocol", "DL_Protocol"]:
            continue
        y = df[channel].values
        message = (f"Consider the following list of numbers representing a time series: {y.tolist()}. "
        f"Some values may be missing (NaN). What is the average {channel} value of this series, "
        f"ignoring NaNs? Respond with only a single float rounded to 2 decimal places — no other text or numbers. Please DO NOT include any other analysis or explanations. "
        )
        try:
            response = send_to_llm(message, provider=provider)
            print("Mean")
            print(response)
            response = extract_number(response)
        except:
            response = np.nan
        gpt_mean_by_channel[channel][i] = response

        message = (f"Consider the following list of numbers representing a time series: {y.tolist()}. "
        f"Some values may be missing (NaN). What is the variance of {channel} for this series, "
        f"ignoring NaNs? Respond with only a single float rounded to 2 decimal places — no other text or numbers. Please DO NOT include any other analysis or explanations. "
        )
        try:
            response = send_to_llm(message, provider=provider)
            print("Variance")
            print(response)
            response = extract_number(response)
        except:
            response = np.nan
        gpt_var_by_channel[channel][i] = response

        seq_len = len(df)
        periodicity_message = (f"Consider the following series: {y.tolist()}. Please "
        f"investigate whether the series exhibits strong periodicity, ignoring any NaN values. If it does, please"
        f"respond with with an integer value representing approximately how often strong"
        f"periods occur in the series. If there is no evidence of strong periodicity please"
        f"respond with the sequence length {seq_len}. Do not include any other numbers in your response whether"
        f"in the form of intermediate calculations or steps. Remember you MUST return an INTEGER value or {seq_len}. Please DO NOT include any other analysis or explanations. ")
        try:
            response = send_to_llm(periodicity_message, provider=provider)
            response = extract_number(response)
        except:
            response = np.nan
        gpt_per_by_channel[channel][i] = response

gpt_mean_df = pd.DataFrame(gpt_mean_by_channel)
gpt_var_df = pd.DataFrame(gpt_var_by_channel)
gpt_trend_df = pd.DataFrame(gpt_trend_by_channel)
gpt_per_df = pd.DataFrame(gpt_per_by_channel)

means_df = means_df[:num_series_analyze]
variances_df = variances_df[:num_series_analyze]
trend_df = trend_df[:num_series_analyze]
periodicity_df = periodicity_df[:num_series_analyze]

squared_error_df_means = (gpt_mean_df - means_df) ** 2
squared_error_df_var = (gpt_var_df - variances_df) ** 2
squared_error_df_periodicity = (gpt_per_df - periodicity_df) ** 2

mse_by_channel_means = squared_error_df_means.mean()
mse_by_channel_var = squared_error_df_var.mean()
mse_by_channel_periodicity = squared_error_df_periodicity.mean()

abs_error_df_means = abs(gpt_mean_df - means_df) 
abs_error_df_var = abs(gpt_var_df - variances_df) 
abs_error_df_periodicity = abs(gpt_per_df - periodicity_df) 

mae_by_channel_means = abs_error_df_means.mean()
mae_by_channel_var = abs_error_df_var.mean()
mae_by_channel_periodicity = abs_error_df_periodicity.mean()

#Accuracy for Trend
#checking for NaNs
conditional_accuracies = {}
for label in [-1, 0, 1]:
    mask = (trend_df == label)
    correct = (gpt_trend_df == trend_df) & mask
    conditional_accuracy = correct.sum() / mask.sum()
    conditional_accuracies[label] = conditional_accuracy      

conditional_accuracy_df = pd.DataFrame(conditional_accuracies)
conditional_accuracy_df.columns = ['True Decreasing (-1)', 'True Stable (0)', 'True Increasing(1)']

#Save results to avoid having to re-run 
os.makedirs("results", exist_ok=True)
conditional_accuracy_df.to_csv("results/periodicity_accuracy_claude.csv", index=False)
mse_by_channel_means.to_csv("results/mse_means_claude.csv", index=False)
mse_by_channel_var.to_csv("results/mse_var_claude.csv", index=False)
mse_by_channel_periodicity.to_csv("results/mse_periodicity_claude.csv", index=False)
mae_by_channel_means.to_csv("results/mae_means_claude.csv", index=False)
mae_by_channel_var.to_csv("results/mae_var_claude.csv", index=False)
mae_by_channel_periodicity.to_csv("results/mae_periodicity_claude.csv", index=False)