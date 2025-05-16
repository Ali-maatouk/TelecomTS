---
license: mit
---
# 📡 TelecomTS: A Telecom Time-Series Dataset

---

## 📖 Overview

**TelecomTS** is a large-scale, high-resolution, multi-modal time-series dataset collected from a real 5G communication network testbed. It is designed to bridge the gap between current academic benchmarks and the complexities of real-world engineering scenarios.

Time-series data powers critical applications in anomaly detection, fault diagnosis, system monitoring, and control. While recent foundation models have shown promise on benchmarks like the UCR Anomaly Archive, they struggle when exposed to the dynamic, bursty, and noisy nature of real-world data. **TelecomTS** addresses these shortcomings and provide a unified benchmark for testing and evaluation.

---

## 🚀 Key Features

- **1M+ Observations** from a live 5G network environment  
- **Multi-modal**: physical (PHY), medium access control (MAC), and network-layer KPIs  
- **Normal vs. Anomalous** data scenarios, enabling robust anomaly detection benchmarks  
- **Rich covariates**: categorical and continuous features reflecting true telecom system diversity  
- **Supports downstream tasks**: forecasting, classification, anomaly detection, root-cause analysis, and QnA 

---

## 📂 Dataset Structure

### 🗃️ Categories

- `normal/`: typical network operation samples  
- `anomalous/`: irregular behaviors and injected or observed anomalies  

### 📁 Contents

Each category includes:
- Raw metrics: `metrics.csv`  
- Human-readable event descriptions: `description.txt`  
- Processed time-series chunks (for easy ingestion): `.jsonl`  
  - Includes variants focused on anomalies

## 🧪 Installation & Usage

### 📦 Installation

To get started, install the 🤗 `datasets` library:

```bash
pip install datasets
```

---

### 📥 Load the Dataset

```python
from datasets import load_dataset

# Load the TelecomTS dataset from Hugging Face in streaming mode
dataset = load_dataset("AliMaatouk/TelecomTS", streaming=True)
```

---

### 📁 Explore Subsets

```python
# Print an example

for sample in dataset['train']:
    print(sample)
    break
```