<p align="center"> 
  <img src="img/framework.png" alt="TelecomTS overview: curation pipeline, covariates, and supported tasks." width="950"/>
</p>

<p align="center"> 
  🤗 <a href="https://huggingface.co/datasets/AliMaatouk/TelecomTS">TelecomTS Dataset</a> &nbsp;|&nbsp;
  <span style="display:inline-flex; align-items:center; gap:6px; vertical-align:middle;">
    <!-- <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQ2OXMnc5dzJNAKeOLGQDnYURnKrRPELeQOcw&s"
         alt="arXiv" height="25"; style="vertical-align:bottom;"/> -->
  📄 <a href="#">arXiv Paper </a>
  </span>
</p>

# TelecomTS: A Multi-Modal Telecom Dataset

## Overview

**TelecomTS** is a large-scale, high-resolution, multi-modal dataset derived from a **5G telecommunications testbed**. It is the first public observability dataset to preserve **deanonymized** observability metrics with **absolute scale information**, encompassing by design various downstream tasks beyond forecasting such as

- 🔎 **Anomaly detection** (binary)
- 🛠️ **Root-cause analysis** (multi-class)
- ⏱️ **Anomaly duration** (sequence labeling)
- 📈 **Forecasting / reconstruction** (next-step, multi-channel)

Observability data, particularly in telecommunications, differs fundamentally from conventional time series (e.g., weather, finance) by being:
- **Zero-inflated**
- **Highly stochastic and bursty**
- **Structurally noisy with minimal discernible temporal patterns**

These characteristics make TelecomTS a challenging benchmark for both language and time series foundation models, as evidenced by our benchmarking experiments.

## Dataset


**Key features**

* ~**32K** data samples, **1M+** total observations from a 5G testbed
* **Multi-modal inputs**:

  * Time-series KPIs across PHY, MAC, and network layers
  * Environment descriptions + natural-language Q&A pairs
* **Heterogeneous covariates**: numeric KPIs and categorical fields (e.g., `UL_Protocol`, `DL_Protocol`)
- **Absolute scale preserved** (no normalization/anonymization)
* **Downstream tasks supported**:

  * 📈 Forecasting
  * 🔎 Anomaly detection
  * 🛠️ Root-cause analysis
  * 🤖 Multi-modal question answering (time series + text)
* **Labels / metadata**: zone, application, mobility, congestion state, anomaly presence

**Sample Overview**

- **Start_time / end_time** – temporal boundaries of the chunk
- **Sampling_rate** - number of timesteps per second
- **Description** – natural language summary of the network environment and time series behaviors
- **KPIs** – key performance indicator names and values
- **Anomalies** – existence, type, anomaly duration, affected KPIs, and troubleshooting tickets
- **Statistics** – mean, variance, and trends of the KPIs
- **Labels** – contextual metadata (zone, application, mobility, congestion, presence of anomalies)
- **QnA** – natural language reasoning tasks over the sample

**Load with 🤗 `datasets`**

```python
from datasets import load_dataset

# Load the full dataset
ds = load_dataset(
    "AliMaatouk/TelecomTS",
    data_files={"full": "**/chunked.jsonl"}
)
print(ds)
```

## Quickstart

> Requires **Python 3.11**

```bash
# 1) Clone
git clone <repo>
cd TelecomTS_Benchmark

# 2) Create & activate a virtual environment
python3.11 -m venv .venv
# macOS/Linux:
source .venv/bin/activate
# Windows (PowerShell):
# .venv\Scripts\Activate.ps1

# 3) Install
python -m pip install --upgrade pip
pip install -r requirements.txt

# 4) Run (uses configs/config.yaml)
# This trains the selected encoder on the chosen task and then evaluates it
python3 src/run.py
````

## Supported Tasks & Models

Choose the **model** and the **task** in `configs/config.yaml`.
Running `python3 src/run.py` **trains** the selected model and **evaluates** it on the chosen task.

* **Tasks** (`task_type`)

  * `anomaly detection`
  * `root-cause analysis`
  * `anomaly duration`
  * `forecasting`

* **Encoders** (`encoder_type`)

  * `TimesNet`
  * `Autoformer`
  * `NonStationary_Transformer`
  * `FEDformer`
  * `Informer`

## Citation

You can find the paper with all details at https://arxiv.org/abs/2510.06063. Please cite it as follows:

```bib
@misc{feng2025telecomtsmultimodalobservabilitydataset,
      title={TelecomTS: A Multi-Modal Observability Dataset for Time Series and Language Analysis}, 
      author={Austin Feng and Andreas Varvarigos and Ioannis Panitsas and Daniela Fernandez and Jinbiao Wei and Yuwei Guo and Jialin Chen and Ali Maatouk and Leandros Tassiulas and Rex Ying},
      year={2025},
      eprint={2510.06063},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2510.06063}, 
}
```
