# OCSVM Intrusion Detection Model Training

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.2-orange.svg)](https://scikit-learn.org/)
[![DVC](https://img.shields.io/badge/DVC-enabled-945DD6.svg)](https://dvc.org/)

A complete training pipeline for **One-Class Support Vector Machine (OCSVM)** based network intrusion detection system using the **CICIDS2017** dataset. This implementation follows the paper-compliant methodology for robust anomaly detection in network traffic.

---

##  Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Quick Start](#quick-start)
---

##  Overview

This training pipeline implements a **paper-compliant One-Class SVM** for network intrusion detection, specifically designed to:

-  **Anomaly Detection**: Train on benign traffic only, detect attacks as anomalies
-  **Unknown Attack Detection**: Evaluate performance on previously unseen attack types
-  **Comprehensive Evaluation**: Dual test sets (overall and unknown attacks)
-  **Reproducible Results**: Fixed random seeds and configurable modes
-  **Visualization**: Automatic generation of performance plots and metrics
-  **DVC Integration**: Dataset versioning and reproducibility

### Key Features

- **Paper-Compliant Methodology**: Follows research paper implementation details
- **Custom Data Splitting**: 80% benign for training, 20% benign + all attacks for testing
- **Unknown Attack Evaluation**: Special test set for DoS Slowloris, DoS Slowhttptest, and Bot attacks
- **Automatic Preprocessing**: Handles missing values, infinite values, and feature scaling
- **Multiple Training Modes**: From quick testing to full dataset training
- **Hyperparameter Tuning**: Grid search with cross-validation
- **Comprehensive Outputs**: Models, scalers, metrics, plots, and detailed summaries

---

##  Dataset

### CICIDS2017 Dataset

The [CICIDS2017 Dataset - Canadian Institute for Cybersecurity](https://www.unb.ca/cic/datasets/ids-2017.html) dataset contains labeled network traffic flows with both benign and various attack types.

#### Dataset Composition

The dataset consists of **8 CSV files** covering different days and attack scenarios:

| File Name | Description | Attack Types |
|-----------|-------------|--------------|
| `Monday-WorkingHours.pcap_ISCX.csv` | Monday traffic | BENIGN only |
| `Tuesday-WorkingHours.pcap_ISCX.csv` | Tuesday traffic | BENIGN, FTP-Patator, SSH-Patator |
| `Wednesday-workingHours.pcap_ISCX.csv` | Wednesday traffic | BENIGN, DoS attacks, Heartbleed |
| `Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv` | Thursday morning | BENIGN, Web attacks |
| `Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv` | Thursday afternoon | BENIGN, Infiltration |
| `Friday-WorkingHours-Morning.pcap_ISCX.csv` | Friday morning | BENIGN, Bot |
| `Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv` | Friday afternoon | BENIGN, PortScan |
| `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv` | Friday DDoS | BENIGN, DDoS attacks |

#### Attack Types

**Known Attacks** (used in training evaluation):
- FTP-Patator, SSH-Patator
- DoS GoldenEye, DoS Hulk
- PortScan, DDoS
- Web Attack  Brute Force, Web Attack  XSS, Web Attack  SQL Injection
- Infiltration, Heartbleed

**Unknown Attacks** (excluded from training, used for generalization testing):
- DoS Slowloris
- DoS Slowhttptest
- Bot

#### Features

Each record contains **78+ network flow features**, including:
- **Basic Flow Features**: Duration, packet counts, byte counts
- **Packet Statistics**: Min, max, mean, std of packet lengths
- **Inter-Arrival Time**: Forward/backward IAT statistics
- **Flag Counts**: SYN, FIN, RST, PSH, ACK, URG, ECE, CWE
- **Bulk Transfer**: Bytes/packets per bulk, bulk rate
- **Subflow Features**: Forward/backward subflow packets/bytes
- **Window Size**: Initial window bytes forward/backward
- **Active/Idle Times**: Active/idle mean, std, max, min

---

##  Prerequisites

### System Requirements

- **Python**: 3.11 or higher

---

##  Installation

### Option 1: Using DVC (Recommended)

If you have access to the DVC remote storage:

```bash
cd Train

# Pull the dataset using DVC
dvc pull data/*.csv.dvc

# Install Python dependencies
pip install -r ../backend_model/requirements.txt
pip install matplotlib seaborn
```



If you don't have DVC access, download the CICIDS2017 dataset manually:

1. **Download the Dataset**
   
   Visit the official source:
   - [CICIDS2017 Dataset - Canadian Institute for Cybersecurity](https://www.unb.ca/cic/datasets/ids-2017.html)
   - Or alternative sources: [Kaggle CICIDS2017](https://www.kaggle.com/datasets/cicdataset/cicids2017)

2. **Extract to Data Directory**

   ```bash
   cd Train/data
   
   # Extract all CSV files here
   # The directory should contain:
   # - Monday-WorkingHours.pcap_ISCX.csv
   # - Tuesday-WorkingHours.pcap_ISCX.csv
   # - Wednesday-workingHours.pcap_ISCX.csv
   # - Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
   # - Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
   # - Friday-WorkingHours-Morning.pcap_ISCX.csv
   # - Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
   # - Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
   ```

3. **Verify Data Structure**

   ```bash
   # From Train directory
   ls -lh data/*.csv
   
   # You should see 8 CSV files
   ```

4. **Install Dependencies**

   ```bash
   pip install scikit-learn numpy pandas matplotlib seaborn
   ```

---

##  Quick Start

### Basic Training (Full Dataset)

For **paper-compliant results** using the complete dataset:

```bash
cd Train
python trainer_paper_replicate.py
```

This will:
- Load all 8 CSV files (~2.8M records)
- Preprocess and clean the data
- Split data (80% benign for training, rest for testing)
- Train OCSVM on benign traffic only
- Evaluate on both overall and unknown attack test sets
- Save model, metrics, and visualizations to `out/` directory

**Expected Runtime**: 26hours

### Quick Test (Small Sample)

For rapid testing and development:

```bash
python trainer_paper_replicate.py --small-sample
```

**Expected Runtime**: 2-5 minutes

---