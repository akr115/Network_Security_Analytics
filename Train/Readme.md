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
- [Training Modes](#training-modes)
- [Data Pipeline](#data-pipeline)
- [Model Architecture](#model-architecture)
- [Output Structure](#output-structure)
- [Advanced Usage](#advanced-usage)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Understanding the Results](#understanding-the-results)
- [Troubleshooting](#troubleshooting)
- [Paper Reference](#paper-reference)

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

The **Canadian Institute for Cybersecurity Intrusion Detection System (CICIDS2017)** dataset contains labeled network traffic flows with both benign and various attack types.

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
- **RAM**: Minimum 16GB (32GB recommended for full dataset)
- **Storage**: ~10GB for dataset + ~5GB for outputs
- **CPU**: Multi-core processor recommended

### Python Dependencies

```bash
# Core ML libraries
scikit-learn>=1.3.2
numpy>=1.26.2
pandas>=2.1.3

# Visualization
matplotlib>=3.8.0
seaborn>=0.13.0

# Data versioning (optional)
dvc>=3.0.0
```

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

### Option 2: Manual Dataset Download

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

##  Training Modes

The training script supports multiple modes for different use cases:

### 1. Full Training Mode (Default)

```bash
python trainer_paper_replicate.py
```

- Uses **ALL available data** (~2.8M records)
- Best accuracy and paper-compliant results
- Recommended for production models
- Runtime: 30-90 minutes

### 2. Small Sample Mode

```bash
python trainer_paper_replicate.py --small-sample
# or
python trainer_paper_replicate.py -s
```

- Uses **100,000 samples** total
- Quick testing and debugging

### 3. Medium Sample Mode

```bash
python trainer_paper_replicate.py --medium-sample
# or
python trainer_paper_replicate.py -m
```

- Uses **500,000 samples**
- Balance between speed and accuracy

### 4. Large Sample Mode

```bash
python trainer_paper_replicate.py --large-sample
# or
python trainer_paper_replicate.py -l
```

- Uses **1,000,000 samples**
- Near-full accuracy with faster training

### 5. Hyperparameter Tuning Mode

```bash
python trainer_paper_replicate.py --hyperparam-tune
# or
python trainer_paper_replicate.py --tune
```

- Performs grid search over hyperparameter space
- Tests multiple combinations of `nu`, `gamma`, `kernel`
- Runtime: 1-3 hours

**Quick Hyperparameter Tuning:**

```bash
python trainer_paper_replicate.py --hyperparam-tune --quick
```

- Reduced parameter grid
- Runtime: 15-30 minutes

---

##  Data Pipeline

### Pipeline Architecture

```

                    1. DATA LOADING                          
   Load 8 CSV files from data/ directory                   
   Handle multiple encodings (UTF-8, Latin-1, etc.)        
   Combine into single DataFrame                            
   Optional sampling based on mode                          

                              

                 2. DATA PREPROCESSING                        
   Clean column names (strip whitespace)                   
   Clean label values                                       
   Handle infinite values (replace with NaN)               
   Handle NaN values (fill with median/0)                  
   Drop constant columns                                    
   Drop unnecessary columns (IP, Timestamp, etc.)          
   Create binary target (BENIGN=0, ATTACK=1)               

                              

                   3. DATA SPLITTING                          
  A. Identify unknown attacks:                              
     - DoS Slowloris, DoS Slowhttptest, Bot                 
  B. Split benign data: 80% train, 20% test                 
  C. Create test sets:                                       
     - Overall: 20% benign + all attacks                    
     - Unknown: Equal benign + unknown attacks only         

                              

                   4. FEATURE SCALING                         
   Fit StandardScaler on training data only                
   Transform all datasets with same scaler                 
   Ensures zero mean, unit variance                        

                              

                    5. MODEL TRAINING                         
   Train One-Class SVM on benign data only                 
   RBF kernel with optimal hyperparameters                 
   nu=0.05 (contamination factor)                          
   gamma='scale' (auto-calculated)                         

                              

                     6. EVALUATION                            
  A. Overall Test Set:                                       
     - Predict on mixed test set                            
     - Calculate metrics (Accuracy, Precision, Recall, F1)  
     - Generate confusion matrix                             
  B. Unknown Attack Test Set:                               
     - Test generalization to unseen attacks                
     - Calculate same metrics                                

                              

                  7. SAVE OUTPUTS                             
   Model artifacts (model, scaler, features)               
   Performance metrics (JSON, plots)                        
   Confusion matrices (PNG)                                 
   Model summary (TXT)                                      
   All saved to timestamped directory in out/              

```

### Data Splitting Strategy

**Training Set** (BENIGN ONLY):
- 80% of benign traffic
- Used to learn "normal" behavior pattern
- ~1.8M samples in full mode

**Overall Test Set** (MIXED):
- 20% of benign traffic
- ALL known attack samples
- ALL unknown attack samples
- ~1M samples in full mode

**Unknown Attack Test Set** (BALANCED):
- Equal number of benign samples (matched to unknown attacks)
- ALL unknown attack samples (Slowloris, Slowhttptest, Bot)
- Tests model generalization
- ~50K-100K samples typically

---

##  Model Architecture

### One-Class SVM Configuration

**Algorithm**: One-Class Support Vector Machine (OCSVM)
**Purpose**: Anomaly detection (semi-supervised learning)

**Hyperparameters** (Default):
```python
{
    'kernel': 'rbf',           # Radial Basis Function kernel
    'gamma': 'scale',          # 1 / (n_features * X.var())
    'nu': 0.05,                # Upper bound on fraction of outliers
    'shrinking': True,         # Use shrinking heuristic
    'cache_size': 22000,       # MB of cache for kernel
    'max_iter': -1,            # No limit on iterations
    'verbose': False           # Silent training
}
```

**Why OCSVM?**
-  Learns boundary around normal traffic
-  Detects anomalies (attacks) without seeing them during training
-  Effective for imbalanced datasets
-  Handles high-dimensional feature spaces
-  Can detect unknown/zero-day attacks

### Feature Engineering

**Input**: 78 network flow features (after preprocessing)

**Dropped Features**:
- `Flow ID` - Unique identifier, not predictive
- `Source IP`, `Destination IP` - PII and context-specific
- `Timestamp` - Temporal information, not used
- `Fwd Header Length.1` - Duplicate column
- Constant columns - No variance

**Scaling**: StandardScaler
- Mean = 0, Std = 1 for all features
- Fitted on training set only
- Applied to all datasets

---

##  Output Structure

After training, a timestamped directory is created in `out/`:

```
out/
 ocsvm_model_YYYYMMDD_HHMMSS/
     ocsvm_model.pkl              # Trained OCSVM model
     feature_scaler.pkl           # StandardScaler object
     feature_names.pkl            # List of feature names
     config.pkl                   # Model configuration
     model_summary.txt            # Human-readable summary
     metrics.json                 # Performance metrics (JSON)
     confusion_matrix_overall.png # Overall test set confusion matrix
     confusion_matrix_unknown.png # Unknown attack confusion matrix
     label_distribution_*.png     # Label distribution plot
     hyperparam_results.json      # Hyperparameter tuning results (if applicable)
```

### Output Files Explained

#### 1. Model Artifacts

**`ocsvm_model.pkl`**
- Serialized trained OCSVM model
- Can be loaded with `pickle.load()`
- Used by backend API for predictions

**`feature_scaler.pkl`**
- Fitted StandardScaler object
- Must be used to scale new data before prediction
- Ensures consistency with training data

**`feature_names.pkl`**
- Ordered list of feature names
- Ensures correct feature ordering for predictions
- Used by API to validate input data

**`config.pkl`**
- Model hyperparameters and configuration
- Training metadata
- Performance metrics

#### 2. Documentation

**`model_summary.txt`** - Example:
```
OCSVM Intrusion Detection Model Summary
==================================================
(Paper-Compliant Implementation)
==================================================

Training Date: 2025-10-31 12:34:56
Mode: Full
Random Seed: 42
Benign Train Ratio: 0.8
Unknown Attacks: ['DoS slowloris', 'DoS Slowhttptest', 'Bot']
Data Split: 80% benign for training, 20% benign + all attacks for testing

Model Parameters:
  kernel: rbf
  gamma: scale
  nu: 0.05
  shrinking: True
  cache_size: 22000

Performance Metrics - Overall Test Set:
  Accuracy: 0.6782
  Precision: 0.9175
  Recall: 0.4569
  F1_score: 0.6101

Performance Metrics - Unknown Attack Test Set:
  Accuracy: 0.7930
  Precision: 0.9270
  Recall: 0.6360
  F1_score: 0.7544
```

**`metrics.json`** - Machine-readable metrics:
```json
{
  "overall": {
    "accuracy": 0.6782,
    "precision": 0.9175,
    "recall": 0.4569,
    "f1_score": 0.6101,
    "support": 1000000
  },
  "unknown": {
    "accuracy": 0.7930,
    "precision": 0.9270,
    "recall": 0.6360,
    "f1_score": 0.7544,
    "support": 75000
  }
}
```

#### 3. Visualizations

**`confusion_matrix_overall.png`**
- Confusion matrix for overall test set
- Shows TP, TN, FP, FN counts
- Heatmap visualization

**`confusion_matrix_unknown.png`**
- Confusion matrix for unknown attack test set
- Evaluates generalization capability

**`label_distribution_*.png`**
- Bar chart of traffic type distribution
- Helps understand dataset composition

---

##  Advanced Usage

### Custom Data Directory

```bash
python trainer_paper_replicate.py --data-dir /path/to/custom/data
```

### Custom Output Directory

```bash
python trainer_paper_replicate.py --output-dir /path/to/output
```

### Combined Options

```bash
# Medium sample with hyperparameter tuning
python trainer_paper_replicate.py --medium-sample --hyperparam-tune

# Large sample with quick hyperparameter tuning
python trainer_paper_replicate.py --large-sample --hyperparam-tune --quick

# Custom directories
python trainer_paper_replicate.py \
  --data-dir ../data/custom \
  --output-dir ../models/custom
```

### Command Line Reference

```
Usage: trainer_paper_replicate.py [OPTIONS]

Options:
  -s, --small-sample      Use small sample (100K samples)
  -m, --medium-sample     Use medium sample (500K samples)
  -l, --large-sample      Use large sample (1M samples)
  --hyperparam-tune       Perform hyperparameter tuning
  --tune                  (alias for --hyperparam-tune)
  --quick                 Quick mode for hyperparameter tuning
  --data-dir DIR          Data directory (default: data)
  --output-dir DIR        Output directory (default: out)
  -h, --help              Show help message

Examples:
  python trainer_paper_replicate.py
  python trainer_paper_replicate.py --small-sample
  python trainer_paper_replicate.py --hyperparam-tune --quick
```

---

##  Hyperparameter Tuning

### Grid Search Configuration

**Full Hyperparameter Grid**:
```python
{
    'nu': [0.01, 0.05, 0.1, 0.2],              # 4 values
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1.0],  # 6 values
    'kernel': ['rbf', 'poly', 'sigmoid'],       # 3 values
    'cache_size': [1000],                       # Fixed
    'shrinking': [True],                        # Fixed
    'max_iter': [1000]                          # Fixed for speed
}
# Total: 4  6  3 = 72 combinations
```

**Quick Hyperparameter Grid** (--quick):
```python
{
    'nu': [0.05, 0.1, 0.2],                    # 3 values
    'gamma': ['scale', 0.01, 0.1],             # 3 values
    'kernel': ['rbf'],                         # 1 value (best)
    'cache_size': [500],                       # Reduced
    'shrinking': [True],                       # Fixed
    'max_iter': [500]                          # Reduced
}
# Total: 3  3  1 = 9 combinations
```

### Running Hyperparameter Tuning

```bash
# Full grid search (72 combinations, ~2-3 hours)
python trainer_paper_replicate.py --hyperparam-tune

# Quick grid search (9 combinations, ~20-30 minutes)
python trainer_paper_replicate.py --hyperparam-tune --quick

# With specific mode
python trainer_paper_replicate.py --medium-sample --hyperparam-tune --quick
```

### Understanding Results

The tuning process outputs:
1. **Progress**: Real-time updates for each combination
2. **Top Results**: Best 5 parameter combinations ranked by F1 score
3. **Detailed Metrics**: Accuracy, Precision, Recall, F1 for each
4. **Training Time**: How long each combination took
5. **Support Vectors**: Number and ratio of support vectors

**Output** (`hyperparam_results.json`):
```json
{
  "best_params": {
    "nu": 0.05,
    "gamma": "scale",
    "kernel": "rbf"
  },
  "best_score": 0.6101,
  "all_results": [
    {
      "params": {...},
      "metrics": {...},
      "training_time": 123.45,
      "n_support_vectors": 45000,
      "support_vector_ratio": 0.025
    }
  ]
}
```

---

##  Understanding the Results

### Performance Metrics

**Accuracy**: Overall correctness
```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

**Precision**: Proportion of predicted attacks that are actual attacks
```
Precision = TP / (TP + FP)
```
- High precision = Few false alarms

**Recall**: Proportion of actual attacks that are detected
```
Recall = TP / (TP + FN)
```
- High recall = Few missed attacks

**F1 Score**: Harmonic mean of precision and recall
```
F1 = 2  (Precision  Recall) / (Precision + Recall)
```
- Balances precision and recall

### Expected Performance

**Overall Test Set** (paper-compliant, full mode):
- Accuracy: ~67-68%
- Precision: ~91-92%
- Recall: ~45-46%
- F1 Score: ~60-61%

**Unknown Attack Test Set**:
- Accuracy: ~79-80%
- Precision: ~92-93%
- Recall: ~63-64%
- F1 Score: ~75-76%

### Interpreting Confusion Matrix

```
                 Predicted
                BENIGN  ATTACK
Actual BENIGN     TN      FP       False alarms
       ATTACK     FN      TP       Missed attacks
```

- **True Negatives (TN)**: Correctly identified benign traffic
- **True Positives (TP)**: Correctly identified attacks
- **False Positives (FP)**: Benign traffic misclassified as attack
- **False Negatives (FN)**: Attacks misclassified as benign ( dangerous)

### Why High Precision, Lower Recall?

OCSVM is **conservative** by design:
-  Very good at confirming attacks (high precision)
-  May miss some attacks to avoid false alarms (lower recall)
- This is typical for anomaly detection methods
- Trade-off can be adjusted via `nu` parameter

##  Paper Reference

This implementation follows the methodology from:

**"Robust Anomaly Detection in Network Traffic: Evaluating Machine Learning Models on CICIDS2017"**

### Key Methodological Points

1. **One-Class Learning**: Train only on benign traffic
2. **Unknown Attack Evaluation**: Test on unseen attack types
3. **Balanced Test Set**: Equal benign/attack samples for unknown test
4. **Paper-Compliant Split**: 80/20 benign train/test ratio
5. **RBF Kernel**: Proven most effective for network traffic
6. **nu = 0.05**: Conservative contamination factor

### Differences from Standard Classification

| Aspect | Standard Classification | This OCSVM Approach |
|--------|------------------------|---------------------|
| Training Data | Benign + Attack samples | Benign only |
| Learning Type | Supervised | Semi-supervised |
| Attack Detection | Pattern matching | Anomaly detection |
| Unknown Attacks | Cannot detect | Can generalize |
| Label Required | Yes, for all data | Only for evaluation |

---

##  Integration with Backend API

The trained model can be used with the backend API:

```bash
# After training
cd ../backend_model

# Copy latest model
cp -r ../Train/out/ocsvm_model_YYYYMMDD_HHMMSS/* production_model/

# Run API
python main.py

# Or use Docker
docker-compose up
```

See `../backend_model/README.md` for API documentation.

---

##  Data Exploration

Use the provided Jupyter notebook for exploratory analysis:

```bash
jupyter notebook data_exloration.ipynb
```

The notebook includes:
- Dataset loading and inspection
- Label distribution visualization
- Feature statistics
- Correlation analysis
- Data quality checks

---

##  Data Versioning with DVC

This project uses **DVC (Data Version Control)** for dataset management.

### DVC Files

- `data/*.csv.dvc` - Metadata files for each CSV
- `out.dvc` - Tracks output directory
- `.dvc/` - DVC configuration (gitignored)

### DVC Commands

```bash
# Pull data from remote storage
dvc pull

# Check status
dvc status

# Add new data
dvc add data/new_file.csv
git add data/new_file.csv.dvc

# Push to remote
dvc push
```

### Without DVC Access

Simply download the CICIDS2017 dataset manually and extract to `data/` directory. The training script will work the same way.

---

##  Notes

### Important Considerations

1. **Random Seed**: Fixed at 42 for reproducibility
2. **Unknown Attacks**: Must be excluded from training for fair evaluation
3. **Feature Scaling**: Always use the saved scaler for new predictions
4. **Memory Usage**: Full dataset requires ~16GB RAM minimum
5. **Training Time**: Varies significantly based on CPU and dataset size


### Future Improvements

Possible enhancements:
- Multi-class classification (identify attack types)
- Ensemble methods (combine multiple models)
- Deep learning approaches (LSTM, CNN)
- Real-time incremental learning
- Feature selection optimization
- Cross-validation for robust evaluation
