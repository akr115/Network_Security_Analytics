#!/usr/bin/env python3
"""
Intrusion Detection System using One-Class SVM
Paper-Compliant Implementation with Label Verification
"""

import os
import pickle
import warnings
import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Tuple, List, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score
)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Configuration
class Config:
    RANDOM_SEED = 42
    BENIGN_TRAIN_RATIO = 0.8  # 80% of benign data for training
    DATA_DIR = "data"
    OUTPUT_DIR = "out"
    
    # Unknown attacks (excluded from training as per paper)
    # Try multiple possible label formats due to encoding issues
    UNKNOWN_ATTACKS = [
        'DoS slowloris', 
        'DoS Slowhttptest', 
        'Bot',
        'DoS slowloris',  # lowercase variant
        'DoS Slowhttptest',  # case variant
        'bot'  # lowercase variant
    ]
    
    # Sample sizes for different modes
    SMALL_SAMPLE_SIZE = 100000
    MEDIUM_SAMPLE_SIZE = 500000
    LARGE_SAMPLE_SIZE = 1000000
    
    SMALL_BENIGN_SAMPLE = 50000
    MEDIUM_BENIGN_SAMPLE = 200000
    LARGE_BENIGN_SAMPLE = 400000
    
    HYPERPARAM_SAMPLE_SIZE = 20000
    HYPERPARAM_BENIGN_SAMPLE = 10000
    
    # OCSVM Parameters (as per paper)
    OCSVM_PARAMS = {
        'kernel': 'rbf',
        'gamma': 'scale',
        'nu': 0.05,
        'shrinking': True,
        'cache_size': 22000,
        'verbose': False,
        'max_iter': -1
    }
    
    OCSVM_PARAMS_SMALL = {
        'kernel': 'rbf',
        'gamma': 'scale',
        'nu': 0.05,
        'shrinking': True,
        'cache_size': 5000,
        'verbose': True,
        'max_iter': 1000
    }
    
    OCSVM_PARAMS_MEDIUM = {
        'kernel': 'rbf',
        'gamma': 'scale',
        'nu': 0.05,
        'shrinking': True,
        'cache_size': 20000,
        'verbose': False,
        'max_iter': -1
    }
    
    OCSVM_PARAMS_LARGE = {
        'kernel': 'rbf',
        'gamma': 'scale',
        'nu': 0.05,
        'shrinking': True,
        'cache_size': 30000,
        'verbose': False,
        'max_iter': -1
    }
    
    # Hyperparameter search space
    HYPERPARAM_GRID = {
        'nu': [0.01, 0.05, 0.1, 0.2],
        'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1.0],
        'kernel': ['rbf', 'poly', 'sigmoid'],
        'cache_size': [1000],
        'shrinking': [True],
        'max_iter': [1000]
    }
    
    HYPERPARAM_GRID_QUICK = {
        'nu': [0.05, 0.1, 0.2],
        'gamma': ['scale', 0.01, 0.1],
        'kernel': ['rbf'],
        'cache_size': [500],
        'shrinking': [True],
        'max_iter': [500]
    }
    
    # File list
    FILE_LIST = [
        "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
        "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
        "Friday-WorkingHours-Morning.pcap_ISCX.csv",
        "Monday-WorkingHours.pcap_ISCX.csv",
        "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
        "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
        "Tuesday-WorkingHours.pcap_ISCX.csv",
        "Wednesday-workingHours.pcap_ISCX.csv"
    ]


class DataLoader:
    """Handle data loading and initial processing"""
    
    def __init__(self, data_dir: str, file_list: List[str], mode: str = "full"):
        self.data_dir = Path(data_dir)
        self.file_list = file_list
        self.mode = mode
    
    def load_data(self) -> pd.DataFrame:
        """Load and combine all CSV files with proper encoding"""
        print("="*60)
        print("LOADING DATA")
        print("="*60)
        
        mode_info = {
            "small": (" SMALL SAMPLE MODE", Config.SMALL_SAMPLE_SIZE, None),
            "medium": (" MEDIUM SAMPLE MODE", Config.MEDIUM_SAMPLE_SIZE, None),
            "large": (" LARGE SAMPLE MODE", Config.LARGE_SAMPLE_SIZE, None),
            "hyperparam": (" HYPERPARAMETER TUNING MODE", Config.HYPERPARAM_SAMPLE_SIZE, None),
            "full": (" FULL TRAINING MODE", None, None)
        }
        
        if self.mode in mode_info:
            print(mode_info[self.mode][0])
            if self.mode == "large":
                print(f"   Large mode: 2x medium size ({Config.LARGE_SAMPLE_SIZE:,} samples)")
        
        data_frames = []
        
        # Load ALL files first (don't limit rows per file)
        for file_name in self.file_list:
            file_path = self.data_dir / file_name
            if file_path.exists():
                print(f"Loading: {file_name}")
                
                # Try different encodings to handle special characters
                for encoding in ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']:
                    try:
                        df = pd.read_csv(file_path, encoding=encoding)
                        print(f"   Loaded {len(df):,} rows (encoding: {encoding})")
                        data_frames.append(df)
                        break
                    except UnicodeDecodeError:
                        continue
                    except Exception as e:
                        print(f"   Error with {encoding}: {e}")
                        continue
            else:
                print(f"Warning: File not found: {file_path}")
        
        if not data_frames:
            raise FileNotFoundError("No data files found!")
        
        combined_data = pd.concat(data_frames, ignore_index=True)
        print(f" Combined all files: {len(combined_data):,} rows")
        
        # Sample AFTER loading all data
        target_size = mode_info.get(self.mode, (None, None))[1]
        if target_size and len(combined_data) > target_size:
            print(f"Sampling {target_size:,} rows from {len(combined_data):,} total rows")
            combined_data = combined_data.sample(n=target_size, random_state=Config.RANDOM_SEED)
        
        print(f" Final combined data shape: {combined_data.shape}")
        print(f" Columns: {len(combined_data.columns)}")
        
        return combined_data


class DataPreprocessor:
    """Handle data preprocessing and cleaning"""
    
    def __init__(self, mode: str = "full"):
        self.mode = mode
        self.columns_to_drop = [
            'Flow ID', 'Source IP', 'Destination IP', 'Timestamp'
        ]
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and preprocess the dataset"""
        print("\n" + "="*60)
        print("DATA CLEANING AND PREPROCESSING")
        print("="*60)
        
        if self.mode != "full":
            print(f" {self.mode.upper()} MODE: Quick preprocessing")
        
        # Clean column names and labels
        df.columns = df.columns.str.strip()
        df['Label'] = df['Label'].str.strip()
        
        # Check for all unique labels
        print("\n All unique labels in dataset:")
        all_labels = df['Label'].unique()
        for label in sorted(all_labels):
            count = len(df[df['Label'] == label])
            print(f"   {repr(label)}: {count:,}")
        
        # Visualize label distribution
        self._visualize_labels(df)
        
        # Handle problematic values BEFORE any splitting
        df = self._handle_problematic_values(df)
        
        # Drop unnecessary columns
        df = self._drop_columns(df)
        
        # Create binary target
        df['Attack'] = (df['Label'] != 'BENIGN').astype(int)
        
        print(f" Final cleaned data shape: {df.shape}")
        return df
    
    def _visualize_labels(self, df: pd.DataFrame) -> None:
        """Visualize the distribution of labels"""
        label_counts = df['Label'].value_counts()
        
        plt.figure(figsize=(14, 7))
        sns.barplot(x=label_counts.index, y=label_counts.values)
        plt.xticks(rotation=45, ha='right')
        
        title = "Distribution of Traffic Types"
        if self.mode != "full":
            title += f" ({self.mode.title()} Mode)"
        plt.title(title)
        
        plt.xlabel("Traffic Label")
        plt.ylabel("Number of Records")
        plt.grid(True)
        plt.tight_layout()
        
        # Save plot
        filename = f'label_distribution_{self.mode}.png'
        plt.savefig(Path(Config.OUTPUT_DIR) / filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\nLabel distribution:")
        print(label_counts)
        
        # Check for unknown attacks
        print(f"\n  Checking for unknown attacks...")
        found_unknown = []
        for label in label_counts.index:
            # Check if label contains any unknown attack pattern
            label_lower = label.lower()
            if 'slowloris' in label_lower or 'slowhttptest' in label_lower or label_lower == 'bot':
                found_unknown.append(label)
                print(f"    Found: {repr(label)} ({label_counts[label]:,} samples)")
        
        if len(found_unknown) < 3:
            print(f"\n     WARNING: Only found {len(found_unknown)} unknown attack types!")
            print(f"   Expected 3: DoS slowloris, DoS Slowhttptest, Bot")
            print(f"   This may affect results comparison with the paper.")
            print(f"\n    TIP: Make sure you're loading ALL CSV files without row limits!")
    
    def _drop_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop unnecessary and constant columns"""
        # Drop specified columns
        for col in self.columns_to_drop:
            if col in df.columns:
                df.drop(col, axis=1, inplace=True)
                print(f"Dropped column: {col}")
        
        # Check for Fwd Header Length.1 duplicate
        if 'Fwd Header Length.1' in df.columns and 'Fwd Header Length' in df.columns:
            if df['Fwd Header Length.1'].equals(df['Fwd Header Length']):
                df.drop('Fwd Header Length.1', axis=1, inplace=True)
                print(f"Dropped duplicate column: Fwd Header Length.1")
        
        # Drop constant columns
        nunique = df.nunique()
        constant_cols = nunique[nunique <= 1].index.tolist()
        constant_cols = [col for col in constant_cols if col not in ['Label', 'Attack']]
        if constant_cols:
            df.drop(constant_cols, axis=1, inplace=True)
            print(f"Dropped {len(constant_cols)} constant columns")
        
        return df
    
    def _handle_problematic_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle infinite, NaN values BEFORE splitting"""
        print("\nHandling problematic values (before splitting)...")
        
        # Separate features from labels
        feature_cols = [col for col in df.columns if col not in ['Label', 'Attack']]
        X = df[feature_cols].copy()
        
        # Handle infinite values first
        inf_count = np.isinf(X.values).sum()
        if inf_count > 0:
            print(f"Found {inf_count:,} infinite values")
            X = X.replace([np.inf, -np.inf], np.nan)
        
        # Handle NaN values
        initial_nan_count = X.isnull().sum().sum()
        if initial_nan_count > 0:
            print(f"Found {initial_nan_count:,} NaN values")
            
            for col in X.columns:
                if X[col].isnull().any():
                    median_val = X[col].median()
                    if not np.isnan(median_val):
                        X[col].fillna(median_val, inplace=True)
                    else:
                        X[col].fillna(0, inplace=True)
        
        # Remove any remaining problematic rows
        clean_mask = ~(X.isnull().any(axis=1) | np.isinf(X.values).any(axis=1))
        if not clean_mask.all():
            print(f"Removing {(~clean_mask).sum():,} problematic rows")
            X = X[clean_mask]
            df = df[clean_mask]
        
        # Update the dataframe with cleaned features
        df[feature_cols] = X
        
        print(" Data cleaning completed (no extreme value capping)")
        return df


class CustomDataSplitter:
    """Handle custom data splitting for OCSVM training"""
    
    def __init__(self, benign_train_ratio: float = 0.8):
        self.benign_train_ratio = benign_train_ratio
    
    def _identify_unknown_attacks(self, df: pd.DataFrame) -> List[str]:
        """Identify unknown attack labels in the dataset"""
        all_labels = df['Label'].unique()
        unknown_labels = []
        
        for label in all_labels:
            label_lower = label.lower()
            # Match patterns for unknown attacks
            if ('slowloris' in label_lower or 
                'slowhttptest' in label_lower or 
                label_lower == 'bot'):
                unknown_labels.append(label)
        
        return unknown_labels
    
    def split_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Custom split as per paper:
        - Training: 80% of benign data only
        - Overall Test: 20% benign + all known attacks + all unknown attacks
        - Unknown Test: Equal benign samples + unknown attacks only
        """
        print("\n" + "="*60)
        print("CUSTOM DATA SPLITTING (AS PER PAPER)")
        print("="*60)
        
        # Identify unknown attacks dynamically
        unknown_attack_labels = self._identify_unknown_attacks(df)
        print(f"Identified unknown attack labels: {unknown_attack_labels}")
        
        if len(unknown_attack_labels) < 3:
            print(f"\n  WARNING: Only {len(unknown_attack_labels)} unknown attack types found!")
            print(f"   Expected 3 types. Results may differ from paper.")
        
        # Separate features and target
        feature_cols = [col for col in df.columns if col not in ['Label', 'Attack']]
        
        print(f"\nTotal samples: {len(df):,}")
        print(f"Features: {len(feature_cols)}")
        
        # Separate data by type
        benign_df = df[df['Label'] == 'BENIGN'].copy()
        unknown_attack_df = df[df['Label'].isin(unknown_attack_labels)].copy()
        known_attack_df = df[
            (df['Attack'] == 1) & 
            (~df['Label'].isin(unknown_attack_labels))
        ].copy()
        
        print(f"\nData composition:")
        print(f"  Benign samples: {len(benign_df):,}")
        print(f"  Known attack samples: {len(known_attack_df):,}")
        print(f"  Unknown attack samples: {len(unknown_attack_df):,}")
        
        # Show unknown attack breakdown
        if len(unknown_attack_df) > 0:
            print(f"\n  Unknown attacks breakdown:")
            for label in unknown_attack_labels:
                count = len(unknown_attack_df[unknown_attack_df['Label'] == label])
                if count > 0:
                    print(f"    {repr(label)}: {count:,}")
        
        # Split benign data: 80% for training, 20% for testing
        benign_train, benign_test = train_test_split(
            benign_df,
            train_size=self.benign_train_ratio,
            random_state=Config.RANDOM_SEED,
            shuffle=True
        )
        
        print(f"\nBenign split:")
        print(f"  Training: {len(benign_train):,} ({self.benign_train_ratio*100:.0f}%)")
        print(f"  Testing:  {len(benign_test):,} ({(1-self.benign_train_ratio)*100:.0f}%)")
        
        # Training set: 80% benign only
        X_train = benign_train[feature_cols]
        y_train = benign_train['Attack']
        
        # Overall Test Set: 20% benign + all known + all unknown
        overall_test_df = pd.concat([benign_test, known_attack_df, unknown_attack_df], ignore_index=True)
        X_test_overall = overall_test_df[feature_cols]
        y_test_overall = overall_test_df['Attack']
        
        # Unknown Attack Test Set: Equal benign + unknown attacks only
        n_unknown = len(unknown_attack_df)
        if n_unknown > 0 and len(benign_test) > 0:
            n_benign_for_unknown = min(n_unknown, len(benign_test))
            benign_for_unknown = benign_test.sample(n=n_benign_for_unknown, random_state=Config.RANDOM_SEED)
            unknown_test_df = pd.concat([benign_for_unknown, unknown_attack_df], ignore_index=True)
            X_test_unknown = unknown_test_df[feature_cols]
            y_test_unknown = unknown_test_df['Attack']
        else:
            X_test_unknown = None
            y_test_unknown = None
        
        print(f"\nFinal splits:")
        print(f"  Training set: {len(X_train):,} samples (100% benign)")
        print(f"  Overall test set: {len(X_test_overall):,} samples")
        
        overall_benign = (y_test_overall == 0).sum()
        overall_attack = (y_test_overall == 1).sum()
        print(f"     {overall_benign:,} benign ({overall_benign/len(y_test_overall)*100:.1f}%)")
        print(f"     {overall_attack:,} attacks ({overall_attack/len(y_test_overall)*100:.1f}%)")
        
        if X_test_unknown is not None:
            print(f"  Unknown attack test set: {len(X_test_unknown):,} samples")
            unknown_benign = (y_test_unknown == 0).sum()
            unknown_attack = (y_test_unknown == 1).sum()
            print(f"     {unknown_benign:,} benign ({unknown_benign/len(y_test_unknown)*100:.1f}%)")
            print(f"     {unknown_attack:,} unknown attacks ({unknown_attack/len(y_test_unknown)*100:.1f}%)")
        else:
            print(f"  Unknown attack test set: None (no unknown attacks in dataset)")
        
        return {
            'X_train': X_train,
            'y_train': y_train,
            'X_test_overall': X_test_overall,
            'y_test_overall': y_test_overall,
            'X_test_unknown': X_test_unknown,
            'y_test_unknown': y_test_unknown,
            'feature_names': feature_cols,
            'unknown_attack_labels': unknown_attack_labels
        }


class HyperparameterTuner:
    """Handle hyperparameter tuning for OCSVM"""
    
    def __init__(self, param_grid: Dict[str, List], quick_mode: bool = False):
        self.param_grid = param_grid
        self.quick_mode = quick_mode
        self.results = []
        self.best_params = None
        self.best_score = -np.inf
    
    def tune(self, X_train: pd.DataFrame, y_train: pd.Series, 
             X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """Perform hyperparameter tuning"""
        print("\n" + "="*60)
        print("HYPERPARAMETER TUNING")
        print("="*60)
        
        max_samples = Config.HYPERPARAM_BENIGN_SAMPLE
        if len(X_train) > max_samples:
            print(f"Sampling {max_samples:,} benign samples for hyperparameter tuning")
            X_train_sample = X_train.sample(n=max_samples, random_state=Config.RANDOM_SEED)
        else:
            X_train_sample = X_train
        
        print(f"Tuning on {len(X_train_sample):,} benign samples")
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_sample)
        X_test_scaled = scaler.transform(X_test)
        
        # Generate parameter combinations
        from itertools import product
        param_names = list(self.param_grid.keys())
        param_values = list(self.param_grid.values())
        param_combinations = list(product(*param_values))
        
        total_combinations = len(param_combinations)
        print(f"Testing {total_combinations} parameter combinations")
        
        if self.quick_mode:
            print(" QUICK MODE: Testing subset of combinations")
            param_combinations = param_combinations[:min(10, total_combinations)]
        
        print(f"Parameter grid:")
        for param, values in self.param_grid.items():
            print(f"  {param}: {values}")
        
        # Test each combination
        for i, param_combo in enumerate(param_combinations):
            params = dict(zip(param_names, param_combo))
            
            print(f"\n[{i+1}/{len(param_combinations)}] Testing: {params}")
            
            try:
                model = OneClassSVM(**params, verbose=False)
                start_time = datetime.now()
                model.fit(X_train_scaled)
                training_time = (datetime.now() - start_time).total_seconds()
                
                y_pred = model.predict(X_test_scaled)
                y_pred_binary = np.where(y_pred == 1, 0, 1)
                
                metrics = {
                    'accuracy': accuracy_score(y_test, y_pred_binary),
                    'precision': precision_score(y_test, y_pred_binary, zero_division=0),
                    'recall': recall_score(y_test, y_pred_binary, zero_division=0),
                    'f1_score': f1_score(y_test, y_pred_binary, zero_division=0)
                }
                
                result = {
                    'params': params.copy(),
                    'metrics': metrics.copy(),
                    'training_time': training_time,
                    'n_support_vectors': model.n_support_[0],
                    'support_vector_ratio': model.n_support_[0] / len(X_train_scaled)
                }
                self.results.append(result)
                
                score = metrics['f1_score']
                if score > self.best_score:
                    self.best_score = score
                    self.best_params = params.copy()
                
                print(f"   F1: {metrics['f1_score']:.4f}, Acc: {metrics['accuracy']:.4f}, "
                      f"Time: {training_time:.2f}s, SVs: {model.n_support_[0]:,}")
                
            except Exception as e:
                print(f"   Failed: {e}")
                continue
        
        self._print_results_summary()
        
        return {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'all_results': self.results
        }
    
    def _print_results_summary(self):
        """Print hyperparameter tuning results summary"""
        print("\n" + "="*60)
        print("HYPERPARAMETER TUNING RESULTS")
        print("="*60)
        
        if not self.results:
            print(" No successful parameter combinations found!")
            return
        
        sorted_results = sorted(self.results, key=lambda x: x['metrics']['f1_score'], reverse=True)
        
        print(f" Tested {len(self.results)} parameter combinations")
        print(f" Best F1 Score: {self.best_score:.4f}")
        print(f" Best Parameters: {self.best_params}")
        
        print("\n Top 5 Results:")
        print("-" * 100)
        print(f"{'Rank':<4} {'F1':<6} {'Acc':<6} {'Prec':<6} {'Rec':<6} {'Time':<6} {'SVs':<8} {'Parameters'}")
        print("-" * 100)
        
        for i, result in enumerate(sorted_results[:5]):
            metrics = result['metrics']
            params_str = ', '.join([f"{k}={v}" for k, v in result['params'].items()])
            if len(params_str) > 40:
                params_str = params_str[:37] + "..."
            
            print(f"{i+1:<4} {metrics['f1_score']:<6.3f} {metrics['accuracy']:<6.3f} "
                  f"{metrics['precision']:<6.3f} {metrics['recall']:<6.3f} "
                  f"{result['training_time']:<6.1f} {result['n_support_vectors']:<8} {params_str}")


class OCSSVMTrainer:
    """One-Class SVM trainer with paper-compliant data splitting"""
    
    def __init__(self, config_params: Dict[str, Any], mode: str = "full"):
        self.params = config_params
        self.mode = mode
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.data_splitter = CustomDataSplitter(Config.BENIGN_TRAIN_RATIO)
        
    def prepare_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Prepare data for training using paper-compliant splitting"""
        print("\n" + "="*60)
        print("DATA PREPARATION")
        print("="*60)
        
        if self.mode != "full":
            print(f" {self.mode.upper()} MODE: Quick data preparation")
        
        split_data = self.data_splitter.split_data(df)
        
        self.feature_names = split_data['feature_names']
        print(f"\n Features: {len(self.feature_names)}")
        print(f" Paper-compliant split completed")
        print(f" Unknown attacks excluded from training: {split_data['unknown_attack_labels']}")
        
        return split_data
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """Train the One-Class SVM model"""
        print("\n" + "="*60)
        print("TRAINING ONE-CLASS SVM")
        print("="*60)
        
        if self.mode != "full":
            print(f" {self.mode.upper()} MODE: Fast training with limited data")
        
        sample_limits = {
            "small": Config.SMALL_BENIGN_SAMPLE,
            "medium": Config.MEDIUM_BENIGN_SAMPLE,
            "large": Config.LARGE_BENIGN_SAMPLE,
            "hyperparam": Config.HYPERPARAM_BENIGN_SAMPLE,
            "full": None
        }
        
        max_samples = sample_limits.get(self.mode)
        if max_samples and len(X_train) > max_samples:
            print(f"Sampling {max_samples:,} benign samples from {len(X_train):,}")
            X_train_sample = X_train.sample(n=max_samples, random_state=Config.RANDOM_SEED)
        else:
            X_train_sample = X_train
        
        print(f"Training on {len(X_train_sample):,} benign samples (100% benign)")
        
        # Feature scaling
        print("Applying StandardScaler (fitted on benign training data only)...")
        self.scaler = StandardScaler()
        
        try:
            X_train_scaled = self.scaler.fit_transform(X_train_sample)
            print(" StandardScaler applied successfully")
        except Exception as e:
            print(f"StandardScaler failed: {e}")
            print("Falling back to RobustScaler...")
            self.scaler = RobustScaler()
            X_train_scaled = self.scaler.fit_transform(X_train_sample)
        
        # Initialize and train model
        print(f"OCSVM parameters (as per paper): {self.params}")
        self.model = OneClassSVM(**self.params)
        
        print("Training model...")
        start_time = datetime.now()
        
        try:
            self.model.fit(X_train_scaled)
            training_time = datetime.now() - start_time
            print(f" Training completed successfully in {training_time.total_seconds():.2f} seconds!")
            print(f"Support vectors: {self.model.n_support_[0]:,}")
            print(f"Support vector ratio: {self.model.n_support_[0] / len(X_train_scaled):.4f}")
        except Exception as e:
            print(f"Training failed: {e}")
            fallback_sizes = {"hyperparam": 1000, "small": 5000, "medium": 20000, "large": 30000, "full": 50000}
            sample_size = min(fallback_sizes.get(self.mode, 50000), len(X_train_scaled))
            indices = np.random.choice(len(X_train_scaled), sample_size, replace=False)
            X_sample = X_train_scaled[indices]
            print(f"Training on reduced sample: {sample_size:,}")
            self.model.fit(X_sample)
            training_time = datetime.now() - start_time
            print(f" Training completed on reduced dataset in {training_time.total_seconds():.2f} seconds!")
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series, test_name: str = "Test") -> Dict[str, float]:
        """Evaluate the trained model"""
        print("\n" + "="*60)
        print(f"MODEL EVALUATION - {test_name}")
        print("="*60)
        
        X_test_scaled = self.scaler.transform(X_test)
        
        print("Making predictions...")
        y_pred_ocsvm = self.model.predict(X_test_scaled)
        y_pred_binary = np.where(y_pred_ocsvm == 1, 0, 1)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred_binary),
            'precision': precision_score(y_test, y_pred_binary, zero_division=0),
            'recall': recall_score(y_test, y_pred_binary, zero_division=0),
            'f1_score': f1_score(y_test, y_pred_binary, zero_division=0)
        }
        
        print(f" Performance Metrics ({test_name}):")
        for metric, value in metrics.items():
            print(f"{metric.capitalize()}: {value:.4f}")
        
        print(f"\n Classification Report ({test_name}):")
        print(classification_report(y_test, y_pred_binary, 
                                  target_names=['Benign', 'Attack'], 
                                  zero_division=0))
        
        cm = confusion_matrix(y_test, y_pred_binary)
        print(f"\n Confusion Matrix ({test_name}):")
        print("           Predicted")
        print("         Benign  Attack")
        print(f"Actual Benign   {cm[0,0]:6d}  {cm[0,1]:6d}")
        print(f"       Attack   {cm[1,0]:6d}  {cm[1,1]:6d}")
        
        print(f"\n Detection Analysis ({test_name}):")
        benign_correct = cm[0,0]
        benign_total = cm[0,0] + cm[0,1]
        attack_detected = cm[1,1]
        attack_total = cm[1,0] + cm[1,1]
        
        if benign_total > 0:
            print(f"Benign detection rate: {benign_correct/benign_total*100:.1f}% ({benign_correct:,}/{benign_total:,})")
        if attack_total > 0:
            print(f"Attack detection rate: {attack_detected/attack_total*100:.1f}% ({attack_detected:,}/{attack_total:,})")
        
        fpr = cm[0,1] / benign_total if benign_total > 0 else 0
        fnr = cm[1,0] / attack_total if attack_total > 0 else 0
        print(f"False Positive Rate: {fpr*100:.2f}%")
        print(f"False Negative Rate: {fnr*100:.2f}%")
        
        return metrics
    
    def evaluate_both_sets(self, split_data: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
        """Evaluate on both Overall and Unknown test sets"""
        results = {}
        
        print("\n" + " EVALUATING ON OVERALL TEST SET ")
        results['overall'] = self.evaluate(
            split_data['X_test_overall'], 
            split_data['y_test_overall'],
            "Overall Test Set"
        )
        
        if split_data['X_test_unknown'] is not None:
            print("\n" + " EVALUATING ON UNKNOWN ATTACK TEST SET ")
            results['unknown'] = self.evaluate(
                split_data['X_test_unknown'],
                split_data['y_test_unknown'],
                "Unknown Attack Test Set"
            )
        else:
            print("\n  No unknown attacks in dataset - skipping unknown attack evaluation")
            results['unknown'] = None
        
        return results


class ModelSaver:
    """Handle model and artifacts saving"""
    
    def __init__(self, output_dir: str, mode: str = "full", hyperparam_results: Dict = None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.mode = mode
        self.hyperparam_results = hyperparam_results
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        suffix = f"_{mode}" if mode != "full" else ""
        self.model_dir = self.output_dir / f"ocsvm_model_{timestamp}{suffix}"
        self.model_dir.mkdir(exist_ok=True)
    
    def save_model(self, trainer: OCSSVMTrainer, metrics: Dict[str, Any]) -> None:
        """Save trained model and related artifacts"""
        print("\n" + "="*60)
        print("SAVING MODEL AND ARTIFACTS")
        print("="*60)
        
        if self.mode != "full":
            print(f" {self.mode.upper()} MODE: Saving model")
        
        try:
            model_path = self.model_dir / "ocsvm_model.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(trainer.model, f)
            print(f" Model saved: {model_path}")
            
            scaler_path = self.model_dir / "feature_scaler.pkl"
            with open(scaler_path, 'wb') as f:
                pickle.dump(trainer.scaler, f)
            print(f" Scaler saved: {scaler_path}")
            
            features_path = self.model_dir / "feature_names.pkl"
            with open(features_path, 'wb') as f:
                pickle.dump(trainer.feature_names, f)
            print(f" Feature names saved: {features_path}")
            
            metrics_path = self.model_dir / "metrics.pkl"
            with open(metrics_path, 'wb') as f:
                pickle.dump(metrics, f)
            print(f" Metrics saved: {metrics_path}")
            
            if self.hyperparam_results:
                hyperparam_path = self.model_dir / "hyperparameter_results.pkl"
                with open(hyperparam_path, 'wb') as f:
                    pickle.dump(self.hyperparam_results, f)
                print(f" Hyperparameter results saved: {hyperparam_path}")
                
                hyperparam_json_path = self.model_dir / "hyperparameter_results.json"
                json_results = {
                    'best_params': self.hyperparam_results['best_params'],
                    'best_score': float(self.hyperparam_results['best_score']),
                    'total_combinations_tested': len(self.hyperparam_results['all_results'])
                }
                with open(hyperparam_json_path, 'w') as f:
                    json.dump(json_results, f, indent=2)
                print(f" Hyperparameter summary saved: {hyperparam_json_path}")
            
            config_path = self.model_dir / "config.pkl"
            config_data = {
                'ocsvm_params': trainer.params,
                'random_seed': Config.RANDOM_SEED,
                'benign_train_ratio': Config.BENIGN_TRAIN_RATIO,
                'unknown_attacks': Config.UNKNOWN_ATTACKS,
                'mode': self.mode,
                'timestamp': datetime.now().isoformat()
            }
            with open(config_path, 'wb') as f:
                pickle.dump(config_data, f)
            print(f" Configuration saved: {config_path}")
            
            self._create_summary(metrics, trainer.params)
            
            print(f"\n All artifacts saved to: {self.model_dir}")
            
        except Exception as e:
            print(f" Error saving model: {e}")
    
    def _create_summary(self, metrics: Dict[str, Any], params: Dict[str, Any]) -> None:
        """Create a human-readable summary file"""
        summary_path = self.model_dir / "model_summary.txt"
        
        with open(summary_path, 'w') as f:
            f.write("OCSVM Intrusion Detection Model Summary\n")
            f.write("="*50 + "\n")
            f.write("(Paper-Compliant Implementation)\n")
            f.write("="*50 + "\n\n")
            f.write(f"Training Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Mode: {self.mode.title()}\n")
            f.write(f"Random Seed: {Config.RANDOM_SEED}\n")
            f.write(f"Benign Train Ratio: {Config.BENIGN_TRAIN_RATIO}\n")
            f.write(f"Unknown Attacks (excluded from training): {Config.UNKNOWN_ATTACKS}\n")
            f.write(f"Data Split: {Config.BENIGN_TRAIN_RATIO*100:.0f}% benign for training, "
                   f"{(1-Config.BENIGN_TRAIN_RATIO)*100:.0f}% benign + all attacks for testing\n\n")
            
            sample_sizes = {
                "small": f"{Config.SMALL_SAMPLE_SIZE:,} total, {Config.SMALL_BENIGN_SAMPLE:,} benign training",
                "medium": f"{Config.MEDIUM_SAMPLE_SIZE:,} total, {Config.MEDIUM_BENIGN_SAMPLE:,} benign training",
                "large": f"{Config.LARGE_SAMPLE_SIZE:,} total, {Config.LARGE_BENIGN_SAMPLE:,} benign training",
                "hyperparam": f"{Config.HYPERPARAM_SAMPLE_SIZE:,} total, {Config.HYPERPARAM_BENIGN_SAMPLE:,} benign training",
                "full": "All available data"
            }
            f.write(f"Sample Size: {sample_sizes.get(self.mode, 'Unknown')}\n\n")
            
            f.write("Model Parameters:\n")
            for param, value in params.items():
                f.write(f"  {param}: {value}\n")
            f.write("\n")
            
            if 'overall' in metrics:
                f.write("Performance Metrics - Overall Test Set:\n")
                for metric, value in metrics['overall'].items():
                    f.write(f"  {metric.capitalize()}: {value:.4f}\n")
                f.write("\n")
            
            if 'unknown' in metrics and metrics['unknown'] is not None:
                f.write("Performance Metrics - Unknown Attack Test Set:\n")
                for metric, value in metrics['unknown'].items():
                    f.write(f"  {metric.capitalize()}: {value:.4f}\n")
                f.write("\n")
            
            if self.hyperparam_results:
                f.write(f"Hyperparameter Tuning Results:\n")
                f.write(f"  Best F1 Score: {self.hyperparam_results['best_score']:.4f}\n")
                f.write(f"  Best Parameters: {self.hyperparam_results['best_params']}\n")
                f.write(f"  Total Combinations Tested: {len(self.hyperparam_results['all_results'])}\n")
                f.write("\n")
            
            f.write("="*50 + "\n")
            f.write("Paper Reference:\n")
            f.write("'Robust Anomaly Detection in Network Traffic:\n")
            f.write("Evaluating Machine Learning Models on CICIDS2017'\n")
            f.write("="*50 + "\n")
            
            if self.mode != "full":
                f.write("\n" + "="*50 + "\n")
                f.write(f"NOTE: This is a {self.mode} mode model.\n")
                if self.mode == "large":
                    f.write("Large mode provides high-quality training with reasonable time.\n")
                elif self.mode == "medium":
                    f.write("Medium mode provides balanced performance and training time.\n")
                elif self.mode != "hyperparam":
                    f.write("For production use, consider running with full dataset.\n")
        
        print(f" Summary saved: {summary_path}")


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Train OCSVM Intrusion Detection System (Paper-Compliant Implementation)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python trainer.py                           # Full training (ALL data)
  python trainer.py --small-sample            # Quick test (100K samples)
  python trainer.py --medium-sample           # Medium test (500K samples)
  python trainer.py --large-sample            # Large test (1M samples)
  python trainer.py --hyperparam-tune         # Hyperparameter tuning
  python trainer.py --hyperparam-tune --quick # Quick hyperparameter tuning

IMPORTANT: For accurate paper replication, use --full mode (no flags) to load ALL data!
        """
    )
    
    parser.add_argument(
        '--small-sample', '-s',
        action='store_true',
        help='Use small sample for quick testing (100K samples)'
    )
    
    parser.add_argument(
        '--medium-sample', '-m',
        action='store_true',
        help='Use medium sample (500K samples)'
    )
    
    parser.add_argument(
        '--large-sample', '-l',
        action='store_true',
        help='Use large sample (1M samples)'
    )
    
    parser.add_argument(
        '--hyperparam-tune', '--tune',
        action='store_true',
        help='Perform hyperparameter tuning'
    )
    
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Quick mode for hyperparameter tuning'
    )
    
    parser.add_argument(
        '--data-dir',
        default=Config.DATA_DIR,
        help=f'Directory containing CSV files (default: {Config.DATA_DIR})'
    )
    
    parser.add_argument(
        '--output-dir',
        default=Config.OUTPUT_DIR,
        help=f'Output directory (default: {Config.OUTPUT_DIR})'
    )
    
    return parser.parse_args()


def main():
    """Main training pipeline"""
    args = parse_arguments()
    
    Config.DATA_DIR = args.data_dir
    Config.OUTPUT_DIR = args.output_dir
    
    if args.hyperparam_tune:
        mode = "hyperparam"
        mode_desc = "HYPERPARAMETER TUNING"
    elif args.large_sample:
        mode = "large"
        mode_desc = "LARGE SAMPLE TEST"
    elif args.medium_sample:
        mode = "medium"
        mode_desc = "MEDIUM SAMPLE TEST"
    elif args.small_sample:
        mode = "small"
        mode_desc = "SMALL SAMPLE TEST"
    else:
        mode = "full"
        mode_desc = "FULL TRAINING (ALL DATA)"
    
    print(f" Starting OCSVM Intrusion Detection Training Pipeline")
    print(f" Mode: {mode_desc}")
    print(f" Paper-Compliant Implementation")
    
    if mode != "full":
        print(f"\n  WARNING: Using {mode} mode - results may differ from paper!")
        print(f"   For accurate replication, run without flags to use ALL data.")
    
    print(f" Data directory: {Config.DATA_DIR}")
    print(f" Output directory: {Config.OUTPUT_DIR}")
    print(f" Data split: {Config.BENIGN_TRAIN_RATIO*100:.0f}% benign for training")
    print(f"  Unknown attacks will be identified and excluded from training")
    
    if args.quick and args.hyperparam_tune:
        print(" Quick hyperparameter tuning enabled")
    
    Path(Config.OUTPUT_DIR).mkdir(exist_ok=True)
    
    try:
        # Load data
        loader = DataLoader(Config.DATA_DIR, Config.FILE_LIST, mode)
        raw_data = loader.load_data()
        
        # Preprocess data
        preprocessor = DataPreprocessor(mode)
        clean_data = preprocessor.clean_data(raw_data)
        
        # Prepare data
        param_configs = {
            "hyperparam": Config.OCSVM_PARAMS_SMALL,
            "small": Config.OCSVM_PARAMS_SMALL,
            "medium": Config.OCSVM_PARAMS_MEDIUM,
            "large": Config.OCSVM_PARAMS_LARGE,
            "full": Config.OCSVM_PARAMS
        }
        
        ocsvm_params = param_configs[mode]
        trainer = OCSSVMTrainer(ocsvm_params, mode)
        split_data = trainer.prepare_data(clean_data)
        
        hyperparam_results = None
        
        # Hyperparameter tuning
        if args.hyperparam_tune:
            param_grid = Config.HYPERPARAM_GRID_QUICK if args.quick else Config.HYPERPARAM_GRID
            tuner = HyperparameterTuner(param_grid, args.quick)
            hyperparam_results = tuner.tune(
                split_data['X_train'], 
                split_data['y_train'],
                split_data['X_test_overall'],
                split_data['y_test_overall']
            )
            
            if hyperparam_results['best_params']:
                print(f"\n Using best parameters: {hyperparam_results['best_params']}")
                trainer.params = hyperparam_results['best_params']
            else:
                print("\n  No successful combinations, using default parameters")
        
        # Train model
        trainer.train(split_data['X_train'], split_data['y_train'])
        
        # Evaluate
        all_metrics = trainer.evaluate_both_sets(split_data)
        
        # Save
        saver = ModelSaver(Config.OUTPUT_DIR, mode, hyperparam_results)
        saver.save_model(trainer, all_metrics)
        
        print(f"\n Training pipeline completed successfully!")
        
        if 'overall' in all_metrics:
            print(f"\n Overall Test Set Results:")
            print(f"   Accuracy:  {all_metrics['overall']['accuracy']:.4f}")
            print(f"   Precision: {all_metrics['overall']['precision']:.4f}")
            print(f"   Recall:    {all_metrics['overall']['recall']:.4f}")
            print(f"   F1 Score:  {all_metrics['overall']['f1_score']:.4f}")
        
        if 'unknown' in all_metrics and all_metrics['unknown'] is not None:
            print(f"\n Unknown Attack Test Set Results:")
            print(f"   Accuracy:  {all_metrics['unknown']['accuracy']:.4f}")
            print(f"   Precision: {all_metrics['unknown']['precision']:.4f}")
            print(f"   Recall:    {all_metrics['unknown']['recall']:.4f}")
            print(f"   F1 Score:  {all_metrics['unknown']['f1_score']:.4f}")
        
        print(f"\n Model saved in: {saver.model_dir}")
        
        if mode != "full":
            print(f"\n  NOTE: This was a {mode} mode run.")
            print(f"   For best results matching the paper, run with ALL data (no flags).")
        
        print(f"\n Implementation follows paper methodology:")
        print(f"   'Robust Anomaly Detection in Network Traffic'")
        
    except Exception as e:
        print(f" Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    np.random.seed(Config.RANDOM_SEED)
    main()
