#!/usr/bin/env python3
"""
Intrusion Detection System using One-Class SVM
Enhanced with custom data splitting and multiple dataset size options
"""

import os
import pickle
import warnings
import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Tuple, List, Dict, Any
from itertools import product

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
    
    # Sample sizes for different modes
    SMALL_SAMPLE_SIZE = 100000      # Total samples for quick test
    MEDIUM_SAMPLE_SIZE = 500000     # Total samples for medium test
    LARGE_SAMPLE_SIZE = 1000000     # Total samples for large test (2x medium)
    
    SMALL_BENIGN_SAMPLE = 50000     # Benign samples for OCSVM training (small)
    MEDIUM_BENIGN_SAMPLE = 200000   # Benign samples for OCSVM training (medium)
    LARGE_BENIGN_SAMPLE = 400000    # Benign samples for OCSVM training (large, 2x medium)
    
    # Hyperparameter tuning sample sizes (even smaller for grid search)
    HYPERPARAM_SAMPLE_SIZE = 20000  # Total samples for hyperparameter tuning
    HYPERPARAM_BENIGN_SAMPLE = 10000  # Benign samples for hyperparameter tuning
    
    # OCSVM Parameters
    OCSVM_PARAMS = {
        'kernel': 'rbf',
        'gamma': 'scale',
        'nu': 0.05,
        'shrinking': True,
        'cache_size': 22000,
        'verbose': False,
        'max_iter': 1000
    }
    
    # OCSVM Parameters for small sample (faster training)
    OCSVM_PARAMS_SMALL = {
        'kernel': 'rbf',
        'gamma': 'scale',
        'nu': 0.1,  # Higher nu for faster training
        'shrinking': True,
        'cache_size': 5000,
        'verbose': True,  # Show progress for small sample
        'max_iter': -1
    }
    
    # # OCSVM Parameters for medium sample
    # OCSVM_PARAMS_MEDIUM = {
    #     'kernel': 'rbf',
    #     'gamma': 'scale',
    #     'nu': 0.07,  # Between small and full
    #     'cache_size': 20000,
    #     'verbose': False,
    #     'max_iter': -1
    # }


    # OCSVM Parameters for medium sample
    OCSVM_PARAMS_MEDIUM = {
        'kernel': 'sigmoid',
        'gamma': 1.0,
        'nu': 0.1,  # Between small and full
        'shrinking': True,
        'cache_size': 20000,
        'verbose': False,
        'max_iter': 1000 # 1000 is better than -1 as seen on hyperparameter tuning
    }
    
    # OCSVM Parameters for large sample
    OCSVM_PARAMS_LARGE = {
        'kernel': 'rbf',
        'gamma': 'scale',
        'nu': 0.04,  # Between small and full
        'shrinking': True,
        'cache_size': 30000,
        'verbose': False,
        'max_iter': -1 # 1000 is better than -1 as seen on hyperparameter tuning
    }
    
    # Hyperparameter search space
    HYPERPARAM_GRID = {
        'nu': [0.01, 0.05, 0.1, 0.2],
        'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1.0],
        'kernel': ['rbf', 'poly', 'sigmoid'],
        'cache_size': [1000],  # Fixed for consistency
        'shrinking': [True],   # Fixed for performance
        'max_iter': [1000]     # Limited for hyperparameter tuning
    }
    
    # Quick hyperparameter search (smaller grid for faster testing)
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
        self.mode = mode  # "full", "small", "medium", "large", "hyperparam"
    
    def load_data(self) -> pd.DataFrame:
        """Load and combine all CSV files"""
        print("="*60)
        print("LOADING DATA")
        print("="*60)
        
        mode_info = {
            "small": ("🔬 SMALL SAMPLE MODE", Config.SMALL_SAMPLE_SIZE, 20000),
            "medium": ("📊 MEDIUM SAMPLE MODE", Config.MEDIUM_SAMPLE_SIZE, 80000),
            "large": ("📈 LARGE SAMPLE MODE", Config.LARGE_SAMPLE_SIZE, 160000),
            "hyperparam": ("🎯 HYPERPARAMETER TUNING MODE", Config.HYPERPARAM_SAMPLE_SIZE, 5000),
            "full": ("🚀 FULL TRAINING MODE", None, None)
        }
        
        if self.mode in mode_info:
            print(mode_info[self.mode][0])
            if self.mode == "large":
                print(f"   Large mode: 2x medium size ({Config.LARGE_SAMPLE_SIZE:,} samples)")
        
        data_frames = []
        total_loaded = 0
        target_size = mode_info.get(self.mode, (None, None, None))[1]
        rows_per_file = mode_info.get(self.mode, (None, None, None))[2]
        
        for file_name in self.file_list:
            file_path = self.data_dir / file_name
            if file_path.exists():
                print(f"Loading: {file_name}")
                
                if rows_per_file:
                    df = pd.read_csv(file_path, nrows=rows_per_file)
                    print(f"  → Loaded {len(df):,} rows ({self.mode} mode)")
                else:
                    df = pd.read_csv(file_path)
                    print(f"  → Loaded {len(df):,} rows")
                
                data_frames.append(df)
                total_loaded += len(df)
                
                # Stop early if we have enough data
                if target_size and total_loaded >= target_size:
                    print(f"✓ Reached target sample size: {total_loaded:,} rows")
                    break
                    
            else:
                print(f"Warning: File not found: {file_path}")
        
        if not data_frames:
            raise FileNotFoundError("No data files found!")
        
        combined_data = pd.concat(data_frames, ignore_index=True)
        
        # Further sample if needed
        if target_size and len(combined_data) > target_size:
            print(f"Sampling {target_size:,} rows from {len(combined_data):,} total rows")
            combined_data = combined_data.sample(n=target_size, random_state=Config.RANDOM_SEED)
        
        print(f"✓ Final combined data shape: {combined_data.shape}")
        print(f"✓ Columns: {len(combined_data.columns)}")
        
        return combined_data


class DataPreprocessor:
    """Handle data preprocessing and cleaning"""
    
    def __init__(self, mode: str = "full"):
        self.mode = mode
        self.columns_to_drop = [
            'Flow ID', 'Source IP', 'Destination IP', 
            'Timestamp', 'Fwd Header Length.1'
        ]
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and preprocess the dataset"""
        print("\n" + "="*60)
        print("DATA CLEANING AND PREPROCESSING")
        print("="*60)
        
        if self.mode != "full":
            print(f"🔬 {self.mode.upper()} MODE: Quick preprocessing")
        
        # Clean column names and labels
        df.columns = df.columns.str.strip()
        df['Label'] = df['Label'].str.strip()
        
        # Visualize label distribution
        self._visualize_labels(df)
        
        # Drop unnecessary columns
        df = self._drop_columns(df)
        
        # Create binary target
        df['Attack'] = (df['Label'] != 'BENIGN').astype(int)
        
        # Handle problematic values
        df = self._handle_problematic_values(df)
        
        print(f"✓ Final cleaned data shape: {df.shape}")
        return df
    
    def _visualize_labels(self, df: pd.DataFrame) -> None:
        """Visualize the distribution of labels"""
        label_counts = df['Label'].value_counts()
        
        plt.figure(figsize=(12, 6))
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
        plt.show()
        
        print(f"Label distribution:\n{label_counts}")
    
    def _drop_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop unnecessary and constant columns"""
        # Drop specified columns
        for col in self.columns_to_drop:
            if col in df.columns:
                df.drop(col, axis=1, inplace=True)
                print(f"Dropped column: {col}")
        
        # Drop constant columns
        nunique = df.nunique()
        constant_cols = nunique[nunique <= 1].index.tolist()
        if constant_cols:
            df.drop(constant_cols, axis=1, inplace=True)
            print(f"Dropped {len(constant_cols)} constant columns")
        
        return df
    
    def _handle_problematic_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle infinite, NaN, and extreme values"""
        print("\nHandling problematic values...")
        
        # Separate features from labels
        feature_cols = [col for col in df.columns if col not in ['Label', 'Attack']]
        X = df[feature_cols].copy()
        
        # Handle infinite values
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
        
        # Cap extreme values using percentiles (skip for hyperparameter mode to save time)
        if self.mode in ["full", "large", "medium"]:
            for col in X.columns:
                q01, q99 = X[col].quantile([0.01, 0.99])
                if q99 > 1e8 or q01 < -1e8:
                    X[col] = X[col].clip(lower=q01, upper=q99)
        
        # Remove any remaining problematic rows
        clean_mask = ~(X.isnull().any(axis=1) | np.isinf(X.values).any(axis=1))
        if not clean_mask.all():
            print(f"Removing {(~clean_mask).sum():,} problematic rows")
            X = X[clean_mask]
            df = df[clean_mask]
        
        # Update the dataframe with cleaned features
        df[feature_cols] = X
        
        print("✓ Data cleaning completed")
        return df


class CustomDataSplitter:
    """Handle custom data splitting for OCSVM training"""
    
    def __init__(self, benign_train_ratio: float = 0.8):
        self.benign_train_ratio = benign_train_ratio
    
    def split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Custom split: 80% benign for training, 20% benign + all attacks for testing
        """
        print("\n" + "="*60)
        print("CUSTOM DATA SPLITTING")
        print("="*60)
        
        # Separate features and target
        feature_cols = [col for col in df.columns if col not in ['Label', 'Attack']]
        X = df[feature_cols]
        y = df['Attack']
        
        print(f"Total samples: {len(df):,}")
        print(f"Features: {len(feature_cols)}")
        
        # Separate benign and attack samples
        benign_mask = (y == 0)
        attack_mask = (y == 1)
        
        X_benign = X[benign_mask]
        y_benign = y[benign_mask]
        X_attack = X[attack_mask]
        y_attack = y[attack_mask]
        
        print(f"Benign samples: {len(X_benign):,}")
        print(f"Attack samples: {len(X_attack):,}")
        
        # Split benign data: 80% for training, 20% for testing
        X_benign_train, X_benign_test, y_benign_train, y_benign_test = train_test_split(
            X_benign, y_benign, 
            train_size=self.benign_train_ratio,
            random_state=Config.RANDOM_SEED,
            shuffle=True
        )
        
        print(f"\nBenign split:")
        print(f"  Training: {len(X_benign_train):,} ({self.benign_train_ratio*100:.0f}%)")
        print(f"  Testing:  {len(X_benign_test):,} ({(1-self.benign_train_ratio)*100:.0f}%)")
        
        # Combine test data: 20% benign + all attacks
        X_test = pd.concat([X_benign_test, X_attack], ignore_index=True)
        y_test = pd.concat([y_benign_test, y_attack], ignore_index=True)
        
        # Training data: 80% benign only (for OCSVM)
        X_train = X_benign_train
        y_train = y_benign_train
        
        print(f"\nFinal split:")
        print(f"  Training set: {len(X_train):,} samples (100% benign)")
        print(f"  Test set:     {len(X_test):,} samples ({len(X_benign_test):,} benign + {len(X_attack):,} attacks)")
        
        # Verify test set composition
        test_benign_count = (y_test == 0).sum()
        test_attack_count = (y_test == 1).sum()
        print(f"  Test composition: {test_benign_count:,} benign ({test_benign_count/len(y_test)*100:.1f}%) + "
              f"{test_attack_count:,} attacks ({test_attack_count/len(y_test)*100:.1f}%)")
        
        return X_train, X_test, y_train, y_test


class HyperparameterTuner:
    """Handle hyperparameter tuning for OCSVM"""
    
    def __init__(self, param_grid: Dict[str, List], quick_mode: bool = False):
        self.param_grid = param_grid
        self.quick_mode = quick_mode
        self.results = []
        self.best_params = None
        self.best_score = -np.inf
    
    def tune(self, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """Perform hyperparameter tuning"""
        print("\n" + "="*60)
        print("HYPERPARAMETER TUNING")
        print("="*60)
        
        # Limit samples for hyperparameter tuning
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
        param_names = list(self.param_grid.keys())
        param_values = list(self.param_grid.values())
        param_combinations = list(product(*param_values))
        
        total_combinations = len(param_combinations)
        print(f"Testing {total_combinations} parameter combinations")
        
        if self.quick_mode:
            print("🚀 QUICK MODE: Testing subset of combinations")
            # Test only first 10 combinations in quick mode
            param_combinations = param_combinations[:min(10, total_combinations)]
        
        print(f"Parameter grid:")
        for param, values in self.param_grid.items():
            print(f"  {param}: {values}")
        
        # Test each combination
        for i, param_combo in enumerate(param_combinations):
            params = dict(zip(param_names, param_combo))
            
            print(f"\n[{i+1}/{len(param_combinations)}] Testing: {params}")
            
            try:
                # Train model with current parameters
                model = OneClassSVM(**params, verbose=False)
                start_time = datetime.now()
                model.fit(X_train_scaled)
                training_time = (datetime.now() - start_time).total_seconds()
                
                # Evaluate model
                y_pred = model.predict(X_test_scaled)
                y_pred_binary = np.where(y_pred == 1, 0, 1)
                
                # Calculate metrics
                metrics = {
                    'accuracy': accuracy_score(y_test, y_pred_binary),
                    'precision': precision_score(y_test, y_pred_binary, zero_division=0),
                    'recall': recall_score(y_test, y_pred_binary, zero_division=0),
                    'f1_score': f1_score(y_test, y_pred_binary, zero_division=0)
                }
                
                # Store results
                result = {
                    'params': params.copy(),
                    'metrics': metrics.copy(),
                    'training_time': training_time,
                    'n_support_vectors': model.n_support_[0],
                    'support_vector_ratio': model.n_support_[0] / len(X_train_scaled)
                }
                self.results.append(result)
                
                # Update best parameters (using F1 score as primary metric)
                score = metrics['f1_score']
                if score > self.best_score:
                    self.best_score = score
                    self.best_params = params.copy()
                
                print(f"  → F1: {metrics['f1_score']:.4f}, Acc: {metrics['accuracy']:.4f}, "
                      f"Time: {training_time:.2f}s, SVs: {model.n_support_[0]:,}")
                
            except Exception as e:
                print(f"  → Failed: {e}")
                continue
        
        # Print results summary
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
            print("❌ No successful parameter combinations found!")
            return
        
        # Sort results by F1 score
        sorted_results = sorted(self.results, key=lambda x: x['metrics']['f1_score'], reverse=True)
        
        print(f"✓ Tested {len(self.results)} parameter combinations")
        print(f"🏆 Best F1 Score: {self.best_score:.4f}")
        print(f"🎯 Best Parameters: {self.best_params}")
        
        print("\n📊 Top 5 Results:")
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
    """One-Class SVM trainer with custom data splitting"""
    
    def __init__(self, config_params: Dict[str, Any], mode: str = "full"):
        self.params = config_params
        self.mode = mode
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.data_splitter = CustomDataSplitter(Config.BENIGN_TRAIN_RATIO)
        
    def prepare_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Prepare data for training using custom splitting"""
        print("\n" + "="*60)
        print("DATA PREPARATION")
        print("="*60)
        
        if self.mode != "full":
            print(f"🔬 {self.mode.upper()} MODE: Quick data preparation")
        
        # Use custom data splitter
        X_train, X_test, y_train, y_test = self.data_splitter.split_data(df)
        
        self.feature_names = list(X_train.columns)
        print(f"\n✓ Features: {len(self.feature_names)}")
        print(f"✓ Custom split completed with {Config.BENIGN_TRAIN_RATIO*100:.0f}% benign for training")
        
        return X_train, X_test, y_train, y_test
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """Train the One-Class SVM model"""
        print("\n" + "="*60)
        print("TRAINING ONE-CLASS SVM")
        print("="*60)
        
        if self.mode != "full":
            print(f"🔬 {self.mode.upper()} MODE: Fast training with limited data")
        
        # Limit benign samples based on mode (X_train is already all benign from custom split)
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
        
        print(f"Training on {len(X_train_sample):,} benign samples")
        
        # Feature scaling
        print("Applying feature scaling...")
        self.scaler = StandardScaler()
        
        try:
            X_train_scaled = self.scaler.fit_transform(X_train_sample)
            print("✓ StandardScaler applied successfully")
        except Exception as e:
            print(f"StandardScaler failed: {e}")
            print("Falling back to RobustScaler...")
            self.scaler = RobustScaler()
            X_train_scaled = self.scaler.fit_transform(X_train_sample)
        
        # Initialize and train model
        print(f"OCSVM parameters: {self.params}")
        self.model = OneClassSVM(**self.params)
        
        print("Training model...")
        start_time = datetime.now()
        
        try:
            self.model.fit(X_train_scaled)
            training_time = datetime.now() - start_time
            print(f"✓ Training completed successfully in {training_time.total_seconds():.2f} seconds!")
            print(f"Support vectors: {self.model.n_support_[0]:,}")
            print(f"Support vector ratio: {self.model.n_support_[0] / len(X_train_scaled):.4f}")
        except Exception as e:
            print(f"Training failed: {e}")
            # Fallback to even smaller sample
            fallback_sizes = {"hyperparam": 1000, "small": 5000, "medium": 20000, "large": 30000, "full": 50000}
            sample_size = min(fallback_sizes.get(self.mode, 50000), len(X_train_scaled))
            indices = np.random.choice(len(X_train_scaled), sample_size, replace=False)
            X_sample = X_train_scaled[indices]
            print(f"Training on reduced sample: {sample_size:,}")
            self.model.fit(X_sample)
            training_time = datetime.now() - start_time
            print(f"✓ Training completed on reduced dataset in {training_time.total_seconds():.2f} seconds!")
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """Evaluate the trained model"""
        print("\n" + "="*60)
        print("MODEL EVALUATION")
        print("="*60)
        
        # Scale test data
        X_test_scaled = self.scaler.transform(X_test)
        
        # Make predictions
        print("Making predictions...")
        y_pred_ocsvm = self.model.predict(X_test_scaled)
        y_pred_binary = np.where(y_pred_ocsvm == 1, 0, 1)  # Convert to binary
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred_binary),
            'precision': precision_score(y_test, y_pred_binary, zero_division=0),
            'recall': recall_score(y_test, y_pred_binary, zero_division=0),
            'f1_score': f1_score(y_test, y_pred_binary, zero_division=0)
        }
        
        # Print results
        print("📊 Performance Metrics:")
        for metric, value in metrics.items():
            print(f"{metric.capitalize()}: {value:.4f}")
        
        # Detailed report
        print("\n📋 Classification Report:")
        print(classification_report(y_test, y_pred_binary, 
                                  target_names=['Benign', 'Attack'], 
                                  zero_division=0))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred_binary)
        print("\n🎯 Confusion Matrix:")
        print("           Predicted")
        print("         Benign  Attack")
        print(f"Actual Benign   {cm[0,0]:6d}  {cm[0,1]:6d}")
        print(f"       Attack   {cm[1,0]:6d}  {cm[1,1]:6d}")
        
        # Additional analysis for custom split
        print(f"\n📈 Custom Split Analysis:")
        benign_correct = cm[0,0]
        benign_total = cm[0,0] + cm[0,1]
        attack_detected = cm[1,1]
        attack_total = cm[1,0] + cm[1,1]
        
        print(f"Benign detection rate: {benign_correct/benign_total*100:.1f}% ({benign_correct:,}/{benign_total:,})")
        print(f"Attack detection rate: {attack_detected/attack_total*100:.1f}% ({attack_detected:,}/{attack_total:,})")
        
        return metrics


class ModelSaver:
    """Handle model and artifacts saving"""
    
    def __init__(self, output_dir: str, mode: str = "full", hyperparam_results: Dict = None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.mode = mode
        self.hyperparam_results = hyperparam_results
        
        # Create timestamped subdirectory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        suffix = f"_{mode}" if mode != "full" else ""
        self.model_dir = self.output_dir / f"ocsvm_model_{timestamp}{suffix}"
        self.model_dir.mkdir(exist_ok=True)
    
    def save_model(self, trainer: OCSSVMTrainer, metrics: Dict[str, float]) -> None:
        """Save trained model and related artifacts"""
        print("\n" + "="*60)
        print("SAVING MODEL AND ARTIFACTS")
        print("="*60)
        
        if self.mode != "full":
            print(f"🔬 {self.mode.upper()} MODE: Saving model")
        
        try:
            # Save model
            model_path = self.model_dir / "ocsvm_model.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(trainer.model, f)
            print(f"✓ Model saved: {model_path}")
            
            # Save scaler
            scaler_path = self.model_dir / "feature_scaler.pkl"
            with open(scaler_path, 'wb') as f:
                pickle.dump(trainer.scaler, f)
            print(f"✓ Scaler saved: {scaler_path}")
            
            # Save feature names
            features_path = self.model_dir / "feature_names.pkl"
            with open(features_path, 'wb') as f:
                pickle.dump(trainer.feature_names, f)
            print(f"✓ Feature names saved: {features_path}")
            
            # Save metrics
            metrics_path = self.model_dir / "metrics.pkl"
            with open(metrics_path, 'wb') as f:
                pickle.dump(metrics, f)
            print(f"✓ Metrics saved: {metrics_path}")
            
            # Save hyperparameter results if available
            if self.hyperparam_results:
                hyperparam_path = self.model_dir / "hyperparameter_results.pkl"
                with open(hyperparam_path, 'wb') as f:
                    pickle.dump(self.hyperparam_results, f)
                print(f"✓ Hyperparameter results saved: {hyperparam_path}")
                
                # Save hyperparameter results as JSON for easy reading
                hyperparam_json_path = self.model_dir / "hyperparameter_results.json"
                json_results = {
                    'best_params': self.hyperparam_results['best_params'],
                    'best_score': float(self.hyperparam_results['best_score']),
                    'total_combinations_tested': len(self.hyperparam_results['all_results'])
                }
                with open(hyperparam_json_path, 'w') as f:
                    json.dump(json_results, f, indent=2)
                print(f"✓ Hyperparameter summary saved: {hyperparam_json_path}")
            
            # Save configuration
            config_path = self.model_dir / "config.pkl"
            config_data = {
                'ocsvm_params': trainer.params,
                'random_seed': Config.RANDOM_SEED,
                'benign_train_ratio': Config.BENIGN_TRAIN_RATIO,
                'mode': self.mode,
                'timestamp': datetime.now().isoformat()
            }
            with open(config_path, 'wb') as f:
                pickle.dump(config_data, f)
            print(f"✓ Configuration saved: {config_path}")
            
            # Create summary file
            self._create_summary(metrics, trainer.params)
            
            print(f"\n🎉 All artifacts saved to: {self.model_dir}")
            
        except Exception as e:
            print(f"❌ Error saving model: {e}")
    
    def _create_summary(self, metrics: Dict[str, float], params: Dict[str, Any]) -> None:
        """Create a human-readable summary file"""
        summary_path = self.model_dir / "model_summary.txt"
        
        with open(summary_path, 'w') as f:
            f.write("OCSVM Intrusion Detection Model Summary\n")
            f.write("="*50 + "\n\n")
            f.write(f"Training Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Mode: {self.mode.title()}\n")
            f.write(f"Random Seed: {Config.RANDOM_SEED}\n")
            f.write(f"Benign Train Ratio: {Config.BENIGN_TRAIN_RATIO}\n")
            f.write(f"Data Split: {Config.BENIGN_TRAIN_RATIO*100:.0f}% benign for training, "
                   f"{(1-Config.BENIGN_TRAIN_RATIO)*100:.0f}% benign + all attacks for testing\n\n")
            
            # Add sample size information
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
            
            f.write("Performance Metrics:\n")
            for metric, value in metrics.items():
                f.write(f"  {metric.capitalize()}: {value:.4f}\n")
            
            if self.hyperparam_results:
                f.write(f"\nHyperparameter Tuning Results:\n")
                f.write(f"  Best F1 Score: {self.hyperparam_results['best_score']:.4f}\n")
                f.write(f"  Best Parameters: {self.hyperparam_results['best_params']}\n")
                f.write(f"  Total Combinations Tested: {len(self.hyperparam_results['all_results'])}\n")
            
            if self.mode != "full":
                f.write("\n" + "="*50 + "\n")
                f.write(f"NOTE: This is a {self.mode} mode model.\n")
                if self.mode == "large":
                    f.write("Large mode provides high-quality training with reasonable time.\n")
                elif self.mode == "medium":
                    f.write("Medium mode provides balanced performance and training time.\n")
                elif self.mode != "hyperparam":
                    f.write("For production use, consider running with full dataset.\n")
        
        print(f"✓ Summary saved: {summary_path}")


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Train OCSVM Intrusion Detection System with Custom Data Splitting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python trainer.py                           # Full training
  python trainer.py --small-sample            # Quick test (100K samples)
  python trainer.py --medium-sample           # Medium test (500K samples)
  python trainer.py --large-sample            # Large test (1M samples, 2x medium)
  python trainer.py --hyperparam-tune         # Hyperparameter tuning
  python trainer.py --hyperparam-tune --quick # Quick hyperparameter tuning
  python trainer.py -s                        # Short form for small sample
  python trainer.py -m                        # Short form for medium sample
  python trainer.py -l                        # Short form for large sample

Dataset Size Comparison:
  Small:  100K samples,  50K benign training
  Medium: 500K samples, 200K benign training  
  Large:   1M samples, 400K benign training (2x medium)
  Full:   All available data

Data Splitting Strategy:
  - 80% of benign data → training (OCSVM learns normal behavior)
  - 20% of benign data + ALL attack data → testing (realistic evaluation)
        """
    )
    
    parser.add_argument(
        '--small-sample', '-s',
        action='store_true',
        help='Use small sample for quick testing (100K samples, faster training)'
    )
    
    parser.add_argument(
        '--medium-sample', '-m',
        action='store_true',
        help='Use medium sample for balanced testing (500K samples, moderate training time)'
    )
    
    parser.add_argument(
        '--large-sample', '-l',
        action='store_true',
        help='Use large sample for high-quality training (1M samples, 2x medium size)'
    )
    
    parser.add_argument(
        '--hyperparam-tune', '--tune',
        action='store_true',
        help='Perform hyperparameter tuning to find best parameters'
    )
    
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Quick mode for hyperparameter tuning (tests fewer combinations)'
    )
    
    parser.add_argument(
        '--data-dir',
        default=Config.DATA_DIR,
        help=f'Directory containing CSV files (default: {Config.DATA_DIR})'
    )
    
    parser.add_argument(
        '--output-dir',
        default=Config.OUTPUT_DIR,
        help=f'Output directory for model artifacts (default: {Config.OUTPUT_DIR})'
    )
    
    return parser.parse_args()


def main():
    """Main training pipeline"""
    args = parse_arguments()
    
    # Update config based on arguments
    Config.DATA_DIR = args.data_dir
    Config.OUTPUT_DIR = args.output_dir
    
    # Determine mode (priority: hyperparam > large > medium > small > full)
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
        mode_desc = "FULL TRAINING"
    
    print(f"🚀 Starting OCSVM Intrusion Detection Training Pipeline")
    print(f"📋 Mode: {mode_desc}")
    if mode == "large":
        print(f"   📈 Large mode: 2x medium size ({Config.LARGE_SAMPLE_SIZE:,} samples)")
    print(f"📁 Data directory: {Config.DATA_DIR}")
    print(f"📁 Output directory: {Config.OUTPUT_DIR}")
    print(f"🔄 Data split: {Config.BENIGN_TRAIN_RATIO*100:.0f}% benign for training, "
          f"{(1-Config.BENIGN_TRAIN_RATIO)*100:.0f}% benign + all attacks for testing")
    
    if args.quick and args.hyperparam_tune:
        print("⚡ Quick hyperparameter tuning enabled")
    
    # Create output directory
    Path(Config.OUTPUT_DIR).mkdir(exist_ok=True)
    
    try:
        # Load data
        loader = DataLoader(Config.DATA_DIR, Config.FILE_LIST, mode)
        raw_data = loader.load_data()
        
        # Preprocess data
        preprocessor = DataPreprocessor(mode)
        clean_data = preprocessor.clean_data(raw_data)
        
        # Prepare data with custom splitting
        param_configs = {
            "hyperparam": Config.OCSVM_PARAMS_SMALL,
            "small": Config.OCSVM_PARAMS_SMALL,
            "medium": Config.OCSVM_PARAMS_MEDIUM,
            "large": Config.OCSVM_PARAMS_LARGE,
            "full": Config.OCSVM_PARAMS
        }
        
        ocsvm_params = param_configs[mode]
        trainer = OCSSVMTrainer(ocsvm_params, mode)
        X_train, X_test, y_train, y_test = trainer.prepare_data(clean_data)
        
        hyperparam_results = None
        
        # Hyperparameter tuning
        if args.hyperparam_tune:
            param_grid = Config.HYPERPARAM_GRID_QUICK if args.quick else Config.HYPERPARAM_GRID
            tuner = HyperparameterTuner(param_grid, args.quick)
            hyperparam_results = tuner.tune(X_train, y_train, X_test, y_test)
            
            # Update trainer with best parameters
            if hyperparam_results['best_params']:
                print(f"\n🎯 Using best parameters for final training: {hyperparam_results['best_params']}")
                trainer.params = hyperparam_results['best_params']
            else:
                print("\n⚠️  No successful hyperparameter combinations found, using default parameters")
        
        # Train model
        trainer.train(X_train, y_train)
        
        # Evaluate model
        metrics = trainer.evaluate(X_test, y_test)
        
        # Save model
        saver = ModelSaver(Config.OUTPUT_DIR, mode, hyperparam_results)
        saver.save_model(trainer, metrics)
        
        print(f"\n🎉 Training pipeline completed successfully!")
        print(f"📊 Final F1 Score: {metrics['f1_score']:.4f}")
        print(f"📊 Final Accuracy: {metrics['accuracy']:.4f}")
        print(f"📁 Model saved in: {saver.model_dir}")
        
        if args.hyperparam_tune and hyperparam_results and hyperparam_results['best_params']:
            print(f"🏆 Best hyperparameters: {hyperparam_results['best_params']}")
            print(f"🏆 Best F1 score from tuning: {hyperparam_results['best_score']:.4f}")
        
        if mode != "full":
            print(f"\n⚠️  NOTE: This was a {mode} mode run.")
            if mode == "large":
                print("   Large mode provides high-quality training with reasonable time.")
            elif mode == "medium":
                print("   Medium mode provides balanced performance and training time.")
            elif mode != "hyperparam":
                print("   For production use, consider running with full dataset.")
        
    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    # Set random seeds for reproducibility
    np.random.seed(Config.RANDOM_SEED)
    
    # Run the training pipeline
    main()
