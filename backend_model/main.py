#!/usr/bin/env python3
"""
FastAPI Backend for OCSVM Intrusion Detection System
Provides real-time network traffic classification using a pre-trained One-Class SVM model
"""

import pickle
import warnings
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, status, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
import uvicorn

examples = 0

# Suppress warnings
warnings.filterwarnings('ignore')

# Configuration
class Config:
    MODEL_DIR = os.getenv("MODEL_DIR", "production_model")
    MODEL_NAME = "ocsvm_model.pkl"
    SCALER_NAME = "feature_scaler.pkl"
    FEATURE_NAMES_FILE = "feature_names.pkl"
    CONFIG_FILE = "config.pkl"
    
    COLUMNS_TO_DROP = [
        'Flow ID', 'Source IP', 'Destination IP', 
        'Timestamp', 'Fwd Header Length.1',
        'Label', 'Attack'
    ]


# Pydantic models for request/response
class NetworkFlowInput(BaseModel):
    """Input model for a single network flow record."""
    data: Dict[str, float] = Field(
        ...,
        description="Dictionary of feature names and their values",
        example={
            "Destination Port": 80,
            "Flow Duration": 120000,
            "Total Fwd Packets": 10,
            "Total Backward Packets": 8,
            "Total Length of Fwd Packets": 5000,
            "Total Length of Bwd Packets": 3000,
        }
    )
    
    @validator('data')
    def validate_data(cls, v):
        if not v:
            raise ValueError("Data dictionary cannot be empty")
        return v


class BatchNetworkFlowInput(BaseModel):
    """Input model for batch predictions"""
    flows: List[Dict[str, float]] = Field(
        ...,
        description="List of network flow records",
        min_items=1
    )


class PredictionOutput(BaseModel):
    """Output model for predictions"""
    prediction: str = Field(..., description="BENIGN or ATTACK")
    confidence: float = Field(..., description="Confidence score (-1 to 1, where -1 is benign, 1 is attack)")
    is_anomaly: bool = Field(..., description="True if classified as attack")


class BatchPredictionOutput(BaseModel):
    """Output model for batch predictions"""
    predictions: List[PredictionOutput]
    total_flows: int
    benign_count: int
    attack_count: int


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    model_info: Optional[Dict[str, Any]] = None


class CSVFlowInput(BaseModel):
    """
    Input model for CSV string format network flow.
    Accepts a single comma-separated string or array from CICFlowMeter.
    """
    csv: Union[str, List] = Field(
        ...,
        description="Comma-separated string or array of network flow values"
    )
    
    @validator('csv')
    def validate_and_parse_csv(cls, v):
        # If it's already a list, validate length
        if isinstance(v, list):
            if len(v) not in [80, 82]:
                raise ValueError(f"CSV array must have 80 or 82 fields, got {len(v)}")
            return v
        
        # If it's a string, parse it
        if isinstance(v, str):
            if not v:
                raise ValueError("CSV string cannot be empty")
            
            # Split by comma and strip whitespace
            fields = [field.strip() for field in v.split(',')]
            
            # Validate field count
            if len(fields) not in [80, 82]:
                raise ValueError(f"CSV must have 80 or 82 fields, got {len(fields)}")
            
            return fields
        
        raise ValueError("CSV must be a string or list")


class BatchCSVFlowInput(BaseModel):
    """Input model for batch CSV predictions"""
    flows: List[Union[str, List]] = Field(
        ...,
        description="List of comma-separated CSV strings or arrays",
        min_items=1
    )


class DataPreprocessor:
    """Handle data preprocessing matching the training pipeline"""
    
    def __init__(self, feature_names: List[str]):
        self.feature_names = feature_names
        self.columns_to_drop = Config.COLUMNS_TO_DROP
    
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess input data to match training pipeline"""
        df.columns = df.columns.str.strip()
        
        missing_features = set(self.feature_names) - set(df.columns)
        if missing_features:
            for feature in missing_features:
                df[feature] = 0.0
        
        df = df[self.feature_names]
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(0, inplace=True)
        df = df.apply(pd.to_numeric, errors='coerce')
        df.fillna(0, inplace=True)
        
        return df


class ModelManager:
    """Manage model loading and predictions"""
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.config = None
        self.preprocessor = None
        self.model_loaded = False
        
    def load_model(self, model_dir: str):
        """Load the trained OCSVM model and preprocessing artifacts"""
        model_path = Path(model_dir)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model directory not found: {model_dir}")
        
        try:
            with open(model_path / Config.MODEL_NAME, 'rb') as f:
                self.model = pickle.load(f)
            
            with open(model_path / Config.SCALER_NAME, 'rb') as f:
                self.scaler = pickle.load(f)
            
            with open(model_path / Config.FEATURE_NAMES_FILE, 'rb') as f:
                self.feature_names = pickle.load(f)
            
            with open(model_path / Config.CONFIG_FILE, 'rb') as f:
                self.config = pickle.load(f)
            
            summary_file = model_path / "model_summary.txt"
            if summary_file.exists():
                self.config.update(self._parse_model_summary(summary_file))
            
            self.preprocessor = DataPreprocessor(self.feature_names)
            
            self.model_loaded = True
            print(f"✓ Model loaded successfully from {model_dir}")
            print(f"✓ Number of features: {len(self.feature_names)}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {str(e)}")
    
    def _parse_model_summary(self, summary_file: Path) -> Dict[str, Any]:
        """Parse model_summary.txt to extract model parameters"""
        params = {}
        try:
            with open(summary_file, 'r') as f:
                lines = f.readlines()
                
            in_params_section = False
            in_metrics_section = False
            
            for line in lines:
                line = line.strip()
                
                if "Model Parameters:" in line:
                    in_params_section = True
                    in_metrics_section = False
                    continue
                elif "Performance Metrics:" in line:
                    in_params_section = False
                    in_metrics_section = True
                    continue
                elif line.startswith("=="):
                    continue
                
                if in_params_section and ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip()
                    value = value.strip()
                    
                    if value.lower() == 'true':
                        params[key] = True
                    elif value.lower() == 'false':
                        params[key] = False
                    elif value.isdigit():
                        params[key] = int(value)
                    else:
                        try:
                            params[key] = float(value)
                        except ValueError:
                            params[key] = value
                
                if in_metrics_section and ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip()
                    value = value.strip()
                    
                    try:
                        params[key] = float(value)
                    except ValueError:
                        params[key] = value
            
            print(f"✓ Parsed {len(params)} parameters from model_summary.txt")
            
        except Exception as e:
            print(f"⚠️  Warning: Could not parse model_summary.txt: {str(e)}")
        
        return params
    
    def predict_single(self, data: Dict[str, float]) -> PredictionOutput:
        """Make prediction for a single network flow"""
        if not self.model_loaded:
            raise RuntimeError("Model not loaded")
        
        df = pd.DataFrame([data])
        df_processed = self.preprocessor.preprocess(df)
        X_scaled = self.scaler.transform(df_processed)
        
        prediction = self.model.predict(X_scaled)[0]
        decision_score = self.model.decision_function(X_scaled)[0]
        
        is_attack = prediction == -1
        label = "ATTACK" if is_attack else "BENIGN"
        
        return PredictionOutput(
            prediction=label,
            confidence=float(decision_score),
            is_anomaly=is_attack
        )
    
    def predict_batch(self, flows: List[Dict[str, float]]) -> BatchPredictionOutput:
        """Make predictions for multiple network flows"""
        if not self.model_loaded:
            raise RuntimeError("Model not loaded")
        
        df = pd.DataFrame(flows)
        df_processed = self.preprocessor.preprocess(df)
        X_scaled = self.scaler.transform(df_processed)
        
        predictions = self.model.predict(X_scaled)
        decision_scores = self.model.decision_function(X_scaled)
        
        results = []
        attack_count = 0
        
        for pred, score in zip(predictions, decision_scores):
            is_attack = pred == -1
            label = "ATTACK" if is_attack else "BENIGN"
            
            if is_attack:
                attack_count += 1
            
            results.append(PredictionOutput(
                prediction=label,
                confidence=float(score),
                is_anomaly=is_attack
            ))
        
        return BatchPredictionOutput(
            predictions=results,
            total_flows=len(flows),
            benign_count=len(flows) - attack_count,
            attack_count=attack_count
        )
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        if not self.model_loaded:
            return {}
        
        info = {
            "model_type": "One-Class SVM",
            "kernel": self.config.get('kernel', 'N/A'),
            "nu": self.config.get('nu', 'N/A'),
            "gamma": self.config.get('gamma', 'N/A'),
            "n_features": len(self.feature_names),
            "features": self.feature_names[:10] + ["..."] if len(self.feature_names) > 10 else self.feature_names
        }
        
        if 'shrinking' in self.config:
            info['shrinking'] = self.config['shrinking']
        if 'cache_size' in self.config:
            info['cache_size'] = self.config['cache_size']
        if 'max_iter' in self.config:
            info['max_iter'] = self.config['max_iter']
        
        metrics = {}
        for metric in ['Accuracy', 'Precision', 'Recall', 'F1_score']:
            if metric in self.config:
                metrics[metric] = self.config[metric]
        
        if metrics:
            info['performance_metrics'] = metrics
        
        return info


# Initialize FastAPI app
app = FastAPI(
    title="OCSVM Intrusion Detection API",
    description="Real-time network intrusion detection using One-Class SVM",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model_manager = ModelManager()


@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    try:
        model_manager.load_model(Config.MODEL_DIR)
        print("🚀 API is ready to accept requests")
    except Exception as e:
        print(f"❌ Failed to load model: {str(e)}")
        print("⚠️  API will start but predictions will fail until model is loaded")


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "message": "OCSVM Intrusion Detection API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if model_manager.model_loaded else "degraded",
        model_loaded=model_manager.model_loaded,
        model_info=model_manager.get_model_info() if model_manager.model_loaded else None
    )


@app.post("/predict", response_model=PredictionOutput)
async def predict(input_data: NetworkFlowInput):
    """Predict if a single network flow is benign or an attack"""
    try:
        result = model_manager.predict_single(input_data.data)
        return result
    except RuntimeError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/predict/batch", response_model=BatchPredictionOutput)
async def predict_batch(input_data: BatchNetworkFlowInput):
    """Predict multiple network flows at once"""
    try:
        result = model_manager.predict_batch(input_data.flows)
        return result
    except RuntimeError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Batch prediction failed: {str(e)}"
        )


@app.get("/features", response_model=Dict[str, Any])
async def get_features():
    """Get the list of features expected by the model"""
    if not model_manager.model_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    return {
        "feature_count": len(model_manager.feature_names),
        "features": model_manager.feature_names
    }


@app.get("/model/info", response_model=Dict[str, Any])
async def get_model_info():
    """Get information about the loaded model"""
    if not model_manager.model_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    return model_manager.get_model_info()


# CSV Mapping and Column Orders
CSV_TO_FEATURE_MAPPING = {
    'dst_port': 'Destination Port',
    'flow_duration': 'Flow Duration',
    'tot_fwd_pkts': 'Total Fwd Packets',
    'tot_bwd_pkts': 'Total Backward Packets',
    'totlen_fwd_pkts': 'Total Length of Fwd Packets',
    'totlen_bwd_pkts': 'Total Length of Bwd Packets',
    'fwd_pkt_len_max': 'Fwd Packet Length Max',
    'fwd_pkt_len_min': 'Fwd Packet Length Min',
    'fwd_pkt_len_mean': 'Fwd Packet Length Mean',
    'fwd_pkt_len_std': 'Fwd Packet Length Std',
    'bwd_pkt_len_max': 'Bwd Packet Length Max',
    'bwd_pkt_len_min': 'Bwd Packet Length Min',
    'bwd_pkt_len_mean': 'Bwd Packet Length Mean',
    'bwd_pkt_len_std': 'Bwd Packet Length Std',
    'flow_byts_s': 'Flow Bytes/s',
    'flow_pkts_s': 'Flow Packets/s',
    'flow_iat_mean': 'Flow IAT Mean',
    'flow_iat_std': 'Flow IAT Std',
    'flow_iat_max': 'Flow IAT Max',
    'flow_iat_min': 'Flow IAT Min',
    'fwd_iat_tot': 'Fwd IAT Total',
    'fwd_iat_mean': 'Fwd IAT Mean',
    'fwd_iat_std': 'Fwd IAT Std',
    'fwd_iat_max': 'Fwd IAT Max',
    'fwd_iat_min': 'Fwd IAT Min',
    'bwd_iat_tot': 'Bwd IAT Total',
    'bwd_iat_mean': 'Bwd IAT Mean',
    'bwd_iat_std': 'Bwd IAT Std',
    'bwd_iat_max': 'Bwd IAT Max',
    'bwd_iat_min': 'Bwd IAT Min',
    'fwd_psh_flags': 'Fwd PSH Flags',
    'bwd_psh_flags': 'Bwd PSH Flags',
    'fwd_urg_flags': 'Fwd URG Flags',
    'bwd_urg_flags': 'Bwd URG Flags',
    'fwd_header_len': 'Fwd Header Length',
    'bwd_header_len': 'Bwd Header Length',
    'fwd_pkts_s': 'Fwd Packets/s',
    'bwd_pkts_s': 'Bwd Packets/s',
    'pkt_len_min': 'Min Packet Length',
    'pkt_len_max': 'Max Packet Length',
    'pkt_len_mean': 'Packet Length Mean',
    'pkt_len_std': 'Packet Length Std',
    'pkt_len_var': 'Packet Length Variance',
    'fin_flag_cnt': 'FIN Flag Count',
    'syn_flag_cnt': 'SYN Flag Count',
    'rst_flag_cnt': 'RST Flag Count',
    'psh_flag_cnt': 'PSH Flag Count',
    'ack_flag_cnt': 'ACK Flag Count',
    'urg_flag_cnt': 'URG Flag Count',
    'cwr_flag_count': 'CWE Flag Count',
    'ece_flag_cnt': 'ECE Flag Count',
    'down_up_ratio': 'Down/Up Ratio',
    'pkt_size_avg': 'Average Packet Size',
    'fwd_seg_size_avg': 'Avg Fwd Segment Size',
    'bwd_seg_size_avg': 'Avg Bwd Segment Size',
    'fwd_byts_b_avg': 'Fwd Avg Bytes/Bulk',
    'fwd_pkts_b_avg': 'Fwd Avg Packets/Bulk',
    'fwd_blk_rate_avg': 'Fwd Avg Bulk Rate',
    'bwd_byts_b_avg': 'Bwd Avg Bytes/Bulk',
    'bwd_pkts_b_avg': 'Bwd Avg Packets/Bulk',
    'bwd_blk_rate_avg': 'Bwd Avg Bulk Rate',
    'subflow_fwd_pkts': 'Subflow Fwd Packets',
    'subflow_fwd_byts': 'Subflow Fwd Bytes',
    'subflow_bwd_pkts': 'Subflow Bwd Packets',
    'subflow_bwd_byts': 'Subflow Bwd Bytes',
    'init_fwd_win_byts': 'Init_Win_bytes_forward',
    'init_bwd_win_byts': 'Init_Win_bytes_backward',
    'fwd_act_data_pkts': 'act_data_pkt_fwd',
    'fwd_seg_size_min': 'min_seg_size_forward',
    'active_mean': 'Active Mean',
    'active_std': 'Active Std',
    'active_max': 'Active Max',
    'active_min': 'Active Min',
    'idle_mean': 'Idle Mean',
    'idle_std': 'Idle Std',
    'idle_max': 'Idle Max',
    'idle_min': 'Idle Min',
}

# 82 fields (with src_ip, dst_ip)
CSV_COLUMN_ORDER = [
    'src_ip', 'dst_ip', 'src_port', 'dst_port', 'protocol', 'timestamp',
    'flow_duration', 'flow_byts_s', 'flow_pkts_s', 'fwd_pkts_s', 'bwd_pkts_s',
    'tot_fwd_pkts', 'tot_bwd_pkts', 'totlen_fwd_pkts', 'totlen_bwd_pkts',
    'fwd_pkt_len_max', 'fwd_pkt_len_min', 'fwd_pkt_len_mean', 'fwd_pkt_len_std',
    'bwd_pkt_len_max', 'bwd_pkt_len_min', 'bwd_pkt_len_mean', 'bwd_pkt_len_std',
    'pkt_len_max', 'pkt_len_min', 'pkt_len_mean', 'pkt_len_std', 'pkt_len_var',
    'fwd_header_len', 'bwd_header_len', 'fwd_seg_size_min', 'fwd_act_data_pkts',
    'flow_iat_mean', 'flow_iat_max', 'flow_iat_min', 'flow_iat_std',
    'fwd_iat_tot', 'fwd_iat_max', 'fwd_iat_min', 'fwd_iat_mean', 'fwd_iat_std',
    'bwd_iat_tot', 'bwd_iat_max', 'bwd_iat_min', 'bwd_iat_mean', 'bwd_iat_std',
    'fwd_psh_flags', 'bwd_psh_flags', 'fwd_urg_flags', 'bwd_urg_flags',
    'fin_flag_cnt', 'syn_flag_cnt', 'rst_flag_cnt', 'psh_flag_cnt',
    'ack_flag_cnt', 'urg_flag_cnt', 'ece_flag_cnt', 'down_up_ratio',
    'pkt_size_avg', 'init_fwd_win_byts', 'init_bwd_win_byts',
    'active_max', 'active_min', 'active_mean', 'active_std',
    'idle_max', 'idle_min', 'idle_mean', 'idle_std',
    'fwd_byts_b_avg', 'fwd_pkts_b_avg', 'bwd_byts_b_avg', 'bwd_pkts_b_avg',
    'fwd_blk_rate_avg', 'bwd_blk_rate_avg', 'fwd_seg_size_avg', 'bwd_seg_size_avg',
    'cwr_flag_count', 'subflow_fwd_pkts', 'subflow_bwd_pkts',
    'subflow_fwd_byts', 'subflow_bwd_byts'
]

# 80 fields (without src_ip, dst_ip) - CICFlowMeter format
CICFLOWMETER_COLUMN_ORDER = [
    'src_port', 'dst_port', 'protocol', 'timestamp',
    'flow_duration', 'flow_byts_s', 'flow_pkts_s', 'fwd_pkts_s', 'bwd_pkts_s',
    'tot_fwd_pkts', 'tot_bwd_pkts', 'totlen_fwd_pkts', 'totlen_bwd_pkts',
    'fwd_pkt_len_max', 'fwd_pkt_len_min', 'fwd_pkt_len_mean', 'fwd_pkt_len_std',
    'bwd_pkt_len_max', 'bwd_pkt_len_min', 'bwd_pkt_len_mean', 'bwd_pkt_len_std',
    'pkt_len_max', 'pkt_len_min', 'pkt_len_mean', 'pkt_len_std', 'pkt_len_var',
    'fwd_header_len', 'bwd_header_len', 'fwd_seg_size_min', 'fwd_act_data_pkts',
    'flow_iat_mean', 'flow_iat_max', 'flow_iat_min', 'flow_iat_std',
    'fwd_iat_tot', 'fwd_iat_max', 'fwd_iat_min', 'fwd_iat_mean', 'fwd_iat_std',
    'bwd_iat_tot', 'bwd_iat_max', 'bwd_iat_min', 'bwd_iat_mean', 'bwd_iat_std',
    'fwd_psh_flags', 'bwd_psh_flags', 'fwd_urg_flags', 'bwd_urg_flags',
    'fin_flag_cnt', 'syn_flag_cnt', 'rst_flag_cnt', 'psh_flag_cnt',
    'ack_flag_cnt', 'urg_flag_cnt', 'ece_flag_cnt', 'down_up_ratio',
    'pkt_size_avg', 'init_fwd_win_byts', 'init_bwd_win_byts',
    'active_max', 'active_min', 'active_mean', 'active_std',
    'idle_max', 'idle_min', 'idle_mean', 'idle_std',
    'fwd_byts_b_avg', 'fwd_pkts_b_avg', 'bwd_byts_b_avg', 'bwd_pkts_b_avg',
    'fwd_blk_rate_avg', 'bwd_blk_rate_avg', 'fwd_seg_size_avg', 'bwd_seg_size_avg',
    'cwr_flag_count', 'subflow_fwd_pkts', 'subflow_bwd_pkts',
    'subflow_fwd_byts', 'subflow_bwd_byts'
]


def csv_array_to_feature_dict(csv_array: List[str]) -> Dict[str, float]:
    """
    Convert CSV array (list of strings) to feature dictionary.
    Handles both 80-field (CICFlowMeter) and 82-field (original) formats.
    """
    # Determine format
    if len(csv_array) == 82:
        column_order = CSV_COLUMN_ORDER
    elif len(csv_array) == 80:
        column_order = CICFLOWMETER_COLUMN_ORDER
    else:
        raise ValueError(f"Unexpected CSV length: {len(csv_array)}")
    
    # Create dictionary from CSV array
    csv_dict = {}
    for i, col_name in enumerate(column_order):
        if i < len(csv_array):
            csv_dict[col_name] = csv_array[i]
    
    # Map to model feature names, excluding dropped columns
    feature_dict = {}
    columns_to_exclude = ['src_ip', 'dst_ip', 'src_port', 'protocol', 'timestamp']
    
    for csv_col, model_col in CSV_TO_FEATURE_MAPPING.items():
        if csv_col in csv_dict and csv_col not in columns_to_exclude:
            try:
                value = csv_dict[csv_col]
                if value == '' or value is None:
                    feature_dict[model_col] = 0.0
                else:
                    feature_dict[model_col] = float(value)
            except (ValueError, TypeError) as e:
                print(f"Warning: Could not convert {csv_col}={csv_dict[csv_col]} to float: {e}")
                feature_dict[model_col] = 0.0
    
    return feature_dict


@app.post("/predict/csv", response_model=PredictionOutput)
async def predict_from_csv(input_data: CSVFlowInput):
    """
    Predict if a network flow is benign or an attack from CSV format.
    
    Accepts either:
    - A comma-separated string: "192.168.10.3,192.168.10.1,61834,53,..."
    - An array: ["192.168.10.3", "192.168.10.1", "61834", "53", ...]
    
    Supports both 80 fields (CICFlowMeter) and 82 fields (with IPs).
    """
    global examples

    try:
        # input_data.csv is now a list of strings (parsed by validator)
        csv_fields = input_data.csv
        
        # Debug: Print first example
        if examples == 0:
            examples += 1
            print(f"\n===== Received CSV Input ({len(csv_fields)} fields) =====")
            print(f"First 10 fields: {csv_fields[:10]}")
            print(f"Last 5 fields: {csv_fields[-5:]}")
            print("================================\n")
        
        # Convert CSV array to feature dictionary
        feature_dict = csv_array_to_feature_dict(csv_fields)
        
        # Debug: Print feature dict for first example
        if examples == 1:
            print(f"\n===== Converted Feature Dict ({len(feature_dict)} features) =====")
            feature_names = list(feature_dict.keys())
            print(f"First 5 features: {feature_names[:5]}")
            print(f"First 5 values: {[feature_dict[k] for k in feature_names[:5]]}")
            print("================================\n")
        
        # Make prediction
        result = model_manager.predict_single(feature_dict)
        
        # Log prediction result
        print(f"✓ Prediction: {result.prediction} (confidence: {result.confidence:.4f})")
        
        return result
        
    except RuntimeError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(e)
        )
    except Exception as e:
        import traceback
        print(f"❌ Error in prediction: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/predict/csv/batch", response_model=BatchPredictionOutput)
async def predict_batch_from_csv(input_data: BatchCSVFlowInput):
    """
    Predict multiple network flows from CSV format.
    
    Accepts a list of CSV strings or arrays.
    """
    try:
        feature_dicts = []
        for i, csv_item in enumerate(input_data.flows):
            # Parse if string
            if isinstance(csv_item, str):
                fields = [field.strip() for field in csv_item.split(',')]
            else:
                fields = csv_item
            
            if len(fields) not in [80, 82]:
                raise ValueError(f"Flow {i}: Expected 80 or 82 fields, got {len(fields)}")
            
            feature_dicts.append(csv_array_to_feature_dict(fields))
        
        result = model_manager.predict_batch(feature_dicts)
        return result
        
    except RuntimeError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(e)
        )
    except Exception as e:
        import traceback
        print(f"❌ Error in batch prediction: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Batch prediction failed: {str(e)}"
        )


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
