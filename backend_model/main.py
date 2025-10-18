#!/usr/bin/env python3
"""
FastAPI Backend for OCSVM Intrusion Detection System
Provides real-time network traffic classification using a pre-trained One-Class SVM model
"""

import pickle
import warnings
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
import uvicorn

# Suppress warnings
warnings.filterwarnings('ignore')

# Configuration
class Config:
    # For local development, use production_model directory
    # For Docker, model is copied to /app/model/
    MODEL_DIR = os.getenv("MODEL_DIR", "production_model")
    MODEL_NAME = "ocsvm_model.pkl"
    SCALER_NAME = "feature_scaler.pkl"
    FEATURE_NAMES_FILE = "feature_names.pkl"
    CONFIG_FILE = "config.pkl"
    
    # Columns that should be excluded from features
    COLUMNS_TO_DROP = [
        'Flow ID', 'Source IP', 'Destination IP', 
        'Timestamp', 'Fwd Header Length.1',
        'Label', 'Attack'
    ]


# Pydantic models for request/response
class NetworkFlowInput(BaseModel):
    """
    Input model for a single network flow record.
    All feature names should match the training data columns.
    """
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
            # ... more features
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


class DataPreprocessor:
    """Handle data preprocessing matching the training pipeline"""
    
    def __init__(self, feature_names: List[str]):
        self.feature_names = feature_names
        self.columns_to_drop = Config.COLUMNS_TO_DROP
    
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess input data to match training pipeline
        """
        # Clean column names
        df.columns = df.columns.str.strip()
        
        # Ensure all expected features are present
        missing_features = set(self.feature_names) - set(df.columns)
        if missing_features:
            # Add missing features with default value 0
            for feature in missing_features:
                df[feature] = 0.0
        
        # Select only the features used in training (in the same order)
        df = df[self.feature_names]
        
        # Handle infinite values
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # Handle NaN values - fill with median (0 for initial request)
        df.fillna(0, inplace=True)
        
        # Ensure all values are numeric
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
            # Load model
            with open(model_path / Config.MODEL_NAME, 'rb') as f:
                self.model = pickle.load(f)
            
            # Load scaler
            with open(model_path / Config.SCALER_NAME, 'rb') as f:
                self.scaler = pickle.load(f)
            
            # Load feature names
            with open(model_path / Config.FEATURE_NAMES_FILE, 'rb') as f:
                self.feature_names = pickle.load(f)
            
            # Load config from pickle (may not have all params)
            with open(model_path / Config.CONFIG_FILE, 'rb') as f:
                self.config = pickle.load(f)
            
            # Parse model_summary.txt for accurate parameters
            summary_file = model_path / "model_summary.txt"
            if summary_file.exists():
                self.config.update(self._parse_model_summary(summary_file))
            
            # Initialize preprocessor
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
                
                # Detect sections
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
                
                # Parse parameters
                if in_params_section and ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip()
                    value = value.strip()
                    
                    # Convert values to appropriate types
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
                
                # Parse metrics
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
        
        # Convert to DataFrame
        df = pd.DataFrame([data])
        
        # Preprocess
        df_processed = self.preprocessor.preprocess(df)
        
        # Scale features
        X_scaled = self.scaler.transform(df_processed)
        
        # Predict (OCSVM returns 1 for inliers/benign, -1 for outliers/attacks)
        prediction = self.model.predict(X_scaled)[0]
        
        # Get decision function score (distance from hyperplane)
        decision_score = self.model.decision_function(X_scaled)[0]
        
        # Map prediction: 1 (inlier) -> BENIGN, -1 (outlier) -> ATTACK
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
        
        # Convert to DataFrame
        df = pd.DataFrame(flows)
        
        # Preprocess
        df_processed = self.preprocessor.preprocess(df)
        
        # Scale features
        X_scaled = self.scaler.transform(df_processed)
        
        # Predict
        predictions = self.model.predict(X_scaled)
        decision_scores = self.model.decision_function(X_scaled)
        
        # Create results
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
        
        # Add additional parameters if available
        if 'shrinking' in self.config:
            info['shrinking'] = self.config['shrinking']
        if 'cache_size' in self.config:
            info['cache_size'] = self.config['cache_size']
        if 'max_iter' in self.config:
            info['max_iter'] = self.config['max_iter']
        
        # Add performance metrics if available
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

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize model manager
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
    """
    Predict if a single network flow is benign or an attack
    """
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
    """
    Predict multiple network flows at once
    """
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
    """
    Get the list of features expected by the model
    """
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
    """
    Get information about the loaded model
    """
    if not model_manager.model_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    return model_manager.get_model_info()


if __name__ == "__main__":
    # Run the API server
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Set to False in production
        log_level="info"
    )
