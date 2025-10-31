# OCSVM Network Intrusion Detection API

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104.1-009688.svg?logo=fastapi)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/docker-enabled-2496ED.svg?logo=docker)](https://www.docker.com/)

A production-ready REST API for **real-time network intrusion detection** using a One-Class Support Vector Machine (OCSVM) anomaly detection model. This service provides high-performance classification of network traffic flows, distinguishing between benign and malicious activity based on the CICIDS2017 dataset features.

---

##  Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Model Details](#model-details)
- [Installation](#installation)
- [API Documentation](#api-documentation)
- [Usage Examples](#usage-examples)
- [Data Format Specifications](#data-format-specifications)
- [Deployment](#deployment)
- [Performance & Monitoring](#performance--monitoring)
- [Development](#development)
- [Troubleshooting](#troubleshooting)
- [Security Considerations](#security-considerations)
- [License](#license)

---

##  Overview

This FastAPI-based microservice serves a pre-trained One-Class SVM model for detecting network intrusions in real-time. The model is trained on the CICIDS2017 dataset and implements a paper-compliant anomaly detection approach that identifies malicious network traffic patterns.

### Key Capabilities

-  **Real-time Classification**: Sub-millisecond prediction latency for single flows
-  **Batch Processing**: Efficient bulk predictions for multiple flows
-  **Multi-Format Input**: Supports JSON, CSV string, and CSV array formats
-  **Auto-Preprocessing**: Automatic feature engineering and scaling
-  **Production-Ready**: Docker containerization, health checks, and monitoring
-  **Interactive Documentation**: Auto-generated OpenAPI/Swagger UI
-  **Comprehensive Validation**: Input validation with detailed error messages

---

##  Features

### Core Functionality

- **Single Flow Prediction**: Classify individual network flows with confidence scores
- **Batch Prediction**: Process multiple flows simultaneously with aggregate statistics
- **CSV Integration**: Direct CSV input support for integration with traffic analyzers
- **Feature Introspection**: Query model's expected features and configuration
- **Model Metadata**: Access detailed model information and performance metrics

### Technical Features

- **Smart Preprocessing**: Automatic handling of missing features, infinite values, and type conversion
- **Flexible Input Formats**: JSON dictionaries, CSV strings, or CSV arrays
- **CORS Enabled**: Cross-origin resource sharing for web integrations
- **Health Monitoring**: Endpoint for Kubernetes/Docker health checks
- **Debug Mode**: Comprehensive logging for development and troubleshooting
- **Non-root Execution**: Security-hardened Docker container
- **Graceful Degradation**: Informative errors when model is unavailable

## Quick Start

### Local Development

1. **Install Dependencies**
   ```bash
   cd backend_model
   pip install -r requirements.txt
   ```

2. **Run the API**
   ```bash
   python main.py
   ```
   
   Or using uvicorn directly:
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000 --reload
   ```

3. **Access the API**
   - API: http://localhost:8000
   - Interactive Docs: http://localhost:8000/docs
   - Alternative Docs: http://localhost:8000/redoc

### Docker Deployment

1. **Build the Docker Image**
   ```bash
   # From the Network_Security_Analytics directory
   cd backend_model
   docker build -t ocsvm-api:latest .
   ```

2. **Run the Container**
   ```bash
   docker run -d \
     --name ocsvm-api \
     -p 8000:8000 \
     ocsvm-api:latest
   ```

3. **Check Container Status**
   ```bash
   docker ps
   docker logs ocsvm-api
   ```

4. **Stop the Container**
   ```bash
   docker stop ocsvm-api
   docker rm ocsvm-api
   ```

## API Endpoints

### Health Check
```bash
GET /health
```
Check if the API and model are loaded properly.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_info": {
    "model_type": "One-Class SVM",
    "kernel": "rbf",
    "n_features": 78
  }
}
```

### Single Prediction
```bash
POST /predict
```
Classify a single network flow.

**Request:**
```json
{
  "data": {
    "Destination Port": 80,
    "Flow Duration": 120000,
    "Total Fwd Packets": 10,
    "Total Backward Packets": 8,
    "Total Length of Fwd Packets": 5000,
    "Total Length of Bwd Packets": 3000,
    ...
  }
}
```

**Response:**
```json
{
  "prediction": "BENIGN",
  "confidence": 0.234,
  "is_anomaly": false
}
```

### Batch Prediction
```bash
POST /predict/batch
```
Classify multiple network flows at once.

**Request:**
```json
{
  "flows": [
    {
      "Destination Port": 80,
      "Flow Duration": 120000,
      ...
    },
    {
      "Destination Port": 443,
      "Flow Duration": 90000,
      ...
    }
  ]
}
```

**Response:**
```json
{
  "predictions": [
    {
      "prediction": "BENIGN",
      "confidence": 0.234,
      "is_anomaly": false
    },
    {
      "prediction": "ATTACK",
      "confidence": -0.456,
      "is_anomaly": true
    }
  ],
  "total_flows": 2,
  "benign_count": 1,
  "attack_count": 1
}
```

### Get Features
```bash
GET /features
```
Get the list of features expected by the model.

### Get Model Info
```bash
GET /model/info
```
Get detailed information about the loaded model.

## Example Usage with cURL

### Health Check
```bash
curl http://localhost:8000/health
```

### Single Prediction
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "data": {
      "Destination Port": 80,
      "Flow Duration": 120000,
      "Total Fwd Packets": 10,
      "Total Backward Packets": 8
    }
  }'
```

## Example Usage with Python

```python
import requests

# API base URL
BASE_URL = "http://localhost:8000"

# Single prediction
flow_data = {
    "data": {
        "Destination Port": 80,
        "Flow Duration": 120000,
        "Total Fwd Packets": 10,
        "Total Backward Packets": 8,
        # ... add all required features
    }
}

response = requests.post(f"{BASE_URL}/predict", json=flow_data)
result = response.json()
print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']}")
print(f"Is Anomaly: {result['is_anomaly']}")

# Batch prediction
batch_data = {
    "flows": [
        {"Destination Port": 80, "Flow Duration": 120000, ...},
        {"Destination Port": 443, "Flow Duration": 90000, ...}
    ]
}

response = requests.post(f"{BASE_URL}/predict/batch", json=batch_data)
result = response.json()
print(f"Total flows: {result['total_flows']}")
print(f"Benign: {result['benign_count']}, Attacks: {result['attack_count']}")
```

## Model Information

- **Model Type**: One-Class SVM (OCSVM)
- **Model Version**: ocsvm_model_20250928_215630
- **Training Mode**: Full dataset
- **Performance**:
  - Accuracy: 57.98%
  - Precision: 63.81%
  - Recall: 54.84%
  - F1 Score: 58.98%

## Architecture

```
backend_model/
 main.py              # FastAPI application
 requirements.txt     # Python dependencies
 Dockerfile          # Docker configuration
 .dockerignore       # Docker build exclusions
 README.md           # This file

The model files are loaded from:
../Train/out/ocsvm_model_20250928_215630/
 ocsvm_model.pkl     # Trained OCSVM model
 feature_scaler.pkl  # StandardScaler for preprocessing
 feature_names.pkl   # List of feature names
 config.pkl          # Model configuration
 model_summary.txt   # Model metadata
```

## Data Preprocessing

The API automatically handles the following preprocessing steps (matching the training pipeline):

1. **Column Name Cleaning**: Strip whitespace from column names
2. **Missing Features**: Add missing features with default value 0
3. **Feature Ordering**: Ensure features are in the same order as training
4. **Infinite Values**: Replace inf/-inf with NaN
5. **NaN Handling**: Fill NaN values with 0
6. **Type Conversion**: Ensure all values are numeric
7. **Scaling**: Apply StandardScaler transformation

## Security Considerations

- The Docker container runs as a non-root user (`apiuser`)
- CORS is configured for all origins (update for production)
- Input validation using Pydantic models
- Health checks for monitoring

## Production Deployment

For production deployment, consider:

1. **Disable reload mode** in uvicorn
2. **Configure CORS** with specific allowed origins
3. **Add authentication** (e.g., API keys, OAuth)
4. **Use HTTPS** with reverse proxy (nginx, traefik)
5. **Set up monitoring** and logging
6. **Configure resource limits** in Docker
7. **Use environment variables** for configuration
8. **Deploy with orchestration** (Docker Compose, Kubernetes)

## Troubleshooting

### Model not loading
- Verify the model path in `main.py` (Config.MODEL_DIR)
- Ensure all model files exist in the directory
- Check file permissions

### Port already in use
```bash
# Find and kill process using port 8000
lsof -ti:8000 | xargs kill -9
```

### Docker build fails
- Ensure you're building from the correct directory
- Check that the model directory exists
- Verify Docker has enough resources

## License

This project is part of the Network Security Analytics system.
