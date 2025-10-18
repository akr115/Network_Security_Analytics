# OCSVM FastAPI Backend - Setup Summary

## Created Files

All files have been created in the `backend_model/` directory:

### Core Application Files

1. **main.py** - FastAPI application
   - Model loading and management
   - Data preprocessing (matching trainer.py pipeline)
   - Prediction endpoints (single and batch)
   - Health checks and model info endpoints
   - CORS configuration
   - Automatic model loading on startup

2. **requirements.txt** - Python dependencies
   - FastAPI and Uvicorn
   - Scikit-learn (ML library)
   - Pandas and NumPy (data processing)
   - Pydantic (data validation)
   - Requests (for health checks)

### Docker Files

3. **Dockerfile** - Container configuration
   - Python 3.11 slim base image
   - Multi-stage optimization
   - Non-root user for security
   - Health check configuration
   - Environment variables setup
   - Model files copied from Train/out directory

4. **docker-compose.yml** - Docker Compose configuration
   - Service definition
   - Port mapping (8000:8000)
   - Health checks
   - Network configuration
   - Restart policy

5. **.dockerignore** - Docker build exclusions
   - Python cache files
   - Development files
   - Documentation
   - Unnecessary data files

### Documentation & Scripts

6. **README.md** - Comprehensive documentation
   - Quick start guide
   - API endpoint documentation
   - Usage examples (cURL and Python)
   - Docker deployment instructions
   - Troubleshooting guide
   - Architecture overview

7. **test_api.py** - API test suite
   - Health check tests
   - Feature retrieval tests
   - Single prediction tests
   - Batch prediction tests
   - Comprehensive test runner

8. **build_and_run.sh** - Automated build and deployment script
   - Docker image building
   - Container management
   - Health verification
   - User-friendly output

9. **run_local.sh** - Local development script
   - Virtual environment setup
   - Dependency installation
   - Local server startup

## Model Configuration

The API uses the pre-trained model from:
```
../Train/out/ocsvm_model_20250928_215630/
```

This model includes:
- `ocsvm_model.pkl` - Trained One-Class SVM
- `feature_scaler.pkl` - StandardScaler for preprocessing
- `feature_names.pkl` - List of feature names (78 features)
- `config.pkl` - Model configuration
- `metrics.pkl` - Performance metrics

## Quick Start

### Option 1: Docker (Recommended)

From the project root directory:

```bash
cd backend_model
chmod +x build_and_run.sh
./build_and_run.sh
```

Or manually:
```bash
# From Network_Security_Analytics directory
docker build -f backend_model/Dockerfile -t ocsvm-api:latest .
docker run -d --name ocsvm-api -p 8000:8000 -e MODEL_DIR=/app/model ocsvm-api:latest
```

Or with Docker Compose:
```bash
cd backend_model
docker-compose up -d
```

### Option 2: Local Development

```bash
cd backend_model
chmod +x run_local.sh
./run_local.sh
```

Or manually:
```bash
cd backend_model
pip install -r requirements.txt
python main.py
```

## API Endpoints

### Core Endpoints

- `GET /` - Root endpoint with API info
- `GET /health` - Health check and model status
- `GET /docs` - Interactive API documentation (Swagger UI)
- `GET /redoc` - Alternative API documentation

### Model Endpoints

- `GET /model/info` - Get model information and parameters
- `GET /features` - Get list of expected features

### Prediction Endpoints

- `POST /predict` - Single flow prediction
  ```json
  {
    "data": {
      "Destination Port": 80,
      "Flow Duration": 120000,
      ...
    }
  }
  ```

- `POST /predict/batch` - Batch prediction
  ```json
  {
    "flows": [
      {"Destination Port": 80, ...},
      {"Destination Port": 443, ...}
    ]
  }
  ```

## Testing

Test the API with the provided test script:

```bash
# Ensure API is running first
python test_api.py
```

Or test individual endpoints:

```bash
# Health check
curl http://localhost:8000/health

# Get features
curl http://localhost:8000/features

# Single prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"data": {"Destination Port": 80, "Flow Duration": 120000, ...}}'
```

## Data Preprocessing Pipeline

The API implements the same preprocessing pipeline as the training script:

1. **Column Name Cleaning** - Strip whitespace
2. **Missing Features** - Add missing features with value 0
3. **Feature Ordering** - Ensure correct feature order
4. **Infinite Value Handling** - Replace inf/-inf with NaN
5. **NaN Handling** - Fill with 0
6. **Type Conversion** - Ensure numeric types
7. **Feature Scaling** - StandardScaler transformation

## Model Performance

Based on model summary (ocsvm_model_20250928_215630):
- **Accuracy**: 57.98%
- **Precision**: 63.81%
- **Recall**: 54.84%
- **F1 Score**: 58.98%

Model Parameters:
- Kernel: RBF
- Gamma: scale
- Nu: 0.05
- Training Mode: Full dataset

## Security Features

- Non-root user in Docker container
- Input validation using Pydantic
- CORS middleware (configure for production)
- Health check monitoring
- Error handling and HTTP status codes

## Next Steps

1. **Test the API**: Run `test_api.py` to verify all endpoints work
2. **Configure CORS**: Update allowed origins in `main.py` for production
3. **Add Authentication**: Implement API keys or OAuth if needed
4. **Set up Monitoring**: Add logging and metrics collection
5. **Deploy**: Use the Docker image for deployment to cloud services
6. **Scale**: Use Kubernetes or Docker Swarm for high availability

## Troubleshooting

### Model not loading
- Check model path is correct
- Verify all pickle files exist
- Check file permissions

### Port already in use
```bash
lsof -ti:8000 | xargs kill -9
```

### Docker container not starting
```bash
docker logs ocsvm-api
```

### API returning errors
- Check logs for detailed error messages
- Verify input data format matches expected features
- Ensure all required features are provided

## Support

For issues or questions:
1. Check the README.md for detailed documentation
2. Review the logs: `docker logs ocsvm-api`
3. Test endpoints with the test script: `python test_api.py`
4. Verify model files are present and accessible
