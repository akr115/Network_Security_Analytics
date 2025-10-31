# Network Security Analytics Platform

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104.1-009688.svg?logo=fastapi)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/docker-enabled-2496ED.svg?logo=docker)](https://www.docker.com/)
[![Elastic](https://img.shields.io/badge/Elastic-Stack-005571.svg?logo=elastic)](https://www.elastic.co/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A **comprehensive, production-ready network security analytics platform** that combines network traffic analysis, machine learning-based intrusion detection, and real-time threat monitoring. This system leverages **One-Class SVM (OCSVM)** for anomaly-based intrusion detection, integrated with the **Elastic Stack** for log management and visualization.

---

##  Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Key Features](#key-features)
- [Components](#components)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Data Flow](#data-flow)
- [API Documentation](#api-documentation)
- [Training Your Own Model](#training-your-own-model)
- [Configuration](#configuration)
- [Monitoring & Visualization](#monitoring--visualization)
- [Performance](#performance)
- [Security Considerations](#security-considerations)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

---

##  Overview

The **Network Security Analytics Platform** is an end-to-end solution for detecting network intrusions using machine learning. It processes network traffic in real-time, extracts flow features, and classifies traffic as **BENIGN** or **ATTACK** using a trained One-Class SVM model.

### What Makes This Special?

-  **Anomaly Detection**: Learns only from benign traffic, can detect unknown attacks
-  **Real-Time Processing**: Processes network flows as they occur
-  **Production-Ready**: Dockerized, scalable, with health checks
-  **Paper-Compliant**: Follows academic research methodology
-  **Comprehensive Pipeline**: From PCAP capture to prediction and logging
-  **Modular Design**: Each component can be used independently
-  **Visualization Ready**: Integrates with Kibana for dashboards

### Use Cases

- **Network Security Monitoring**: Detect malicious traffic in real-time
- **Intrusion Detection System (IDS)**: Identify various attack types
- **Security Research**: Analyze network behavior patterns
- **Threat Hunting**: Investigate suspicious network activities
- **Compliance**: Monitor and log network security events

---

##  System Architecture

```

                         NETWORK TRAFFIC CAPTURE                         
                    (PCAP files or Live Interface)                       

                                 
                                 

                          CICFLOWMETER                                   
   Extracts 82 flow features from network packets                      
   Converts PCAP  CSV with flow statistics                            
   Supports real-time and offline processing                           

                                 
                                  CSV Files

                            FILEBEAT                                     
   Monitors CSV output directory                                       
   Parses CSV into structured data                                     
   Forwards to Logstash                                                

                                 
                                  Beats Protocol

                            LOGSTASH                                     
   Receives flow data from Filebeat                                    
   Validates 82-field CSV format                                       
   Forwards to ML API for prediction                                   
   (Optional) Sends to Elasticsearch                                   

                                 
                                  HTTP POST

                        OCSVM PREDICTION API                             
   FastAPI-based REST service                                          
   Loads pre-trained OCSVM model                                       
   Classifies: BENIGN or ATTACK                                        
   Returns confidence scores                                           

                                 
                                  Prediction Results

                    ELASTICSEARCH (Optional)                             
   Stores predictions and flow data                                    
   Enables historical analysis                                         
   Powers Kibana dashboards                                            

                                 
                                 

                         KIBANA (Optional)                               
   Visualizes attack patterns                                          
   Real-time dashboards                                                
   Alert management                                                    

```

---

##  Key Features

### Machine Learning

- **One-Class SVM**: Anomaly detection trained on CICIDS2017 dataset
- **78 Network Features**: Comprehensive flow-based features
- **High Precision**: ~91-92% precision on attack detection
- **Unknown Attack Detection**: Generalizes to unseen attack types
- **Confidence Scoring**: Provides decision scores for predictions

### Network Analysis

- **CICFlowMeter Integration**: Industry-standard flow feature extraction
- **82-Field CSV Format**: Compatible with CICIDS2017 dataset
- **Real-Time Processing**: Live capture from network interfaces
- **Batch Processing**: Process multiple PCAP files
- **Multiple Protocols**: TCP, UDP, ICMP support

### Infrastructure

- **Docker Compose**: One-command deployment
- **Microservices Architecture**: Independent, scalable components
- **Health Checks**: Built-in monitoring endpoints
- **Auto-Restart**: Resilient to failures
- **Volume Persistence**: Data survives container restarts

### API & Integration

- **RESTful API**: FastAPI with auto-generated docs
- **Multiple Input Formats**: JSON, CSV string, CSV array
- **Batch Predictions**: Process multiple flows at once
- **OpenAPI Specification**: Interactive documentation
- **CORS Enabled**: Web application integration

---

##  Components

### 1. **CICFlowMeter** (`integrations/cicflowmeter/`)

Network flow feature extractor that converts packet captures to flow statistics.

**Key Capabilities**:
- Extracts 82 flow features from PCAP files
- Real-time capture from network interfaces
- Batch processing of multiple PCAPs
- CSV output with standardized format

**Technology**: Python, Scapy, Custom Flow Session

 [CICFlowMeter Documentation](integrations/cicflowmeter/README.md)

### 2. **Filebeat** (`integrations/filebeat/`)

Lightweight shipper for forwarding and centralizing log data.

**Key Capabilities**:
- Monitors CSV output directory
- Decodes CSV into structured events
- Forwards to Logstash via Beats protocol
- Automatic field parsing

**Technology**: Elastic Beats

### 3. **Logstash** (`integrations/logstash/`)

Data processing pipeline that ingests, transforms, and routes data.

**Key Capabilities**:
- Receives data from Filebeat
- Validates 82-field CSV format
- HTTP output to ML API
- Optional Elasticsearch integration
- Field filtering and transformation

**Technology**: Elastic Logstash

### 4. **OCSVM Prediction API** (`backend_model/`)

FastAPI-based REST service for real-time intrusion detection.

**Key Capabilities**:
- Pre-trained OCSVM model
- Single and batch predictions
- CSV, JSON input formats
- Health monitoring
- Model introspection endpoints

**Technology**: FastAPI, scikit-learn, pandas, NumPy

 [API Documentation](backend_model/README.md)

### 5. **Training Pipeline** (`Train/`)

Complete pipeline for training custom OCSVM models on CICIDS2017 dataset.

**Key Capabilities**:
- Paper-compliant methodology
- Multiple training modes
- Hyperparameter tuning
- Performance visualization
- DVC integration

**Technology**: scikit-learn, pandas, matplotlib, DVC

 [Training Documentation](Train/README.md)

### 6. **Suricata IDS** (`integrations/suricata/`) *(Optional)*

Open-source intrusion detection system for additional threat detection.

**Key Capabilities**:
- Rule-based detection
- Protocol analysis
- Alert generation
- Log output for Elasticsearch

**Technology**: Suricata IDS

---

##  Quick Start

### Prerequisites

- **Docker**: 20.10+ and Docker Compose
- **System**: 8GB RAM minimum, 16GB recommended
- **Storage**: ~15GB for datasets and models
- **Network**: Internet access for pulling images

### 1. Clone the Repository

```bash
git clone https://github.com/akr115/Network_Security_Analytics.git
cd Network_Security_Analytics
```

### 2. Pull Training Data (Optional - for custom training)

If you have DVC access:

```bash
cd Train
dvc pull
cd ..
```

Otherwise, download CICIDS2017 dataset manually (see [Training Documentation](Train/README.md)).

### 3. Start the Platform

```bash
# Start all services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f ocsvm-api
```

### 4. Verify Services

```bash
# Check API health
curl http://localhost:8000/health

# View API documentation
open http://localhost:8000/docs
```

### 5. Process a PCAP File

```bash
# Place your PCAP in the integrations/pcaps directory
cp /path/to/your/capture.pcap integrations/pcaps/

# CICFlowMeter will process it automatically
# Check flows output
ls -lh integrations/flows/

# Filebeat will forward to Logstash  API
# Check API logs for predictions
docker-compose logs ocsvm-api
```

---

##  Installation

### Full Installation (All Components)

```bash
# 1. Clone repository
git clone https://github.com/akr115/Network_Security_Analytics.git
cd Network_Security_Analytics

# 2. Set up training data (choose one option)
# Option A: Using DVC (if you have access)
cd Train && dvc pull && cd ..

# Option B: Manual download
# Download CICIDS2017 from https://www.unb.ca/cic/datasets/ids-2017.html
# Extract CSVs to Train/data/

# 3. Build and start services
docker-compose up --build -d

# 4. Verify everything is running
docker-compose ps
```

### API-Only Installation

If you only want the prediction API:

```bash
cd backend_model
docker-compose up -d
```

### Local Development (Without Docker)

```bash
# API
cd backend_model
pip install -r requirements.txt
python main.py

# Training
cd Train
pip install scikit-learn numpy pandas matplotlib seaborn
python trainer_paper_replicate.py
```

---

##  Usage

### Processing PCAP Files

#### Method 1: Using Docker Compose (Automated)

```bash
# 1. Place PCAP files in integrations/pcaps/
cp network_capture.pcap integrations/pcaps/

# 2. Start CICFlowMeter (if not already running)
docker-compose up -d cicflowmeter

# 3. Monitor the flow
docker-compose logs -f cicflowmeter filebeat logstash ocsvm-api
```

#### Method 2: Manual CICFlowMeter

```bash
# Inside the cicflowmeter container
docker-compose run cicflowmeter cicflowmeter \
  -f /pcaps/capture.pcap \
  -c /flows/output.csv
```

#### Method 3: Real-Time Capture

```bash
# Capture from network interface (requires root)
docker run --rm --network host \
  -v $(pwd)/integrations/flows:/flows \
  cicflowmeter \
  cicflowmeter -i eth0 -u http://localhost:8000/predict/csv
```

### Making Predictions via API

#### Single Flow Prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "data": {
      "Destination Port": 80,
      "Flow Duration": 120000,
      "Total Fwd Packets": 10,
      "Total Backward Packets": 8,
      "Total Length of Fwd Packets": 5000,
      "Total Length of Bwd Packets": 3000
    }
  }'
```

#### CSV Format Prediction

```bash
curl -X POST http://localhost:8000/predict/csv \
  -H "Content-Type: application/json" \
  -d '{
    "csv": "192.168.1.1,10.0.0.1,443,80,6,2024-01-01,..."
  }'
```

#### Batch Predictions

```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "flows": [
      {"Destination Port": 80, "Flow Duration": 120000, ...},
      {"Destination Port": 443, "Flow Duration": 90000, ...}
    ]
  }'
```

### Training a New Model

```bash
cd Train

# Full training (recommended)
python trainer_paper_replicate.py

# Quick test
python trainer_paper_replicate.py --small-sample

# With hyperparameter tuning
python trainer_paper_replicate.py --hyperparam-tune

# Deploy trained model
cp -r out/ocsvm_model_YYYYMMDD_HHMMSS/* ../backend_model/production_model/
docker-compose restart ocsvm-api
```

---

##  Project Structure

```
Network_Security_Analytics/
 README.md                    # This file
 docker-compose.yml           # Main orchestration file

 backend_model/               # ML API Service
    main.py                 # FastAPI application
    Dockerfile              # API container image
    docker-compose.yml      # Standalone API compose
    requirements.txt        # Python dependencies
    test_api.py            # API test script
    production_model/       # Pre-trained model artifacts
       ocsvm_model.pkl
       feature_scaler.pkl
       feature_names.pkl
       config.pkl
    README.md              # API documentation

 Train/                      # Model Training Pipeline
    trainer_paper_replicate.py  # Main training script
    trainer.py             # Alternative trainer (deprecated)
    data_exloration.ipynb  # Jupyter notebook for EDA
    data/                  # CICIDS2017 dataset
       *.csv             # Training data files
       *.csv.dvc         # DVC tracking files
    out/                   # Training outputs (runtime)
       ocsvm_model_*/    # Timestamped model directories
    out.dvc               # DVC tracking for outputs
    README.md             # Training documentation

 integrations/              # Integration Services
    cicflowmeter/         # Flow feature extractor
       Dockerfile
       src/cicflowmeter/
       README.md
   
    filebeat/             # Log shipper
       filebeat.yml     # Filebeat configuration
       modules/
   
    logstash/             # Data pipeline
       pipeline/
           logstash.conf
   
    suricata/             # IDS (optional)
       Dockerfile
       suricata.yaml
       rules/
       logs/
   
    pcaps/                # Input PCAP files
       sample.pcap
   
    flows/                # Output CSV files (runtime)

 .dvc/                     # DVC configuration
 .dvcignore               # DVC ignore patterns
 .gitignore               # Git ignore patterns
 .gitattributes           # Git LFS configuration
```

---

##  Data Flow

### End-to-End Flow Processing

```
1. NETWORK TRAFFIC CAPTURE
   
   [PCAP File] or [Live Interface]
   
2. FEATURE EXTRACTION (CICFlowMeter)
   
   [CSV with 82 fields per flow]
   
3. LOG SHIPPING (Filebeat)
   
   [Structured event with decoded CSV]
   
4. DATA PROCESSING (Logstash)
   
    Validate 82 fields
    Extract CSV array
    Clean metadata
   
5. ML PREDICTION (FastAPI + OCSVM)
   
    Convert CSV  feature dict (78 features)
    Apply StandardScaler
    OCSVM prediction
   
6. RESULT
   
   {"prediction": "benign"} or {"prediction": "attack"}
   
7. (OPTIONAL) STORAGE & VISUALIZATION
   
   [Elasticsearch]  [Kibana Dashboard]
```

### Feature Transformation

```
82 CSV Fields  78 ML Features

Dropped in transformation:
- src_ip, dst_ip       (PII, context-specific)
- src_port             (High cardinality)
- protocol             (Encoded in other features)
- timestamp            (Temporal, not used)

Example CSV (82 fields):
192.168.1.1,10.0.0.1,443,80,6,2024-01-01,...,0,0,0

 Mapping

Feature Dict (78 features):
{
  "Destination Port": 80,
  "Flow Duration": 120000,
  "Total Fwd Packets": 10,
  ...
}

 Scaling

Scaled Features (mean=0, std=1):
array([0.23, -1.45, 0.87, ...])

 Prediction

OCSVM Decision: -1 (attack) or 1 (benign)
```

---

##  API Documentation

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Root endpoint, API information |
| `/health` | GET | Health check, model status |
| `/predict` | POST | Single flow prediction (JSON) |
| `/predict/batch` | POST | Multiple flow predictions |
| `/predict/csv` | POST | Prediction from CSV format |
| `/features` | GET | List expected features |
| `/model/info` | GET | Model metadata and metrics |
| `/docs` | GET | Interactive API documentation (Swagger UI) |
| `/redoc` | GET | Alternative API documentation (ReDoc) |

### Interactive Documentation

Once the API is running, visit:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Example Requests

See [backend_model/README.md](backend_model/README.md) for detailed API examples.

---

##  Training Your Own Model

### Quick Training

```bash
cd Train

# Download dataset (if not using DVC)
# Place CICIDS2017 CSV files in data/

# Train with all data (recommended)
python trainer_paper_replicate.py

# Results will be in out/ocsvm_model_YYYYMMDD_HHMMSS/
```

### Training Modes

```bash
# Small sample (fast testing)
python trainer_paper_replicate.py --small-sample

# Medium sample (balanced)
python trainer_paper_replicate.py --medium-sample

# Large sample (near-full accuracy)
python trainer_paper_replicate.py --large-sample

# Full dataset (production)
python trainer_paper_replicate.py

# With hyperparameter tuning
python trainer_paper_replicate.py --hyperparam-tune
```

### Deploying Trained Model

```bash
# 1. Train model
cd Train
python trainer_paper_replicate.py

# 2. Copy to production
cp -r out/ocsvm_model_20251031_120000/* ../backend_model/production_model/

# 3. Rebuild API container
cd ../backend_model
docker-compose up --build -d

# 4. Verify
curl http://localhost:8000/health
```

See [Train/README.md](Train/README.md) for comprehensive training documentation.

---

##  Configuration

### Docker Compose Services

Edit `docker-compose.yml` to enable/disable services:

```yaml
# Enable Elasticsearch & Kibana (commented by default)
# Uncomment the elasticsearch and kibana sections

# Adjust resource limits
services:
  ocsvm-api:
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G
```

### Environment Variables

**OCSVM API** (`backend_model/`):
```bash
MODEL_DIR=/app/model        # Model directory path
DEBUG=false                 # Enable debug logging
PYTHONUNBUFFERED=1         # Python output buffering
```

**CICFlowMeter**:
```bash
# Set in docker-compose.yml or runtime
-v ./custom/pcaps:/pcaps    # Custom PCAP directory
-v ./custom/flows:/flows    # Custom output directory
```

### Logstash Configuration

Edit `integrations/logstash/pipeline/logstash.conf`:

```ruby
# Change API endpoint
output {
  http {
    url => "http://custom-api:8000/predict/csv"
  }
}

# Add Elasticsearch output
output {
  elasticsearch {
    hosts => ["elasticsearch:9200"]
    index => "network-flows-%{+YYYY.MM.dd}"
  }
}
```

### Filebeat Configuration

Edit `integrations/filebeat/filebeat.yml`:

```yaml
# Change monitored paths
filebeat.inputs:
  - type: filestream
    paths:
      - /custom/path/*.csv

# Adjust parsing
processors:
  - decode_csv_fields:
      separator: ";"  # Change delimiter
```

---

##  Monitoring & Visualization

### Built-in Health Checks

```bash
# API health
curl http://localhost:8000/health

# Docker health status
docker-compose ps
```

### Docker Logs

```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f ocsvm-api

# Last 100 lines
docker-compose logs --tail=100 ocsvm-api
```

### Elasticsearch + Kibana (Optional)

1. **Enable in docker-compose.yml**:
   ```bash
   # Uncomment elasticsearch and kibana sections
   docker-compose up -d elasticsearch kibana
   ```

2. **Access Kibana**:
   - URL: http://localhost:5601
   - Create index pattern: `network-flows-*`
   - Build dashboards for attack visualization

3. **Sample Kibana Visualizations**:
   - Attack types distribution (pie chart)
   - Timeline of attacks (area chart)
   - Top attacked ports (bar chart)
   - Attack vs Benign ratio (metric)

### Metrics & Performance

Monitor using Docker stats:

```bash
# Resource usage
docker stats

# Container-specific
docker stats ocsvm-api
```

---

##  Performance

### Throughput

- **API Latency**: <5ms per prediction (single flow)
- **Batch Processing**: ~1000 flows/second
- **CICFlowMeter**: ~10,000 packets/second
- **End-to-End**: ~500-1000 flows/second (full pipeline)

### Resource Requirements

| Component | CPU | RAM | Storage |
|-----------|-----|-----|---------|
| OCSVM API | 1-2 cores | 2-4 GB | 500 MB |
| CICFlowMeter | 2-4 cores | 4-8 GB | 1 GB |
| Filebeat | 0.5 cores | 512 MB | 100 MB |
| Logstash | 1-2 cores | 2-4 GB | 500 MB |
| Elasticsearch | 2-4 cores | 4-8 GB | 10+ GB |
| **Total (without Elastic)** | **4-8 cores** | **8-16 GB** | **2 GB** |
| **Total (with Elastic)** | **8-16 cores** | **16-32 GB** | **15+ GB** |

### Scaling

**Horizontal Scaling**:
```yaml
# docker-compose.yml
services:
  ocsvm-api:
    deploy:
      replicas: 3  # Run 3 instances
```

**Load Balancing**:
- Add nginx/traefik reverse proxy
- Distribute requests across API replicas
- Use Docker Swarm or Kubernetes for orchestration

---

##  Security Considerations

### Production Deployment Checklist

- [ ] **Change Default Ports**: Avoid exposing default ports externally
- [ ] **Enable Authentication**: Add API keys or OAuth to endpoints
- [ ] **Use HTTPS**: Deploy behind TLS-terminating reverse proxy
- [ ] **Network Segmentation**: Use Docker networks appropriately
- [ ] **Resource Limits**: Set CPU/memory limits in docker-compose
- [ ] **Update Dependencies**: Regularly update base images and packages
- [ ] **Secrets Management**: Use Docker secrets or environment files
- [ ] **Logging**: Enable audit logging for all API requests
- [ ] **Monitoring**: Set up alerts for anomalies
- [ ] **Backup**: Regular backups of models and configurations

### API Security

```python
# Add authentication to main.py
from fastapi.security import APIKeyHeader

API_KEY_HEADER = APIKeyHeader(name="X-API-Key")

@app.post("/predict")
async def predict(
    input_data: NetworkFlowInput,
    api_key: str = Depends(API_KEY_HEADER)
):
    if api_key != os.getenv("API_KEY"):
        raise HTTPException(401, "Invalid API key")
    # ... prediction logic
```

### Network Isolation

```yaml
# docker-compose.yml
networks:
  frontend:  # Public-facing
  backend:   # Internal services only
  
services:
  ocsvm-api:
    networks:
      - backend
  nginx:
    networks:
      - frontend
      - backend
```

---

##  Troubleshooting

### Common Issues

#### 1. API Container Fails to Start

**Symptoms**: `docker-compose ps` shows `ocsvm-api` as `Exit 1`

**Solutions**:
```bash
# Check logs
docker-compose logs ocsvm-api

# Common causes:
# - Missing model files
ls -la backend_model/production_model/

# - Port conflict
lsof -ti:8000 | xargs kill -9

# Rebuild
docker-compose up --build -d ocsvm-api
```

#### 2. No Predictions Appearing

**Symptoms**: PCAP processed but no predictions in logs

**Debug Steps**:
```bash
# 1. Check CICFlowMeter output
ls -lh integrations/flows/

# 2. Check Filebeat is reading
docker-compose logs filebeat | grep -i csv

# 3. Check Logstash is receiving
docker-compose logs logstash | grep -i csv

# 4. Check API is responding
curl http://localhost:8000/health

# 5. Test API directly
curl -X POST http://localhost:8000/predict/csv \
  -H "Content-Type: application/json" \
  -d '{"csv": "..."}'
```

#### 3. Memory Issues

**Symptoms**: Container killed, out of memory errors

**Solutions**:
```bash
# Increase Docker memory limit
# Docker Desktop > Settings > Resources > Memory: 8GB+

# Reduce cache size in training
# Edit Train/trainer_paper_replicate.py
# OCSVM_PARAMS['cache_size'] = 5000
```

#### 4. CICFlowMeter Not Processing PCAP

**Solutions**:
```bash
# Check PCAP is in correct location
ls -lh integrations/pcaps/

# Run manually
docker-compose run cicflowmeter \
  cicflowmeter -f /pcaps/your_file.pcap -c /flows/output.csv

# Check permissions
chmod 644 integrations/pcaps/*.pcap
```

#### 5. CSV Format Mismatch

**Symptoms**: "CSV must have 80 or 82 fields" error

**Solutions**:
```bash
# Check CSV field count
head -1 integrations/flows/output.csv | tr ',' '\n' | wc -l

# CICFlowMeter should output 82 fields
# If different, check CICFlowMeter version/configuration
```

### Getting Help

1. **Check Documentation**: Component-specific READMEs
2. **Enable Debug Logging**: Set `DEBUG=true` in environment
3. **Review Logs**: `docker-compose logs -f [service]`
4. **GitHub Issues**: https://github.com/akr115/Network_Security_Analytics/issues
5. **Stack Overflow**: Tag with `network-security` and `ocsvm`

---

##  Contributing

Contributions are welcome! Please follow these guidelines:

### Development Setup

```bash
# Fork and clone
git clone https://github.com/YOUR_USERNAME/Network_Security_Analytics.git
cd Network_Security_Analytics

# Create feature branch
git checkout -b feature/your-feature

# Make changes and test
docker-compose up --build

# Commit with meaningful messages
git commit -m "Add feature: description"

# Push and create PR
git push origin feature/your-feature
```

### Code Standards

- **Python**: Follow PEP 8, use type hints
- **Docker**: Multi-stage builds, minimize layers
- **Documentation**: Update READMEs for changes
- **Testing**: Add tests for new features
- **Commits**: Conventional commit messages

### Pull Request Process

1. Update documentation
2. Add/update tests
3. Ensure all tests pass
4. Update CHANGELOG if applicable
5. Request review from maintainers

---

##  License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### Third-Party Licenses

- **CICFlowMeter**: [Original License](integrations/cicflowmeter/LICENSE)
- **Elastic Stack**: Apache 2.0
- **FastAPI**: MIT License
- **scikit-learn**: BSD-3-Clause

---

##  Acknowledgments

### Research & Datasets

- **CICIDS2017 Dataset**: Canadian Institute for Cybersecurity, University of New Brunswick
- **Paper**: "Robust Anomaly Detection in Network Traffic: Evaluating Machine Learning Models on CICIDS2017"

### Technologies

- **CICFlowMeter**: [hieulw/cicflowmeter](https://github.com/hieulw/cicflowmeter)
- **Elastic Stack**: Elasticsearch, Logstash, Kibana, Beats
- **FastAPI**: Modern Python web framework
- **scikit-learn**: Machine learning library
- **Docker**: Containerization platform

### Contributors

- **akr115**: Project creator and maintainer
- Community contributors (see GitHub contributors page)

---

##  Additional Resources

### Documentation

- [Backend API Documentation](backend_model/README.md)
- [Training Pipeline Documentation](Train/README.md)
- [CICFlowMeter Documentation](integrations/cicflowmeter/README.md)

### External Links

- [CICIDS2017 Dataset](https://www.unb.ca/cic/datasets/ids-2017.html)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [scikit-learn OCSVM](https://scikit-learn.org/stable/modules/generated/sklearn.svm.OneClassSVM.html)
- [Elastic Stack Documentation](https://www.elastic.co/guide/index.html)
- [Docker Compose Documentation](https://docs.docker.com/compose/)

### Related Projects

- [Awesome Network Security](https://github.com/hslatman/awesome-threat-intelligence)
- [ML for Cybersecurity](https://github.com/jivoi/awesome-ml-for-cybersecurity)

---

##  Contact & Support

- **GitHub Issues**: [Report bugs or request features](https://github.com/akr115/Network_Security_Analytics/issues)
- **Discussions**: [Ask questions](https://github.com/akr115/Network_Security_Analytics/discussions)
- **Email**: [Your contact email]

---

** If you find this project useful, please consider giving it a star on GitHub!**

---

*Last Updated: October 31, 2025*
