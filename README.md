# Network Security Analytics Platform

A **comprehensive, production-ready network security analytics platform** that combines network traffic analysis, machine learning-based intrusion detection, and real-time threat monitoring. This system leverages **One-Class SVM (OCSVM)** for anomaly-based intrusion detection, integrated with the **Elastic Stack** for log management and visualization.

---

##  Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Components](#components)
- [Quick Start](#quick-start)
- [Usage](#usage)


---

##  Overview

The **Network Security Analytics Platform** is an end-to-end solution for detecting network intrusions using machine learning. It processes network traffic in real-time, extracts flow features, and classifies traffic as **BENIGN** or **ATTACK** using a trained One-Class SVM model.

### What Makes This Special?

-  **Anomaly Detection**: Learns only from benign traffic, can detect unknown attacks
-  **Real-Time Processing**: Processes network flows as they occur
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

##  Components

### 1. **CICFlowMeter** (`integrations/cicflowmeter/`)

Network flow feature extractor that converts packet captures to flow statistics.

**Key Capabilities**:
- Real-time capture from network interfaces
- Batch processing of multiple PCAPs
- CSV output with standardized format

**Technology**: Python, Scapy, Custom Flow Session

 [CICFlowMeter Documentation](integrations/cicflowmeter/README.md)

### 2. **Filebeat** (`integrations/filebeat/`)

Lightweight shipper for forwarding and centralizing log data.

**Key Capabilities**:
- Monitors CSV output directory
- Forwards to Logstash via Beats protocol

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


##  Quick Start

### Prerequisites

- **Docker**
- **git**
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



##  Usage

### Processing PCAP Files

#### Method 1: Using Docker Compose (Automated)

```bash
# 1. Place PCAP files in integrations/pcaps/

# 2. Start CICFlowMeter (if not already running)
docker-compose up -d cicflowmeter

# 3. Monitor the flow
```



---

*Last Updated: October 31, 2025*
