# EDGP AI Model Service

A FastAPI-based service for AI-powered data quality checking, focusing on anomaly detection and duplication identification using advanced machine learning models with local AI capabilities.

## 🌟 Features

- **🤖 AI-Powered Detection**: Advanced anomaly detection using locally cached Hugging Face transformer models
- **🔍 Smart Anomaly Detection**: Identifies outliers using AI embeddings with sklearn fallback (Isolation Forest, DBSCAN)
- **👥 Intelligent Duplication Detection**: Finds semantic duplicates using AI similarity analysis
- **📊 Multiple Input Formats**: Supports JSON payloads and file uploads (CSV, JSON)
- **🚀 RESTful API**: Comprehensive API with automatic documentation
- **⚙️ Configurable Thresholds**: Customizable detection sensitivity for both AI and traditional methods
- **🏥 Health Monitoring**: Built-in health checks and service monitoring
- **📈 Scalable Architecture**: Modular design with AI/sklearn fallback for reliability
- **💾 Local AI Models**: Pre-downloaded models in ./models folder (no internet required)
- **🔄 Automatic Fallback**: Seamlessly switches between AI and traditional ML methods

## 📁 Project Structure

```
edgp-ai-model/
├── src/
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes.py              # API endpoints
│   ├── models/
│   │   ├── __init__.py
│   │   ├── schemas.py             # Pydantic models
│   │   └── anomaly_detector.py    # Enhanced AI models for detection
│   ├── services/
│   │   ├── __init__.py
│   │   └── data_quality_service.py # Business logic
│   ├── utils/
│   │   ├── __init__.py
│   │   └── helpers.py             # Utility functions
│   └── __init__.py
├── config/
│   ├── __init__.py
│   └── settings.py                # Configuration management
├── models/                        # 🤖 Local AI models cache
│   └── models--sentence-transformers--all-MiniLM-L6-v2/
├── examples/                      # 📚 Usage examples and demos
│   ├── ai_usage_example.py        # Comprehensive AI examples
│   ├── check_ai_setup.py          # AI setup verification
│   ├── show_ai_models.py          # Model information script
│   ├── demo.py                    # Basic demo script
│   └── usage_guide.py             # Complete usage guide
├── docs/                          # 📖 Documentation
│   ├── AI_INTEGRATION_GUIDE.md    # AI integration instructions
│   ├── AI_MODEL_USAGE_GUIDE.md    # AI usage examples
│   ├── AI_SETUP_COMPLETE.md       # Setup completion guide
│   ├── MANUAL_AI_SETUP.md         # Manual setup instructions
│   └── TUTORIAL.md                # Step-by-step tutorial
├── tests/
│   ├── test_api.py                # API tests
│   ├── test_models.py             # Model tests
│   ├── sample_data.csv            # Test data
│   └── sample_data.json           # Test data
├── main.py                        # Application entry point
├── requirements.txt               # Dependencies
├── .env                          # Environment variables
├── .gitignore                    # Git ignore file
├── Dockerfile                    # Docker configuration
├── docker-compose.yml            # Docker Compose setup
└── README.md                     # This documentation
```

## 🤖 AI Models (Ready to Use!)

### ✅ **Pre-installed Local AI Models**
Your service includes pre-downloaded AI models for advanced data quality analysis:

- **Primary Model**: `sentence-transformers/all-MiniLM-L6-v2` (87 MB)
- **Location**: `./models/models--sentence-transformers--all-MiniLM-L6-v2/`
- **Status**: ✅ Downloaded and cached locally
- **Capabilities**: 
  - 384-dimensional embeddings for semantic analysis
  - Advanced anomaly detection using similarity patterns
  - Intelligent duplicate detection with semantic understanding
  - Works with mixed data types (numeric, text, categorical)

### 🔧 **Model Information**
```bash
# Check AI model status and test functionality
python examples/show_ai_models.py

# Run comprehensive AI examples
python examples/ai_usage_example.py

# Verify AI setup
python examples/check_ai_setup.py
```

### 🧠 **AI vs Traditional Detection**
- **AI Mode**: Uses transformer embeddings for semantic pattern recognition
- **Fallback Mode**: Uses sklearn (Isolation Forest, DBSCAN) if AI unavailable
- **Automatic Switching**: Service automatically uses best available method
- **No Internet Required**: All models cached locally

## Setup and Installation

### Prerequisites

- Python 3.11+
- pip or conda

### Quick Start (5 minutes)

1. **Clone and navigate to the project**:
   ```bash
   git clone <repository-url>
   cd edgp-ai-model
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Start the service**:
   ```bash
   python main.py
   ```
   
   You should see:
   ```
   INFO: Starting EDGP AI Model Service v1.0.0
   INFO: Uvicorn running on http://127.0.0.1:8000
   ```

4. **Test the API** (in a new terminal):
   ```bash
   # Quick health check
   curl http://127.0.0.1:8000/api/v1/health
   
   # Test with sample data
   curl -X POST "http://127.0.0.1:8000/api/v1/analyze" \
        -H "Content-Type: application/json" \
        -d '{"data":[{"age":25,"income":50000},{"age":999,"income":999999}],"check_type":"both"}'
   ```

5. **Run examples and demos**:
   ```bash
   # Test AI model functionality
   python examples/ai_usage_example.py
   
   # Run the complete usage guide
   python examples/usage_guide.py
   
   # Check AI model status
   python examples/show_ai_models.py
   ```

6. **View the interactive API docs**:
   Open http://127.0.0.1:8000/docs in your browser

### Local Development Setup

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd edgp-ai-model
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment** (optional):
   ```bash
   cp .env.example .env
   # Edit .env file with your configurations
   ```

4. **Run the application**:
   ```bash
   python main.py
   ```

   Or using uvicorn directly:
   ```bash
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```

### Docker Setup

1. **Build and run with Docker Compose**:
   ```bash
   docker-compose up --build
   ```

2. **Or build and run manually**:
   ```bash
   docker build -t edgp-ai-model .
   docker run -p 8000:8000 edgp-ai-model
   ```

## API Documentation

Once the service is running, visit:
- **API Documentation**: http://localhost:8000/docs
- **Alternative Docs**: http://localhost:8000/redoc
- **Service Info**: http://localhost:8000/

### Available Endpoints

#### Health and Status
- `GET /` - Service information
- `GET /status` - Simple health check
- `GET /api/v1/health` - Detailed health check with AI model status
- `GET /api/v1/info` - Service configuration info
- `GET /api/v1/model-info` - AI model information and capabilities

#### Data Quality Analysis
- `POST /api/v1/analyze` - Analyze data from JSON payload (AI-powered)
- `POST /api/v1/analyze-file` - Analyze data from uploaded file (AI-powered)

## Usage Examples

### Quick Test
```bash
# Start the service
python main.py

# In another terminal, test the API
curl http://127.0.0.1:8000/api/v1/health
```

### Complete Usage Guide
```bash
# Run comprehensive examples
python examples/usage_guide.py

# Test AI capabilities
python examples/ai_usage_example.py

# Check AI model status
python examples/show_ai_models.py
```

### 1. Analyze JSON Data (Basic)

```bash
curl -X POST "http://localhost:8000/api/v1/analyze" \
     -H "Content-Type: application/json" \
     -d '{
       "data": [
         {"age": 25, "income": 50000, "name": "Alice"},
         {"age": 30, "income": 60000, "name": "Bob"},
         {"age": 1000, "income": 1000000, "name": "Outlier"}
       ],
       "check_type": "both",
       "anomaly_threshold": 0.5,
       "duplication_threshold": 0.95
     }'
```

### 2. Upload and Analyze File

```bash
curl -X POST "http://localhost:8000/api/v1/analyze-file" \
     -F "file=@tests/sample_data.csv" \
     -F "check_type=both" \
     -F "anomaly_threshold=0.5"
```

### 3. Python Client Example

```python
import requests
import json

# Sample data
data = {
    "data": [
        {"name": "Alice", "age": 25, "score": 85},
        {"name": "Bob", "age": 30, "score": 90},
        {"name": "Charlie", "age": 1000, "score": -50}  # Anomaly
    ],
    "check_type": "both"
}

# Make request
response = requests.post(
    "http://localhost:8000/api/v1/analyze",
    json=data
)

result = response.json()
print(f"Found {len(result['anomalies'])} anomalies")
print(f"Found {len(result['duplications'])} duplication groups")
```

### Real-World Examples

#### Employee Data Quality Check
```python
import requests

data = {
    "data": [
        {"id": 1, "name": "Alice", "age": 25, "salary": 50000, "department": "Engineering"},
        {"id": 2, "name": "Bob", "age": 30, "salary": 60000, "department": "Marketing"},
        # Anomaly: Suspicious data
        {"id": 3, "name": "Outlier", "age": 999, "salary": 9999999, "department": "Unknown"},
        # Duplicate of Alice
        {"id": 4, "name": "Alice", "age": 25, "salary": 50000, "department": "Engineering"}
    ],
    "check_type": "both",
    "anomaly_threshold": 0.3
}

response = requests.post("http://localhost:8000/api/v1/analyze", json=data)
result = response.json()

print(f"Anomalies found: {len(result['anomalies'])}")
print(f"Duplicates found: {len(result['duplications'])}")
```

#### Financial Transaction Analysis
```python
# Detect fraudulent transactions
transaction_data = {
    "data": [
        {"amount": 100.50, "merchant": "Grocery Store", "category": "Food"},
        {"amount": 50.25, "merchant": "Gas Station", "category": "Transport"},
        # Suspicious transaction
        {"amount": 50000.00, "merchant": "Unknown", "category": "Transfer"}
    ],
    "check_type": "anomaly",
    "anomaly_threshold": 0.2  # High sensitivity
}

response = requests.post("http://localhost:8000/api/v1/analyze", json=transaction_data)
```

#### Customer Data Deduplication
```python
# Find duplicate customer records
customer_data = {
    "data": [
        {"email": "alice@email.com", "name": "Alice Johnson", "phone": "123-456-7890"},
        {"email": "bob@email.com", "name": "Bob Smith", "phone": "234-567-8901"},
        # Near duplicate
        {"email": "alice.johnson@email.com", "name": "Alice Johnson", "phone": "123-456-7890"}
    ],
    "check_type": "duplication",
    "duplication_threshold": 0.8  # Detect similar records
}

response = requests.post("http://localhost:8000/api/v1/analyze", json=customer_data)
```

## Configuration

### Environment Variables

The service can be configured using environment variables or the `.env` file:

```bash
# API Configuration
APP_NAME=EDGP AI Model Service
APP_VERSION=1.0.0
DEBUG=true

# Server Configuration
HOST=127.0.0.1
PORT=8000

# Model Configuration
MODEL_CACHE_DIR=./models
MAX_FILE_SIZE=10485760

# Detection Thresholds
ANOMALY_THRESHOLD=0.5
DUPLICATION_THRESHOLD=0.95

# Logging
LOG_LEVEL=INFO
```

### Detection Types

- `"anomaly"` - Only anomaly detection
- `"duplication"` - Only duplication detection  
- `"both"` - Both anomaly and duplication detection (default)

## AI Models

### Anomaly Detection
- **Isolation Forest**: Identifies outliers by isolating anomalies in feature space
- **DBSCAN Clustering**: Detects anomalies as noise points outside clusters
- **Threshold-based**: Configurable sensitivity for anomaly classification

### Duplication Detection
- **Cosine Similarity**: Measures similarity between data records
- **Configurable Threshold**: Adjustable similarity threshold for duplicate classification
- **Column-wise Analysis**: Identifies which columns contribute to duplications

## Testing

### Run Tests

```bash
# Install test dependencies
pip install pytest pytest-asyncio

# Run all tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_api.py -v

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=html
```

### Test with Sample Data

```bash
# Test with sample CSV
curl -X POST "http://localhost:8000/api/v1/analyze-file" \
     -F "file=@tests/sample_data.csv" \
     -F "check_type=both"

# Test with sample JSON
curl -X POST "http://localhost:8000/api/v1/analyze" \
     -H "Content-Type: application/json" \
     -d @tests/sample_data.json
```

## Performance Considerations

- **Memory Usage**: Large datasets are processed in chunks to manage memory
- **Processing Time**: Depends on data size and complexity
- **Concurrent Requests**: FastAPI handles multiple requests asynchronously
- **Model Caching**: Trained models are cached for reuse

## Development

### Adding New Detection Methods

1. Extend the `TabularAnomalyDetector` or `DuplicationDetector` classes
2. Update the API schemas if needed
3. Add corresponding tests
4. Update documentation

### Code Style

- Follow PEP 8 guidelines
- Use type hints throughout the codebase
- Document functions and classes with docstrings
- Maintain test coverage above 80%

## Deployment

### Production Considerations

1. **Security**: Configure CORS properly, use HTTPS
2. **Scaling**: Consider horizontal scaling with load balancers
3. **Monitoring**: Implement proper logging and monitoring
4. **Database**: For large-scale usage, consider persistent storage for models
5. **Authentication**: Add authentication middleware if required

### Cloud Deployment

The service can be deployed on various cloud platforms:
- **AWS**: ECS, Lambda, or EC2
- **Google Cloud**: Cloud Run, GKE, or Compute Engine
- **Azure**: Container Instances, AKS, or Virtual Machines

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Ensure all tests pass
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For questions, issues, or contributions, please:
1. Check the documentation
2. Search existing issues
3. Create a new issue with detailed information

## Roadmap

- [x] ✅ **Integration with Hugging Face transformers for advanced anomaly detection**
- [x] ✅ **AI-powered semantic duplicate detection**
- [x] ✅ **Local model caching and offline inference**
- [ ] Support for time-series anomaly detection
- [ ] Real-time streaming data analysis
- [ ] Advanced visualization dashboards
- [ ] Integration with popular data platforms
- [ ] Custom model training capabilities
- [ ] Multi-language model support
