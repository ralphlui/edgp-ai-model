# Amazon Science Model Integration Guide

## Current Implementation Status

The EDGP AI Model service currently has the Amazon Science model name configured but **not actively integrated**. Here's the current state and integration options:

## 🔍 Current Configuration

In `config/settings.py`:
```python
model_name: str = "amazon-science/tabular-anomaly-detection"
```

**Status**: The model name is defined but the service uses local scikit-learn models (Isolation Forest, DBSCAN).

## 🚀 Integration Options

### Option 1: Local Hugging Face Integration (Recommended)

**How it works:**
- Download the model from Hugging Face Hub to local cache
- Run inference locally using the downloaded model
- No internet required after initial download
- Full control over model versioning and caching

**Pros:**
- ✅ Fast inference (no network latency)
- ✅ Data privacy (no data sent to cloud)
- ✅ Offline operation after download
- ✅ Cost-effective (no API charges)
- ✅ Consistent performance

**Cons:**
- ❌ Requires local compute resources
- ❌ Model download time on first use
- ❌ Storage space for model files

**Implementation:**
```python
from transformers import AutoModel, AutoTokenizer
from huggingface_hub import hf_hub_download

# Download and cache model locally
model = AutoModel.from_pretrained(
    "amazon-science/tabular-anomaly-detection",
    cache_dir="./models",
    trust_remote_code=True
)

# Run inference locally
outputs = model(input_data)
```

### Option 2: Cloud API Integration

**How it works:**
- Send data to Hugging Face Inference API
- Receive predictions via REST API
- Model runs on Hugging Face's servers

**Pros:**
- ✅ No local compute requirements
- ✅ Always latest model version
- ✅ Scalable infrastructure

**Cons:**
- ❌ Network latency
- ❌ Data privacy concerns
- ❌ API rate limits and costs
- ❌ Requires internet connection

**Implementation:**
```python
import requests

API_URL = "https://api-inference.huggingface.co/models/amazon-science/tabular-anomaly-detection"
headers = {"Authorization": f"Bearer {hf_token}"}

response = requests.post(API_URL, headers=headers, json=data)
```

## 🎯 Recommended Approach

**For EDGP AI Model Service, I recommend Option 1 (Local Integration)** because:

1. **Enterprise Requirements**: Better for enterprise use cases requiring data privacy
2. **Performance**: Faster response times for real-time analysis
3. **Reliability**: No dependency on external API availability
4. **Cost**: No per-request charges
5. **Security**: Data stays within your infrastructure

## 🔧 Implementation Steps

### Step 1: Install Dependencies

```bash
# Add to requirements.txt
torch>=2.0.0
transformers>=4.30.0
huggingface-hub>=0.15.0
```

### Step 2: Update Model Loader

The enhanced anomaly detector I created (`anomaly_detector_enhanced.py`) includes:

```python
class HuggingFaceAnomalyDetector:
    def __init__(self, model_name: str = "amazon-science/tabular-anomaly-detection"):
        # Download model to local cache
        self.model = AutoModel.from_pretrained(
            model_name,
            cache_dir=self.model_cache_dir,
            trust_remote_code=True  # Required for custom models
        )
    
    def predict_anomalies_with_hf_model(self, data: pd.DataFrame):
        # Convert data to model format
        # Run inference
        # Return anomaly scores
```

### Step 3: Fallback Strategy

The implementation includes automatic fallback:

```python
def predict_anomalies(self, data: pd.DataFrame):
    # Try Hugging Face model first
    if self.hf_detector and self.hf_detector.is_loaded:
        try:
            return self.hf_detector.predict_anomalies_with_hf_model(data)
        except Exception as e:
            logger.warning(f"HF model failed: {e}. Using local models.")
    
    # Fallback to sklearn models
    return self._predict_anomalies_local(data)
```

## 📊 Model Information Endpoint

Added endpoint to check which model is being used:

```python
@app.get("/api/v1/model-info")
def get_model_info():
    detector = TabularAnomalyDetector()
    return detector.get_model_info()
    # Returns: {
    #   "model_name": "amazon-science/tabular-anomaly-detection",
    #   "using_huggingface": true,
    #   "huggingface_available": true,
    #   "fallback_models": ["IsolationForest", "DBSCAN"]
    # }
```

## 🚧 Important Notes

### About amazon-science/tabular-anomaly-detection

⚠️ **Note**: The exact model `amazon-science/tabular-anomaly-detection` may not exist on Hugging Face Hub. You should:

1. **Verify the model exists**: Check https://huggingface.co/amazon-science/
2. **Use actual model name**: Find the correct model name from Amazon Science
3. **Check model format**: Ensure it's compatible with transformers library

### Common Amazon Science Models

Some actual models you might find:
- `amazon-science/deebert-base`
- `amazon-science/chronos-t5-small`
- Custom models from Amazon Research papers

## 🛠️ Next Steps

1. **Verify Model Availability**: Check if the exact model exists
2. **Install Dependencies**: Add torch, transformers, huggingface-hub
3. **Test Integration**: Start with a simple model download test
4. **Performance Testing**: Compare HF model vs local sklearn performance
5. **Production Deployment**: Configure model caching and error handling

## 📈 Current Service Status

✅ **Ready for Integration**: The service architecture supports both local and cloud models
✅ **Fallback Implemented**: Automatic fallback to sklearn if HF model fails  
✅ **Configuration Ready**: Model name is configurable via environment variables
✅ **API Compatible**: Existing endpoints will work with either model type

Would you like me to proceed with installing the dependencies and implementing the full integration?
