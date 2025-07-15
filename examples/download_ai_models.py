# 🤖 AI Model Download and Usage Guide

## How to Download and Use AI Models Locally

This guide shows you how to download and integrate AI models locally with the EDGP AI Model Service.

## 📦 Step 1: Install AI Model Dependencies

### Option A: Using pip (Recommended)
```bash
# Install PyTorch (CPU version)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Install Transformers and Hugging Face Hub
pip install transformers huggingface-hub datasets

# Verify installation
python -c "import torch, transformers; print('✅ AI dependencies installed successfully')"
```

### Option B: Using conda
```bash
# Install PyTorch via conda-forge
conda install pytorch cpuonly -c pytorch

# Install transformers
conda install transformers -c conda-forge

# Install huggingface hub
pip install huggingface-hub
```

## 🔍 Step 2: Find Available AI Models

### Popular Anomaly Detection Models on Hugging Face:

1. **General Purpose Models:**
   - `microsoft/DialoGPT-medium` (for text anomalies)
   - `sentence-transformers/all-MiniLM-L6-v2` (for similarity)
   - `huggingface/CodeBERTa-small-v1` (for code anomalies)

2. **Financial/Tabular Models:**
   - `microsoft/table-transformer-structure-recognition`
   - `google/tapas-base`
   - Custom fine-tuned models

3. **Research Models:**
   - Check Amazon Science papers and repositories
   - Look for domain-specific models

### Search for Models:
```bash
# Search Hugging Face Hub
pip install huggingface-hub
python -c "
from huggingface_hub import HfApi
api = HfApi()
models = api.list_models(search='anomaly detection', limit=10)
for model in models:
    print(f'Model: {model.modelId}')
"
```

## 🚀 Step 3: Create Model Download Script

Let me create a script to download and test AI models:

```python
#!/usr/bin/env python3
"""
AI Model Download and Integration Script
Downloads and tests various AI models for anomaly detection.
"""

import os
import torch
from transformers import AutoModel, AutoTokenizer, AutoConfig
from huggingface_hub import hf_hub_download, list_repo_files
import pandas as pd
import numpy as np
from typing import Dict, Any, List
import json

class ModelDownloader:
    def __init__(self, cache_dir: str = "./models"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    
    def download_model(self, model_name: str) -> Dict[str, Any]:
        """Download a model from Hugging Face Hub."""
        print(f"🔍 Checking model: {model_name}")
        
        try:
            # Check if model exists and get info
            config = AutoConfig.from_pretrained(model_name, cache_dir=self.cache_dir)
            print(f"✅ Model config loaded: {model_name}")
            
            # Download the model
            print(f"📥 Downloading model: {model_name}")
            model = AutoModel.from_pretrained(
                model_name,
                cache_dir=self.cache_dir,
                trust_remote_code=True
            )
            
            # Try to get tokenizer (not all models have one)
            tokenizer = None
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    cache_dir=self.cache_dir
                )
                print(f"✅ Tokenizer loaded")
            except:
                print(f"ℹ️  No tokenizer available for this model")
            
            return {
                "status": "success",
                "model": model,
                "tokenizer": tokenizer,
                "config": config,
                "model_name": model_name
            }
            
        except Exception as e:
            print(f"❌ Error downloading {model_name}: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "model_name": model_name
            }
    
    def test_model_with_data(self, model_info: Dict[str, Any], test_data: pd.DataFrame):
        """Test a downloaded model with sample data."""
        if model_info["status"] != "success":
            print(f"❌ Cannot test model due to download error")
            return
        
        print(f"🧪 Testing model: {model_info['model_name']}")
        
        try:
            model = model_info["model"]
            
            # Convert data to tensors (simple approach)
            # This is a basic example - real implementation depends on model architecture
            numeric_data = test_data.select_dtypes(include=[np.number])
            if len(numeric_data.columns) > 0:
                # Convert to tensor
                input_tensor = torch.tensor(numeric_data.values, dtype=torch.float32)
                
                # Run inference
                with torch.no_grad():
                    try:
                        outputs = model(input_tensor)
                        print(f"✅ Model inference successful")
                        print(f"   Output shape: {outputs.last_hidden_state.shape if hasattr(outputs, 'last_hidden_state') else 'Custom output'}")
                        return True
                    except Exception as e:
                        print(f"⚠️  Model inference failed: {str(e)}")
                        print(f"   This model might need different input format")
                        return False
            else:
                print(f"⚠️  No numeric data found for testing")
                return False
                
        except Exception as e:
            print(f"❌ Error testing model: {str(e)}")
            return False

def test_popular_models():
    """Test downloading some popular models."""
    downloader = ModelDownloader()
    
    # Sample test data
    test_data = pd.DataFrame({
        'age': [25, 30, 35, 999],
        'salary': [50000, 60000, 70000, 999999],
        'score': [85, 90, 88, -50]
    })
    
    # List of models to try (start with smaller/simpler ones)
    models_to_try = [
        "sentence-transformers/all-MiniLM-L6-v2",  # Good for embeddings
        "microsoft/DialoGPT-small",                # Small language model
        "distilbert-base-uncased",                 # Smaller BERT
        # Add more models as needed
    ]
    
    successful_models = []
    
    for model_name in models_to_try:
        print(f"\\n{'='*60}")
        model_info = downloader.download_model(model_name)
        
        if model_info["status"] == "success":
            successful_models.append(model_name)
            # Test with data
            downloader.test_model_with_data(model_info, test_data)
        
        print(f"{'='*60}")
    
    print(f"\\n🎉 Successfully downloaded models:")
    for model in successful_models:
        print(f"  ✅ {model}")
    
    return successful_models

if __name__ == "__main__":
    print("🚀 AI Model Download and Test Script")
    print("This script will download and test AI models for anomaly detection")
    
    # Test model downloads
    successful_models = test_popular_models()
    
    if successful_models:
        print(f"\\n✅ {len(successful_models)} models downloaded successfully!")
        print("These models are now cached locally and ready to use.")
    else:
        print("\\n❌ No models downloaded successfully.")
        print("Check your internet connection and try again.")
```
