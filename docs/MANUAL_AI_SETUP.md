# 🤖 Manual AI Model Installation Guide

## Step-by-Step Installation Instructions

Since automated installation might have issues with conda environments, here's a manual approach:

### 1. 📦 Install Dependencies Manually

Open your terminal and run these commands one by one:

```bash
# Navigate to your project directory
cd /Users/jjfwang/Documents/02-NUS/01-Capstone/edgp-ai-model

# Install PyTorch (CPU version - smaller and faster for most tasks)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Install Transformers library
pip install transformers

# Install Hugging Face Hub
pip install huggingface-hub

# Install Sentence Transformers (easier to use for embeddings)
pip install sentence-transformers

# Install additional ML libraries
pip install datasets scikit-learn
```

### 2. 🧪 Test Installation

Run this Python code to test if everything works:

```python
# Test script - save as test_ai_installation.py
try:
    import torch
    print("✅ PyTorch installed:", torch.__version__)
except ImportError:
    print("❌ PyTorch not installed")

try:
    import transformers
    print("✅ Transformers installed:", transformers.__version__)
except ImportError:
    print("❌ Transformers not installed")

try:
    import sentence_transformers
    print("✅ Sentence Transformers installed:", sentence_transformers.__version__)
except ImportError:
    print("❌ Sentence Transformers not installed")

try:
    from huggingface_hub import HfApi
    print("✅ Hugging Face Hub installed")
except ImportError:
    print("❌ Hugging Face Hub not installed")

print("\n🎉 If all show ✅, you're ready to download AI models!")
```

### 3. 🚀 Quick Model Download Test

```python
# Quick test - save as quick_model_test.py
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np

print("📥 Downloading a lightweight model...")

# Download a small, fast model (only ~90MB)
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
print("✅ Model downloaded successfully!")

# Test with sample data
sample_data = [
    "Normal transaction: amount=100, user=john, location=NYC",
    "Normal transaction: amount=50, user=jane, location=LA", 
    "SUSPICIOUS: amount=999999, user=hacker, location=Unknown"
]

print("\n🧪 Testing model with sample data...")
embeddings = model.encode(sample_data)
print(f"✅ Generated embeddings shape: {embeddings.shape}")

# Simple similarity check
from sklearn.metrics.pairwise import cosine_similarity
similarities = cosine_similarity(embeddings)
print("\n📊 Similarity matrix:")
print(similarities)

print("\n🎉 AI model is working! The last row should have lower similarity to others.")
```

### 4. 🔍 Available Models for Download

Here are some good models to try:

#### Small and Fast Models (Recommended to start):
```python
# Sentence embedding models (good for anomaly detection)
"sentence-transformers/all-MiniLM-L6-v2"        # 90MB, very fast
"sentence-transformers/all-distilroberta-v1"    # 290MB, good quality

# Language models
"distilbert-base-uncased"                       # 250MB, good for text
"microsoft/DialoGPT-small"                      # 350MB, conversational
```

#### Larger, More Powerful Models:
```python
# Better quality but slower
"sentence-transformers/all-mpnet-base-v2"       # 420MB, high quality
"bert-base-uncased"                             # 440MB, classic BERT
"microsoft/DialoGPT-medium"                     # 1GB, better conversations
```

### 5. 💡 Usage Examples

#### Example 1: Basic Anomaly Detection
```python
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Your data
data = pd.DataFrame({
    'age': [25, 30, 35, 28, 999],  # 999 is clearly anomalous
    'salary': [50000, 60000, 70000, 55000, 999999],  # 999999 is anomalous
    'department': ['Eng', 'Marketing', 'Sales', 'Eng', 'Unknown']
})

# Convert to text representation
texts = []
for _, row in data.iterrows():
    text = f"age: {row['age']}, salary: {row['salary']}, department: {row['department']}"
    texts.append(text)

# Get embeddings
embeddings = model.encode(texts)

# Find anomalies using similarity
similarities = cosine_similarity(embeddings)
avg_similarities = [np.mean(similarities[i][np.arange(len(similarities)) != i]) for i in range(len(similarities))]

# Lowest similarity = highest anomaly score
anomaly_threshold = np.percentile(avg_similarities, 20)  # Bottom 20%
anomalies = [i for i, sim in enumerate(avg_similarities) if sim < anomaly_threshold]

print("Anomalies found at indices:", anomalies)
print("Anomalous rows:")
for idx in anomalies:
    print(f"  {idx}: {data.iloc[idx].to_dict()}")
```

#### Example 2: Duplicate Detection
```python
# Same setup as above...

# Find duplicates using high similarity
duplicate_threshold = 0.95  # 95% similarity
duplicates = []

for i in range(len(similarities)):
    for j in range(i+1, len(similarities)):
        if similarities[i][j] > duplicate_threshold:
            duplicates.append((i, j, similarities[i][j]))

print("Duplicate pairs found:")
for i, j, sim in duplicates:
    print(f"  Rows {i} and {j} are {sim:.3f} similar")
    print(f"    Row {i}: {data.iloc[i].to_dict()}")
    print(f"    Row {j}: {data.iloc[j].to_dict()}")
```

### 6. 🔧 Integration with Your Service

To integrate with your existing EDGP AI service, update your `src/models/anomaly_detector.py`:

```python
# At the top of anomaly_detector.py, add:
try:
    from sentence_transformers import SentenceTransformer
    import torch
    AI_MODELS_AVAILABLE = True
except ImportError:
    AI_MODELS_AVAILABLE = False

# Then add this new class:
class HuggingFaceAnomalyDetector:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        if not AI_MODELS_AVAILABLE:
            raise ValueError("AI models not available. Please install: pip install sentence-transformers torch")
        
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
    
    def detect_anomalies(self, data, contamination=0.1):
        # Convert data to text representation
        texts = []
        for _, row in data.iterrows():
            text = " | ".join([f"{col}: {val}" for col, val in row.items()])
            texts.append(text)
        
        # Get embeddings and detect anomalies
        embeddings = self.model.encode(texts)
        similarities = cosine_similarity(embeddings)
        
        # Calculate anomaly scores
        anomaly_scores = []
        for i in range(len(similarities)):
            others = np.concatenate([similarities[i][:i], similarities[i][i+1:]])
            avg_sim = np.mean(others)
            anomaly_scores.append(1 - avg_sim)  # Lower similarity = higher anomaly score
        
        # Find anomalies
        threshold = np.percentile(anomaly_scores, (1 - contamination) * 100)
        anomalies = [i for i, score in enumerate(anomaly_scores) if score > threshold]
        
        return {
            "anomalies": anomalies,
            "scores": anomaly_scores,
            "threshold": threshold,
            "method": f"AI-based using {self.model_name}"
        }
```

### 7. 🏃‍♂️ Quick Start Commands

```bash
# 1. Install everything at once
pip install torch --index-url https://download.pytorch.org/whl/cpu transformers huggingface-hub sentence-transformers

# 2. Test installation
python -c "import torch, transformers, sentence_transformers; print('✅ All AI libraries installed!')"

# 3. Download and test a model
python -c "
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
print('✅ Model downloaded and ready!')
print('Model path:', model._modules['0'].auto_model.config._name_or_path)
"

# 4. Check where models are stored
python -c "
import os
from pathlib import Path
cache_dir = Path.home() / '.cache' / 'huggingface'
print('Models cached in:', cache_dir)
if cache_dir.exists():
    print('Cache size:', sum(f.stat().st_size for f in cache_dir.rglob('*') if f.is_file()) // (1024**2), 'MB')
"
```

### 8. 🛠️ Troubleshooting

#### Common Issues:

1. **"No module named 'torch'"**
   ```bash
   pip install torch --index-url https://download.pytorch.org/whl/cpu
   ```

2. **"No module named 'transformers'"**
   ```bash
   pip install transformers
   ```

3. **Memory issues with large models**
   - Use smaller models like `all-MiniLM-L6-v2`
   - Add `device='cpu'` when loading models

4. **Download fails**
   ```python
   # Try with explicit cache directory
   from sentence_transformers import SentenceTransformer
   model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', 
                              cache_folder='./models')
   ```

5. **Conda environment issues**
   ```bash
   # If using conda, try:
   conda install pytorch cpuonly -c pytorch
   pip install transformers sentence-transformers
   ```

### 9. 🎯 Next Steps

1. **Install the dependencies** using the commands above
2. **Test with the quick examples** to make sure everything works
3. **Integrate into your service** using the enhanced detector code
4. **Test with your actual data** to see how well it performs
5. **Experiment with different models** to find the best one for your use case

The models will be automatically cached locally (usually in `~/.cache/huggingface/`) so you only download them once!
