#!/usr/bin/env python3
"""
Complete AI Model Setup and Testing Script
This script will install dependencies, download models, and test them with your service.
"""

import subprocess
import sys
import os
import pandas as pd
from pathlib import Path

def install_dependencies():
    """Install required AI dependencies."""
    print("📦 Installing AI dependencies...")
    
    # Commands to try (in order of preference)
    install_commands = [
        # Try PyTorch CPU version first
        [sys.executable, "-m", "pip", "install", "torch", "--index-url", "https://download.pytorch.org/whl/cpu"],
        [sys.executable, "-m", "pip", "install", "transformers"],
        [sys.executable, "-m", "pip", "install", "huggingface-hub"],
        [sys.executable, "-m", "pip", "install", "datasets"],
        [sys.executable, "-m", "pip", "install", "sentence-transformers"],
    ]
    
    for cmd in install_commands:
        try:
            print(f"Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(f"✅ Success: {cmd[-1]}")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed: {cmd[-1]} - {e}")
            return False
    
    return True

def test_imports():
    """Test if all required packages can be imported."""
    print("\n🧪 Testing imports...")
    
    required_packages = [
        "torch",
        "transformers", 
        "huggingface_hub",
        "sentence_transformers"
    ]
    
    all_good = True
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError as e:
            print(f"❌ {package}: {e}")
            all_good = False
    
    return all_good

def download_and_test_models():
    """Download and test AI models."""
    print("\n🤖 Downloading and testing AI models...")
    
    try:
        # Import required libraries
        import torch
        from transformers import AutoModel, AutoTokenizer
        from sentence_transformers import SentenceTransformer
        
        # Test models (from simple to complex)
        models_to_test = [
            {
                "name": "sentence-transformers/all-MiniLM-L6-v2",
                "type": "sentence_transformer",
                "description": "Lightweight sentence embedding model"
            },
            {
                "name": "distilbert-base-uncased", 
                "type": "transformers",
                "description": "Lightweight BERT variant"
            }
        ]
        
        successful_models = []
        
        for model_info in models_to_test:
            print(f"\n📥 Testing: {model_info['name']}")
            print(f"   Description: {model_info['description']}")
            
            try:
                if model_info["type"] == "sentence_transformer":
                    # Test sentence transformer
                    model = SentenceTransformer(model_info["name"])
                    
                    # Test with sample text
                    test_texts = ["This is a test sentence", "Another test sentence"]
                    embeddings = model.encode(test_texts)
                    print(f"   ✅ Embedding shape: {embeddings.shape}")
                    
                elif model_info["type"] == "transformers":
                    # Test regular transformer
                    tokenizer = AutoTokenizer.from_pretrained(model_info["name"])
                    model = AutoModel.from_pretrained(model_info["name"])
                    
                    # Test with sample input
                    inputs = tokenizer("This is a test", return_tensors="pt")
                    with torch.no_grad():
                        outputs = model(**inputs)
                    print(f"   ✅ Output shape: {outputs.last_hidden_state.shape}")
                
                successful_models.append(model_info["name"])
                print(f"   ✅ {model_info['name']} downloaded and tested successfully")
                
            except Exception as e:
                print(f"   ❌ Failed to test {model_info['name']}: {str(e)}")
        
        return successful_models
        
    except Exception as e:
        print(f"❌ Error in model testing: {str(e)}")
        return []

def test_with_sample_data():
    """Test AI models with sample tabular data."""
    print("\n📊 Testing with sample tabular data...")
    
    try:
        from sentence_transformers import SentenceTransformer
        
        # Create sample dataset with anomalies
        sample_data = pd.DataFrame({
            'age': [25, 30, 35, 28, 999, 32, 29],  # 999 is anomaly
            'salary': [50000, 60000, 70000, 55000, 999999, 62000, 58000],  # 999999 is anomaly
            'score': [85, 90, 88, 87, -50, 89, 86],  # -50 is anomaly
            'department': ['Engineering', 'Marketing', 'Sales', 'Engineering', 'Unknown', 'Marketing', 'Sales']
        })
        
        print("Sample data:")
        print(sample_data)
        
        # Load model
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        
        # Convert rows to text and get embeddings
        text_data = []
        for _, row in sample_data.iterrows():
            text = f"age: {row['age']}, salary: {row['salary']}, score: {row['score']}, department: {row['department']}"
            text_data.append(text)
        
        embeddings = model.encode(text_data)
        print(f"\n✅ Generated embeddings shape: {embeddings.shape}")
        
        # Simple anomaly detection using distance from mean
        import numpy as np
        from sklearn.metrics.pairwise import cosine_similarity
        
        # Calculate similarities
        similarities = cosine_similarity(embeddings)
        
        # Find points with low average similarity (potential anomalies)
        avg_similarities = []
        for i in range(len(similarities)):
            others = np.concatenate([similarities[i][:i], similarities[i][i+1:]])
            avg_sim = np.mean(others)
            avg_similarities.append(avg_sim)
        
        # Find anomalies (lowest 20% similarity)
        threshold = np.percentile(avg_similarities, 20)
        anomalies = [i for i, sim in enumerate(avg_similarities) if sim < threshold]
        
        print(f"\n🔍 Anomaly Detection Results:")
        print(f"   Threshold: {threshold:.3f}")
        print(f"   Detected anomalies at indices: {anomalies}")
        
        for idx in anomalies:
            print(f"   Row {idx}: {sample_data.iloc[idx].to_dict()}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing with sample data: {str(e)}")
        return False

def create_usage_example():
    """Create a complete usage example."""
    print("\n📝 Creating usage example...")
    
    example_code = '''
# 🚀 Complete Example: Using AI Models for Anomaly Detection

import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def detect_anomalies_with_ai(data: pd.DataFrame, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """
    Detect anomalies in tabular data using AI embeddings.
    
    Args:
        data: Input dataframe
        model_name: Hugging Face model name
    
    Returns:
        List of anomaly indices and scores
    """
    # Load the AI model
    model = SentenceTransformer(model_name)
    
    # Convert each row to text representation
    text_data = []
    for _, row in data.iterrows():
        # Create text representation of the row
        text_parts = [f"{col}: {val}" for col, val in row.items()]
        text = " | ".join(text_parts)
        text_data.append(text)
    
    # Get embeddings for each row
    embeddings = model.encode(text_data)
    
    # Calculate similarity matrix
    similarity_matrix = cosine_similarity(embeddings)
    
    # Calculate anomaly scores (lower avg similarity = higher anomaly score)
    anomaly_scores = []
    for i in range(len(similarity_matrix)):
        # Get similarities to all other points (excluding self)
        similarities = np.concatenate([
            similarity_matrix[i][:i],
            similarity_matrix[i][i+1:]
        ])
        # Average similarity (higher = more normal)
        avg_similarity = np.mean(similarities)
        # Anomaly score (lower similarity = higher anomaly score)
        anomaly_score = 1 - avg_similarity
        anomaly_scores.append(anomaly_score)
    
    # Identify anomalies (top 10% highest scores)
    threshold = np.percentile(anomaly_scores, 90)
    anomalies = [i for i, score in enumerate(anomaly_scores) if score > threshold]
    
    return {
        "anomalies": anomalies,
        "scores": anomaly_scores,
        "threshold": threshold
    }

# Example usage
if __name__ == "__main__":
    # Create sample data with clear anomalies
    data = pd.DataFrame({
        'age': [25, 30, 35, 28, 999, 32, 29, 31],          # 999 is anomaly
        'salary': [50000, 60000, 70000, 55000, 999999, 62000, 58000, 61000],  # 999999 is anomaly  
        'score': [85, 90, 88, 87, -50, 89, 86, 88],        # -50 is anomaly
        'department': ['Eng', 'Marketing', 'Sales', 'Eng', 'Unknown', 'Marketing', 'Sales', 'Eng']
    })
    
    print("Sample Data:")
    print(data)
    
    # Detect anomalies
    results = detect_anomalies_with_ai(data)
    
    print(f"\\nAnomalies detected at indices: {results['anomalies']}")
    print(f"Anomaly threshold: {results['threshold']:.3f}")
    
    print("\\nAnomalous rows:")
    for idx in results['anomalies']:
        score = results['scores'][idx]
        print(f"  Row {idx} (score: {score:.3f}): {data.iloc[idx].to_dict()}")
'''
    
    # Save example to file
    with open("ai_anomaly_example.py", "w") as f:
        f.write(example_code)
    
    print("✅ Example saved to ai_anomaly_example.py")
    return True

def main():
    """Main setup function."""
    print("🚀 AI Model Setup and Testing")
    print("=" * 50)
    
    # Step 1: Install dependencies
    if not install_dependencies():
        print("❌ Failed to install dependencies. Please install manually:")
        print("   pip install torch --index-url https://download.pytorch.org/whl/cpu")
        print("   pip install transformers huggingface-hub sentence-transformers")
        return False
    
    # Step 2: Test imports
    if not test_imports():
        print("❌ Import test failed. Please check installation.")
        return False
    
    # Step 3: Download and test models
    successful_models = download_and_test_models()
    if not successful_models:
        print("❌ No models downloaded successfully.")
        return False
    
    print(f"\n✅ Successfully downloaded {len(successful_models)} models:")
    for model in successful_models:
        print(f"   • {model}")
    
    # Step 4: Test with sample data
    if test_with_sample_data():
        print("✅ Sample data testing successful!")
    else:
        print("⚠️  Sample data testing failed, but models are downloaded.")
    
    # Step 5: Create usage example
    create_usage_example()
    
    print("\n🎉 Setup Complete!")
    print("\nNext steps:")
    print("1. Run: python ai_anomaly_example.py")
    print("2. Integrate AI models into your service using the enhanced detector")
    print("3. Test with your own data")
    print("\nModels are cached locally in ~/.cache/huggingface/transformers/")
    
    return True

if __name__ == "__main__":
    main()
