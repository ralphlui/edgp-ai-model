#!/usr/bin/env python3
"""
🤖 AI Model Access Guide
This script shows you exactly how to access and use your downloaded AI models.
"""

def show_model_location():
    """Show where AI models are stored."""
    from pathlib import Path
    
    print("📍 AI Model Locations:")
    print("=" * 50)
    
    # Main cache directory
    cache_dir = Path.home() / '.cache' / 'huggingface'
    print(f"Main Cache: {cache_dir}")
    
    # Specific model directory
    model_dir = cache_dir / 'hub' / 'models--sentence-transformers--all-MiniLM-L6-v2'
    print(f"Model Directory: {model_dir}")
    print(f"Model Exists: {model_dir.exists()}")
    
    if model_dir.exists():
        # Show model files
        snapshots = model_dir / 'snapshots'
        if snapshots.exists():
            for snapshot in snapshots.iterdir():
                if snapshot.is_dir():
                    print(f"\n📁 Model Files in {snapshot.name}:")
                    for file in sorted(snapshot.iterdir()):
                        if file.is_file():
                            size_mb = file.stat().st_size / (1024 * 1024)
                            print(f"   {file.name}: {size_mb:.1f} MB")
                    break

def test_model_access():
    """Test accessing the AI model."""
    print("\n🧪 Testing AI Model Access:")
    print("=" * 50)
    
    try:
        # Import and load model
        from sentence_transformers import SentenceTransformer
        print("✅ Importing sentence_transformers - SUCCESS")
        
        # Load the cached model (no download needed)
        print("📥 Loading model from cache...")
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        print("✅ Model loaded successfully!")
        
        # Show model info
        print(f"   Model Name: {model._modules['0'].auto_model.config._name_or_path}")
        print(f"   Embedding Dimension: {model.get_sentence_embedding_dimension()}")
        print(f"   Max Sequence Length: {model.max_seq_length}")
        print(f"   Device: {model.device}")
        
        # Test the model
        print("\n🔍 Testing model with sample data...")
        test_texts = [
            "Normal customer transaction",
            "Regular employee data", 
            "SUSPICIOUS ANOMALOUS ACTIVITY!!!"
        ]
        
        embeddings = model.encode(test_texts)
        print(f"✅ Generated embeddings: {embeddings.shape}")
        
        # Show similarity analysis
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = cosine_similarity(embeddings)
        
        print("\n📊 Similarity Analysis:")
        for i in range(len(test_texts)):
            for j in range(i+1, len(test_texts)):
                sim = similarities[i][j]
                print(f"   '{test_texts[i][:20]}...' vs '{test_texts[j][:20]}...': {sim:.3f}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("   Try: pip install sentence-transformers torch")
        return False
        
    except Exception as e:
        print(f"❌ Model Error: {e}")
        return False

def show_usage_examples():
    """Show how to use the AI model in your code."""
    print("\n💡 Usage Examples:")
    print("=" * 50)
    
    print("""
# 1. Basic Model Loading (uses cached model)
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# 2. Anomaly Detection with AI
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def detect_anomalies_with_ai(data):
    # Convert data to text
    texts = []
    for _, row in data.iterrows():
        text = " | ".join([f"{col}: {val}" for col, val in row.items()])
        texts.append(text)
    
    # Get AI embeddings
    embeddings = model.encode(texts)
    
    # Calculate similarities
    similarities = cosine_similarity(embeddings)
    
    # Find anomalies (low average similarity)
    anomaly_scores = []
    for i in range(len(similarities)):
        others = np.concatenate([similarities[i][:i], similarities[i][i+1:]])
        avg_sim = np.mean(others) if len(others) > 0 else 0
        anomaly_scores.append(1 - avg_sim)
    
    return anomaly_scores

# 3. Integration with Your FastAPI Service
# See: AI_INTEGRATION_GUIDE.md for complete integration steps
""")

def main():
    """Main function to demonstrate AI model access."""
    print("🚀 AI Model Access and Usage Guide")
    print("=" * 60)
    
    # Show where models are stored
    show_model_location()
    
    # Test model access
    model_works = test_model_access()
    
    if model_works:
        # Show usage examples
        show_usage_examples()
        
        print("\n🎉 SUCCESS! Your AI models are ready to use!")
        print("\n📚 Next Steps:")
        print("1. Follow AI_INTEGRATION_GUIDE.md to integrate with your FastAPI service")
        print("2. Run: python ai_usage_example.py for comprehensive examples")
        print("3. Test with your own data using the examples above")
        
    else:
        print("\n❌ AI models not accessible. Please check installation:")
        print("   pip install sentence-transformers torch transformers")

if __name__ == "__main__":
    main()
