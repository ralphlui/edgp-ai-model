#!/usr/bin/env python3
"""
Quick AI Installation Check
Run this to see what AI packages you already have installed.
"""

def check_ai_packages():
    """Check which AI packages are available."""
    packages_to_check = {
        'torch': 'PyTorch',
        'transformers': 'Hugging Face Transformers', 
        'huggingface_hub': 'Hugging Face Hub',
        'sentence_transformers': 'Sentence Transformers',
        'datasets': 'Hugging Face Datasets',
        'sklearn': 'Scikit-learn',
        'numpy': 'NumPy',
        'pandas': 'Pandas'
    }
    
    installed = []
    missing = []
    
    print("🔍 Checking AI package installation...")
    print("=" * 50)
    
    for package, name in packages_to_check.items():
        try:
            module = __import__(package)
            version = getattr(module, '__version__', 'unknown')
            print(f"✅ {name}: {version}")
            installed.append(package)
        except ImportError:
            print(f"❌ {name}: Not installed")
            missing.append(package)
    
    print("\n" + "=" * 50)
    print(f"📊 Summary: {len(installed)}/{len(packages_to_check)} packages installed")
    
    if missing:
        print(f"\n🚨 Missing packages: {', '.join(missing)}")
        print("\n💡 To install missing packages, run:")
        
        if 'torch' in missing:
            print("   pip install torch --index-url https://download.pytorch.org/whl/cpu")
        if 'transformers' in missing:
            print("   pip install transformers")
        if 'huggingface_hub' in missing:
            print("   pip install huggingface-hub") 
        if 'sentence_transformers' in missing:
            print("   pip install sentence-transformers")
        if 'datasets' in missing:
            print("   pip install datasets")
    else:
        print("\n🎉 All AI packages are installed! You can download AI models.")
    
    return len(missing) == 0

def test_quick_model_download():
    """Test downloading a small AI model."""
    try:
        print("\n🚀 Testing AI model download...")
        from sentence_transformers import SentenceTransformer
        
        # Try to load a very small model
        print("📥 Downloading sentence-transformers/all-MiniLM-L6-v2 (90MB)...")
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        
        # Test with simple data
        test_texts = ["This is a test", "Another test sentence", "Completely different text"]
        embeddings = model.encode(test_texts)
        
        print(f"✅ Model downloaded and working!")
        print(f"   Generated embeddings shape: {embeddings.shape}")
        
        # Quick anomaly test
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np
        
        similarities = cosine_similarity(embeddings)
        print(f"   Similarity matrix shape: {similarities.shape}")
        
        # The third text should be less similar to the first two
        sim_1_2 = similarities[0, 1]
        sim_1_3 = similarities[0, 2]
        
        print(f"   Similarity between text 1-2: {sim_1_2:.3f}")
        print(f"   Similarity between text 1-3: {sim_1_3:.3f}")
        
        if sim_1_3 < sim_1_2:
            print("✅ AI model is correctly detecting differences!")
        else:
            print("⚠️  AI model results seem unusual, but model is working")
        
        return True
        
    except Exception as e:
        print(f"❌ Model download/test failed: {str(e)}")
        return False

if __name__ == "__main__":
    print("🤖 AI Installation and Model Test")
    print("=" * 60)
    
    # Check installations
    all_installed = check_ai_packages()
    
    if all_installed:
        # Test model download
        model_works = test_quick_model_download()
        
        if model_works:
            print("\n🎉 SUCCESS! AI models are ready to use.")
            print("\nNext steps:")
            print("1. Check the MANUAL_AI_SETUP.md for integration examples")
            print("2. Try the examples in AI_MODEL_USAGE_GUIDE.md")  
            print("3. Integrate AI detection into your service")
        else:
            print("\n⚠️  Packages installed but model test failed.")
            print("Check your internet connection and try again.")
    else:
        print("\n📝 Please install missing packages first.")
        print("See MANUAL_AI_SETUP.md for detailed instructions.")
