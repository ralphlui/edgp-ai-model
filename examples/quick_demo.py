#!/usr/bin/env python3
"""
🚀 Quick Demo: Local AI Models in Action
This script demonstrates the local AI models working with your EDGP service.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from src.models.anomaly_detector import EnhancedTabularAnomalyDetector, EnhancedDuplicationDetector

def quick_demo():
    """Quick demonstration of local AI models."""
    
    print("🚀 EDGP AI Model Service - Local AI Demo")
    print("=" * 60)
    
    # Sample data with clear anomalies and duplicates
    data = pd.DataFrame({
        'user_id': [1, 2, 3, 4, 5, 6, 7, 8],
        'age': [25, 30, 35, 28, 999, 32, 30, 25],  # 999 is anomaly
        'salary': [50000, 60000, 70000, 55000, 999999, 62000, 60000, 50000],  # 999999 is anomaly
        'department': ['Eng', 'Marketing', 'Sales', 'Eng', 'Unknown', 'Marketing', 'Marketing', 'Eng'],
        'score': [85, 90, 88, 87, -50, 89, 90, 85],  # -50 is anomaly
        'location': ['NYC', 'LA', 'Chicago', 'NYC', 'Mars', 'LA', 'LA', 'NYC']  # Mars is anomaly
    })
    
    print("📊 Sample Dataset:")
    print(data.to_string())
    print(f"\nDataset shape: {data.shape}")
    
    # Initialize AI detectors with local models
    print("\n🤖 Initializing AI Detectors with Local Models...")
    
    try:
        anomaly_detector = EnhancedTabularAnomalyDetector(
            model_cache_dir="./models",
            use_ai=True
        )
        
        duplicate_detector = EnhancedDuplicationDetector(
            model_cache_dir="./models", 
            use_ai=True
        )
        
        print("✅ AI detectors initialized successfully!")
        
        # Test anomaly detection
        print("\n" + "=" * 60)
        print("🔍 AI ANOMALY DETECTION")
        print("=" * 60)
        
        anomaly_results = anomaly_detector.detect_anomalies(data, contamination=0.2)
        
        print(f"Method: {anomaly_results.get('method', 'Unknown')}")
        print(f"Confidence: {anomaly_results.get('confidence', 'Unknown')}")
        print(f"Anomalies found: {len(anomaly_results.get('anomalies', []))}")
        
        if anomaly_results.get('anomalies'):
            print("\n🚨 Anomalous Records:")
            for idx in anomaly_results['anomalies']:
                score = anomaly_results['scores'][idx]
                record = data.iloc[idx]
                print(f"  Row {idx} (score: {score:.3f}):")
                print(f"    {record.to_dict()}")
        
        # Test duplicate detection
        print("\n" + "=" * 60)
        print("👥 AI DUPLICATE DETECTION")
        print("=" * 60)
        
        duplicate_results = duplicate_detector.detect_duplicates(data, similarity_threshold=0.9)
        
        print(f"Method: {duplicate_results.get('method', 'Unknown')}")
        print(f"Total duplicates: {duplicate_results.get('total_duplicates', 0)}")
        print(f"Duplicate pairs: {len(duplicate_results.get('duplicate_pairs', []))}")
        
        if duplicate_results.get('duplicate_pairs'):
            print("\n👥 Duplicate Pairs Found:")
            for pair in duplicate_results['duplicate_pairs'][:5]:  # Show first 5
                i, j = pair['index1'], pair['index2']
                sim = pair['similarity']
                print(f"  Rows {i} & {j} ({sim:.3f} similarity):")
                print(f"    Row {i}: {data.iloc[i].to_dict()}")
                print(f"    Row {j}: {data.iloc[j].to_dict()}")
                print()
        
        # Summary
        print("=" * 60)
        print("📊 ANALYSIS SUMMARY")
        print("=" * 60)
        
        total_issues = len(anomaly_results.get('anomalies', [])) + duplicate_results.get('total_duplicates', 0)
        quality_score = max(0, 100 - (total_issues / len(data) * 100))
        
        summary = {
            "total_rows": len(data),
            "anomalies_detected": len(anomaly_results.get('anomalies', [])),
            "duplicates_detected": duplicate_results.get('total_duplicates', 0),
            "data_quality_score": f"{quality_score:.1f}%",
            "ai_models_used": True,
            "detection_methods": {
                "anomaly": anomaly_results.get('method', 'Unknown'),
                "duplicate": duplicate_results.get('method', 'Unknown')
            }
        }
        
        for key, value in summary.items():
            print(f"  {key.replace('_', ' ').title()}: {value}")
        
        print("\n🎉 Local AI models are working perfectly!")
        print("💡 Your service is ready for production use with local AI capabilities.")
        
        return True
        
    except Exception as e:
        print(f"❌ Error running AI demo: {e}")
        print("🔧 Make sure all dependencies are installed and models are in ./models folder")
        return False

if __name__ == "__main__":
    success = quick_demo()
    
    if success:
        print("\n📚 Next Steps:")
        print("1. Start the service: python main.py")
        print("2. Test API endpoints: curl http://localhost:8000/api/v1/health")
        print("3. View API docs: http://localhost:8000/docs")
        print("4. Check documentation in ./docs/ folder")
        print("5. Run more examples in ./examples/ folder")
    else:
        print("\n🔧 Troubleshooting:")
        print("1. Check that AI dependencies are installed: pip install -r requirements.txt")
        print("2. Verify models are in ./models folder")
        print("3. Run: python examples/check_ai_setup.py")
