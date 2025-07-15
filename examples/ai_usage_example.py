#!/usr/bin/env python3
"""
🚀 Complete AI Model Usage Example
This script demonstrates how to use the downloaded AI model for anomaly and duplicate detection.
"""

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json
from typing import Dict, List, Any

class AIAnomalyDetector:
    """AI-powered anomaly detector using Hugging Face models."""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """Initialize with a pre-trained model."""
        print(f"🤖 Loading AI model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        print(f"✅ Model loaded successfully!")
    
    def data_to_text(self, data: pd.DataFrame) -> List[str]:
        """Convert dataframe rows to text representations."""
        texts = []
        for _, row in data.iterrows():
            # Create a natural language representation of each row
            text_parts = []
            for col, val in row.items():
                text_parts.append(f"{col}: {val}")
            text = " | ".join(text_parts)
            texts.append(text)
        return texts
    
    def detect_anomalies(self, data: pd.DataFrame, contamination: float = 0.1) -> Dict[str, Any]:
        """
        Detect anomalies using AI embeddings.
        
        Args:
            data: Input dataframe
            contamination: Expected proportion of anomalies (0.1 = 10%)
        
        Returns:
            Dictionary with anomaly detection results
        """
        print(f"🔍 Analyzing {len(data)} rows for anomalies...")
        
        # Convert data to text
        texts = self.data_to_text(data)
        
        # Get AI embeddings
        embeddings = self.model.encode(texts)
        print(f"📊 Generated embeddings: {embeddings.shape}")
        
        # Calculate similarity matrix
        similarity_matrix = cosine_similarity(embeddings)
        
        # Calculate anomaly scores
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
        
        # Determine threshold and anomalies
        threshold = np.percentile(anomaly_scores, (1 - contamination) * 100)
        anomalies = [i for i, score in enumerate(anomaly_scores) if score > threshold]
        
        print(f"🚨 Found {len(anomalies)} anomalies (threshold: {threshold:.3f})")
        
        return {
            "anomalies": anomalies,
            "scores": anomaly_scores,
            "threshold": threshold,
            "method": f"AI-based using {self.model_name}",
            "contamination": contamination
        }
    
    def detect_duplicates(self, data: pd.DataFrame, similarity_threshold: float = 0.95) -> Dict[str, Any]:
        """
        Detect near-duplicates using AI embeddings.
        
        Args:
            data: Input dataframe
            similarity_threshold: Minimum similarity to consider duplicates (0.95 = 95%)
        
        Returns:
            Dictionary with duplicate detection results
        """
        print(f"🔍 Analyzing {len(data)} rows for duplicates...")
        
        # Convert data to text and get embeddings
        texts = self.data_to_text(data)
        embeddings = self.model.encode(texts)
        
        # Calculate similarity matrix
        similarity_matrix = cosine_similarity(embeddings)
        
        # Find duplicate pairs
        duplicate_pairs = []
        for i in range(len(similarity_matrix)):
            for j in range(i + 1, len(similarity_matrix)):
                similarity = similarity_matrix[i][j]
                if similarity >= similarity_threshold:
                    duplicate_pairs.append({
                        "index1": i,
                        "index2": j,
                        "similarity": similarity
                    })
        
        # Group duplicates
        duplicate_groups = []
        processed = set()
        
        for pair in duplicate_pairs:
            if pair["index1"] not in processed and pair["index2"] not in processed:
                group = [pair["index1"], pair["index2"]]
                processed.update(group)
                
                # Find other members of this group
                for other_pair in duplicate_pairs:
                    if (other_pair["index1"] in group or other_pair["index2"] in group):
                        if other_pair["index1"] not in group:
                            group.append(other_pair["index1"])
                            processed.add(other_pair["index1"])
                        if other_pair["index2"] not in group:
                            group.append(other_pair["index2"])
                            processed.add(other_pair["index2"])
                
                duplicate_groups.append(sorted(group))
        
        total_duplicates = len(set([idx for pair in duplicate_pairs for idx in [pair["index1"], pair["index2"]]]))
        print(f"👥 Found {len(duplicate_pairs)} duplicate pairs, {total_duplicates} total duplicate records")
        
        return {
            "duplicate_pairs": duplicate_pairs,
            "duplicate_groups": duplicate_groups,
            "total_duplicates": total_duplicates,
            "similarity_threshold": similarity_threshold,
            "method": f"AI-based using {self.model_name}"
        }

def create_sample_data() -> pd.DataFrame:
    """Create sample data with clear anomalies and duplicates."""
    data = pd.DataFrame({
        'user_id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'age': [25, 30, 35, 28, 999, 32, 29, 30, 31, 28],  # 999 is anomaly
        'salary': [50000, 60000, 70000, 55000, 999999, 62000, 58000, 60000, 61000, 55000],  # 999999 is anomaly
        'department': ['Engineering', 'Marketing', 'Sales', 'Engineering', 'Unknown', 'Marketing', 'Sales', 'Marketing', 'Engineering', 'Engineering'],
        'score': [85, 90, 88, 87, -50, 89, 86, 90, 88, 87],  # -50 is anomaly
        'location': ['NYC', 'LA', 'Chicago', 'NYC', 'Mars', 'LA', 'Chicago', 'LA', 'NYC', 'NYC']  # Mars is anomaly
    })
    
    # Add some duplicate-like records
    duplicate_row = data.iloc[1].copy()  # Copy row with user_id=2
    duplicate_row['user_id'] = 11
    data = pd.concat([data, duplicate_row.to_frame().T], ignore_index=True)
    
    return data

def demonstrate_ai_detection():
    """Demonstrate AI-powered anomaly and duplicate detection."""
    print("🎯 AI-Powered Data Quality Analysis Demo")
    print("=" * 60)
    
    # Create sample data
    data = create_sample_data()
    print(f"📊 Sample dataset with {len(data)} rows:")
    print(data.to_string())
    
    # Initialize AI detector
    detector = AIAnomalyDetector()
    
    print("\n" + "=" * 60)
    print("🔍 ANOMALY DETECTION")
    print("=" * 60)
    
    # Detect anomalies
    anomaly_results = detector.detect_anomalies(data, contamination=0.2)  # Expect 20% anomalies
    
    print(f"\n📈 Anomaly Scores:")
    for i, score in enumerate(anomaly_results["scores"]):
        status = "🚨 ANOMALY" if i in anomaly_results["anomalies"] else "✅ Normal"
        print(f"  Row {i}: {score:.3f} - {status}")
    
    print(f"\n🚨 Detected Anomalies:")
    for idx in anomaly_results["anomalies"]:
        print(f"  Row {idx}: {data.iloc[idx].to_dict()}")
    
    print("\n" + "=" * 60)
    print("👥 DUPLICATE DETECTION")
    print("=" * 60)
    
    # Detect duplicates
    duplicate_results = detector.detect_duplicates(data, similarity_threshold=0.9)
    
    if duplicate_results["duplicate_pairs"]:
        print(f"\n👥 Detected Duplicate Pairs:")
        for pair in duplicate_results["duplicate_pairs"]:
            i, j = pair["index1"], pair["index2"]
            sim = pair["similarity"]
            print(f"  Rows {i} and {j} ({sim:.3f} similarity):")
            print(f"    Row {i}: {data.iloc[i].to_dict()}")
            print(f"    Row {j}: {data.iloc[j].to_dict()}")
    else:
        print("No duplicates found with current threshold.")
    
    print("\n" + "=" * 60)
    print("📊 SUMMARY")
    print("=" * 60)
    
    quality_score = max(0, 100 - (len(anomaly_results["anomalies"]) + duplicate_results["total_duplicates"]) / len(data) * 100)
    
    summary = {
        "total_rows": len(data),
        "anomalies_detected": len(anomaly_results["anomalies"]),
        "duplicates_detected": duplicate_results["total_duplicates"],
        "data_quality_score": f"{quality_score:.1f}%",
        "model_used": detector.model_name
    }
    
    print(json.dumps(summary, indent=2))
    
    return anomaly_results, duplicate_results

def test_with_custom_data():
    """Example of how to use with your own data."""
    print("\n" + "=" * 60)
    print("💡 USING WITH YOUR OWN DATA")
    print("=" * 60)
    
    # Example: Load your own CSV file
    # your_data = pd.read_csv("your_data.csv")
    
    # For demo, create financial transaction data
    financial_data = pd.DataFrame({
        'transaction_id': range(1, 8),
        'amount': [100.0, 250.0, 75.5, 999999.0, 150.0, 200.0, 100.0],  # 999999 is anomaly
        'merchant': ['Store A', 'Store B', 'Store C', 'Hacker Store', 'Store A', 'Store D', 'Store A'],
        'category': ['Grocery', 'Clothing', 'Gas', 'Unknown', 'Grocery', 'Restaurant', 'Grocery'],
        'location': ['NYC', 'LA', 'Chicago', 'Nowhere', 'NYC', 'Miami', 'NYC']
    })
    
    print("💳 Financial Transaction Data:")
    print(financial_data.to_string())
    
    detector = AIAnomalyDetector()
    
    # Detect anomalies in financial data
    print("\n🔍 Analyzing financial transactions for anomalies...")
    anomalies = detector.detect_anomalies(financial_data, contamination=0.15)
    
    print(f"\n🚨 Suspicious Transactions:")
    for idx in anomalies["anomalies"]:
        score = anomalies["scores"][idx]
        transaction = financial_data.iloc[idx]
        print(f"  Transaction {transaction['transaction_id']}: Anomaly Score {score:.3f}")
        print(f"    Amount: ${transaction['amount']}, Merchant: {transaction['merchant']}")
        print(f"    Category: {transaction['category']}, Location: {transaction['location']}")

if __name__ == "__main__":
    print("🚀 AI Model Usage Examples")
    print("This demonstrates how to use downloaded AI models for data quality analysis.\n")
    
    try:
        # Main demonstration
        anomaly_results, duplicate_results = demonstrate_ai_detection()
        
        # Custom data example
        test_with_custom_data()
        
        print("\n🎉 Demo completed successfully!")
        print("\n💡 Key takeaways:")
        print("  • AI models can detect subtle patterns in data")
        print("  • Models work with any tabular data (numeric, text, mixed)")
        print("  • Models are cached locally (~/.cache/huggingface/)")
        print("  • You can adjust thresholds based on your needs")
        print("  • Integration with FastAPI service is straightforward")
        
    except Exception as e:
        print(f"❌ Error in demo: {str(e)}")
        print("Make sure all AI packages are installed correctly.")
