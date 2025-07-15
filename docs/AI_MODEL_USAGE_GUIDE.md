# 🤖 Using Local AI Models with EDGP Service

## Step 4: Enhanced Anomaly Detector with Local AI Models

This enhanced detector uses locally downloaded Hugging Face models for better anomaly detection.

```python
import torch
import pandas as pd
import numpy as np
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)

class HuggingFaceAnomalyDetector:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", cache_dir: str = "./models"):
        """
        Initialize with a Hugging Face model for better anomaly detection.
        
        Args:
            model_name: Name of the Hugging Face model to use
            cache_dir: Directory to cache downloaded models
        """
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.model = None
        self.tokenizer = None
        self.scaler = StandardScaler()
        self.is_ready = False
        
        self._load_model()
    
    def _load_model(self):
        """Load the Hugging Face model and tokenizer."""
        try:
            logger.info(f"Loading model: {self.model_name}")
            
            # Load model and tokenizer
            self.model = AutoModel.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
                trust_remote_code=True
            )
            
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    cache_dir=self.cache_dir
                )
            except:
                logger.warning(f"No tokenizer available for {self.model_name}")
                self.tokenizer = None
            
            # Set to evaluation mode
            self.model.eval()
            self.is_ready = True
            
            logger.info(f"✅ Model {self.model_name} loaded successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to load model {self.model_name}: {str(e)}")
            self.is_ready = False
    
    def _get_embeddings(self, data: pd.DataFrame) -> np.ndarray:
        """Convert dataframe to embeddings using the AI model."""
        if not self.is_ready:
            raise ValueError("Model not ready. Check model loading.")
        
        # Handle different data types
        embeddings_list = []
        
        for _, row in data.iterrows():
            # Convert row to text representation
            text_repr = self._row_to_text(row)
            
            # Get embeddings
            if self.tokenizer:
                # Use tokenizer if available
                inputs = self.tokenizer(
                    text_repr,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512
                )
                
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    # Use mean pooling of last hidden states
                    embedding = outputs.last_hidden_state.mean(dim=1).squeeze()
            else:
                # Fallback: use numeric data directly
                numeric_data = row.select_dtypes(include=[np.number]).values
                if len(numeric_data) > 0:
                    # Convert to tensor and get model representation
                    input_tensor = torch.tensor(numeric_data, dtype=torch.float32).unsqueeze(0)
                    
                    with torch.no_grad():
                        try:
                            outputs = self.model(input_tensor)
                            embedding = outputs.last_hidden_state.mean(dim=1).squeeze() if hasattr(outputs, 'last_hidden_state') else outputs.squeeze()
                        except:
                            # If model doesn't accept this input, use the data as-is
                            embedding = torch.tensor(numeric_data, dtype=torch.float32)
                else:
                    # Fallback to zero embedding
                    embedding = torch.zeros(384)  # Common embedding size
            
            embeddings_list.append(embedding.numpy())
        
        return np.array(embeddings_list)
    
    def _row_to_text(self, row: pd.Series) -> str:
        """Convert a dataframe row to text representation."""
        text_parts = []
        for col, val in row.items():
            text_parts.append(f"{col}: {val}")
        return " | ".join(text_parts)
    
    def detect_anomalies(self, data: pd.DataFrame, contamination: float = 0.1) -> Dict[str, Any]:
        """
        Detect anomalies using AI model embeddings and distance-based analysis.
        
        Args:
            data: Input dataframe
            contamination: Expected proportion of anomalies
            
        Returns:
            Dictionary with anomaly detection results
        """
        if not self.is_ready:
            logger.error("Model not ready for anomaly detection")
            return {"error": "Model not loaded", "anomalies": [], "scores": []}
        
        try:
            # Get AI model embeddings
            embeddings = self._get_embeddings(data)
            
            # Calculate similarity matrix
            similarity_matrix = cosine_similarity(embeddings)
            
            # Calculate anomaly scores based on average similarity to other points
            anomaly_scores = []
            for i in range(len(similarity_matrix)):
                # Exclude self-similarity (diagonal)
                similarities = np.concatenate([
                    similarity_matrix[i][:i],
                    similarity_matrix[i][i+1:]
                ])
                # Lower average similarity = higher anomaly score
                avg_similarity = np.mean(similarities)
                anomaly_score = 1 - avg_similarity
                anomaly_scores.append(anomaly_score)
            
            # Determine anomalies based on threshold
            threshold = np.percentile(anomaly_scores, (1 - contamination) * 100)
            anomalies = [i for i, score in enumerate(anomaly_scores) if score > threshold]
            
            return {
                "anomalies": anomalies,
                "scores": anomaly_scores,
                "threshold": threshold,
                "method": f"AI-based using {self.model_name}",
                "contamination": contamination
            }
            
        except Exception as e:
            logger.error(f"Error in AI anomaly detection: {str(e)}")
            return {"error": str(e), "anomalies": [], "scores": []}
    
    def detect_duplicates(self, data: pd.DataFrame, similarity_threshold: float = 0.95) -> Dict[str, Any]:
        """
        Detect duplicates using AI model embeddings.
        
        Args:
            data: Input dataframe
            similarity_threshold: Minimum similarity to consider duplicates
            
        Returns:
            Dictionary with duplicate detection results
        """
        if not self.is_ready:
            logger.error("Model not ready for duplicate detection")
            return {"error": "Model not loaded", "duplicates": []}
        
        try:
            # Get AI model embeddings
            embeddings = self._get_embeddings(data)
            
            # Calculate similarity matrix
            similarity_matrix = cosine_similarity(embeddings)
            
            # Find duplicate pairs
            duplicate_pairs = []
            duplicate_groups = []
            
            for i in range(len(similarity_matrix)):
                for j in range(i + 1, len(similarity_matrix)):
                    if similarity_matrix[i][j] >= similarity_threshold:
                        duplicate_pairs.append({
                            "index1": i,
                            "index2": j,
                            "similarity": similarity_matrix[i][j]
                        })
            
            # Group duplicates
            processed = set()
            for pair in duplicate_pairs:
                if pair["index1"] not in processed and pair["index2"] not in processed:
                    group = [pair["index1"], pair["index2"]]
                    processed.add(pair["index1"])
                    processed.add(pair["index2"])
                    
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
            
            return {
                "duplicate_pairs": duplicate_pairs,
                "duplicate_groups": duplicate_groups,
                "similarity_threshold": similarity_threshold,
                "method": f"AI-based using {self.model_name}",
                "total_duplicates": len(set([idx for pair in duplicate_pairs for idx in [pair["index1"], pair["index2"]]]))
            }
            
        except Exception as e:
            logger.error(f"Error in AI duplicate detection: {str(e)}")
            return {"error": str(e), "duplicates": []}

# Enhanced Data Quality Service with AI Models
class EnhancedDataQualityService:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """Initialize with AI-powered detectors."""
        self.ai_detector = HuggingFaceAnomalyDetector(model_name=model_name)
        # Keep fallback to sklearn-based detector
        from .anomaly_detector import TabularAnomalyDetector, DuplicationDetector
        self.fallback_anomaly_detector = TabularAnomalyDetector()
        self.fallback_duplicate_detector = DuplicationDetector()
    
    def analyze_data_quality(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Comprehensive data quality analysis using AI models with fallback.
        """
        results = {
            "timestamp": pd.Timestamp.now().isoformat(),
            "data_shape": data.shape,
            "ai_analysis": {},
            "fallback_analysis": {}
        }
        
        # Try AI-based analysis first
        if self.ai_detector.is_ready:
            try:
                ai_anomalies = self.ai_detector.detect_anomalies(data)
                ai_duplicates = self.ai_detector.detect_duplicates(data)
                
                results["ai_analysis"] = {
                    "anomalies": ai_anomalies,
                    "duplicates": ai_duplicates,
                    "model_used": self.ai_detector.model_name,
                    "status": "success"
                }
            except Exception as e:
                results["ai_analysis"] = {
                    "status": "error",
                    "error": str(e)
                }
        else:
            results["ai_analysis"] = {
                "status": "model_not_ready",
                "message": "AI model not available, using fallback"
            }
        
        # Fallback analysis
        try:
            fallback_anomalies = self.fallback_anomaly_detector.detect_anomalies(data)
            fallback_duplicates = self.fallback_duplicate_detector.detect_duplicates(data)
            
            results["fallback_analysis"] = {
                "anomalies": fallback_anomalies,
                "duplicates": fallback_duplicates,
                "method": "sklearn-based",
                "status": "success"
            }
        except Exception as e:
            results["fallback_analysis"] = {
                "status": "error",
                "error": str(e)
            }
        
        # Summary
        primary_analysis = results.get("ai_analysis", {}) if results["ai_analysis"].get("status") == "success" else results.get("fallback_analysis", {})
        
        if primary_analysis.get("status") == "success":
            anomaly_count = len(primary_analysis.get("anomalies", {}).get("anomalies", []))
            duplicate_count = primary_analysis.get("duplicates", {}).get("total_duplicates", 0)
            
            results["summary"] = {
                "total_rows": len(data),
                "anomalies_detected": anomaly_count,
                "duplicates_detected": duplicate_count,
                "quality_score": max(0, 100 - (anomaly_count + duplicate_count) / len(data) * 100),
                "primary_method": "AI-powered" if results["ai_analysis"].get("status") == "success" else "sklearn-based"
            }
        else:
            results["summary"] = {
                "status": "analysis_failed",
                "message": "Both AI and fallback analysis failed"
            }
        
        return results
```
