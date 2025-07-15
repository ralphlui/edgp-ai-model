"""
Data quality service for anomaly and duplication detection.
"""
import logging
import time
import pandas as pd
from typing import List, Dict, Any
from ..models.anomaly_detector import EnhancedTabularAnomalyDetector, EnhancedDuplicationDetector
from ..models.schemas import (
    DataQualityRequest, 
    DataQualityResponse, 
    DataQualityCheckType,
    AnomalyResult,
    DuplicationResult
)
from config.settings import settings

logger = logging.getLogger(__name__)

class DataQualityService:
    """Service for performing data quality checks."""
    
    def __init__(self):
        """Initialize the data quality service with enhanced AI detectors."""
        self.anomaly_detector = EnhancedTabularAnomalyDetector(
            model_cache_dir=settings.model_cache_dir,
            use_ai=settings.use_local_models
        )
        self.duplication_detector = EnhancedDuplicationDetector(
            model_cache_dir=settings.model_cache_dir,
            use_ai=settings.use_local_models
        )
        logger.info("Data quality service initialized")
    
    async def analyze_data_quality(self, request: DataQualityRequest) -> DataQualityResponse:
        """Analyze data quality based on the request."""
        start_time = time.time()
        
        try:
            # Convert data to DataFrame
            df = pd.DataFrame(request.data)
            
            if df.empty:
                raise ValueError("No data provided for analysis")
            
            # Filter columns if specified
            if request.columns_to_check:
                missing_columns = set(request.columns_to_check) - set(df.columns)
                if missing_columns:
                    raise ValueError(f"Columns not found in data: {missing_columns}")
                df = df[request.columns_to_check]
            
            # Initialize results
            anomalies = []
            duplications = []
            
            # Perform anomaly detection
            if request.check_type in [DataQualityCheckType.ANOMALY, DataQualityCheckType.BOTH]:
                anomaly_threshold = request.anomaly_threshold or settings.anomaly_threshold
                anomaly_results = self.anomaly_detector.predict_anomalies(df, anomaly_threshold)
                
                anomalies = [
                    AnomalyResult(
                        row_index=result["row_index"],
                        anomaly_score=result["anomaly_score"],
                        is_anomaly=result["is_anomaly"],
                        affected_columns=result["affected_columns"]
                    )
                    for result in anomaly_results
                ]
            
            # Perform duplication detection
            if request.check_type in [DataQualityCheckType.DUPLICATION, DataQualityCheckType.BOTH]:
                duplication_threshold = request.duplication_threshold or settings.duplication_threshold
                duplication_results = self.duplication_detector.detect_duplications(df, duplication_threshold)
                
                duplications = [
                    DuplicationResult(
                        row_indices=result["row_indices"],
                        similarity_score=result["similarity_score"],
                        duplicate_columns=result["duplicate_columns"]
                    )
                    for result in duplication_results
                ]
            
            # Calculate summary statistics
            summary = self._generate_summary(df, anomalies, duplications, request.check_type)
            
            processing_time = time.time() - start_time
            
            return DataQualityResponse(
                total_rows=len(df),
                check_type=request.check_type,
                anomalies=anomalies,
                duplications=duplications,
                summary=summary,
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Error in data quality analysis: {str(e)}")
            raise
    
    def _generate_summary(self, df: pd.DataFrame, anomalies: List[AnomalyResult], 
                         duplications: List[DuplicationResult], 
                         check_type: DataQualityCheckType) -> Dict[str, Any]:
        """Generate summary statistics for the analysis."""
        summary = {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "columns": df.columns.tolist()
        }
        
        if check_type in [DataQualityCheckType.ANOMALY, DataQualityCheckType.BOTH]:
            summary.update({
                "anomaly_count": len(anomalies),
                "anomaly_percentage": (len(anomalies) / len(df)) * 100 if len(df) > 0 else 0,
                "avg_anomaly_score": sum(a.anomaly_score for a in anomalies) / len(anomalies) if anomalies else 0
            })
        
        if check_type in [DataQualityCheckType.DUPLICATION, DataQualityCheckType.BOTH]:
            total_duplicate_rows = sum(len(d.row_indices) for d in duplications)
            summary.update({
                "duplication_groups": len(duplications),
                "duplicate_rows": total_duplicate_rows,
                "duplication_percentage": (total_duplicate_rows / len(df)) * 100 if len(df) > 0 else 0,
                "avg_similarity_score": sum(d.similarity_score for d in duplications) / len(duplications) if duplications else 0
            })
        
        # Basic data statistics
        summary["data_types"] = df.dtypes.astype(str).to_dict()
        summary["missing_values"] = df.isnull().sum().to_dict()
        summary["missing_percentage"] = (df.isnull().sum() / len(df) * 100).to_dict()
        
        return summary
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform a health check of the service."""
        try:
            # Test with a small dataset
            test_data = pd.DataFrame({
                'A': [1, 2, 3, 4, 5],
                'B': ['a', 'b', 'c', 'd', 'e']
            })
            
            # Quick test of anomaly detection
            self.anomaly_detector.predict_anomalies(test_data)
            
            # Quick test of duplication detection
            self.duplication_detector.detect_duplications(test_data)
            
            return {
                "status": "healthy",
                "anomaly_detector": "ready",
                "duplication_detector": "ready",
                "model_loaded": True
            }
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "model_loaded": False
            }
