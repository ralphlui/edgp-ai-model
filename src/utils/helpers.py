"""
Utility functions for the EDGP AI Model service.
"""
import logging
import pandas as pd
from typing import Dict, Any, List, Union
import json

def setup_logging(log_level: str = "INFO", log_format: str = None) -> None:
    """Set up logging configuration."""
    if log_format is None:
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("app.log")
        ]
    )

def validate_data_format(data: Union[List[Dict], pd.DataFrame]) -> pd.DataFrame:
    """Validate and convert data to DataFrame format."""
    if isinstance(data, list):
        if not data:
            raise ValueError("Data list is empty")
        if not all(isinstance(item, dict) for item in data):
            raise ValueError("All items in data list must be dictionaries")
        return pd.DataFrame(data)
    elif isinstance(data, pd.DataFrame):
        if data.empty:
            raise ValueError("DataFrame is empty")
        return data
    else:
        raise ValueError("Data must be a list of dictionaries or a pandas DataFrame")

def sanitize_data(df: pd.DataFrame) -> pd.DataFrame:
    """Sanitize data for processing."""
    # Remove completely empty rows
    df = df.dropna(how='all')
    
    # Convert infinite values to NaN
    df = df.replace([float('inf'), float('-inf')], pd.NA)
    
    # Ensure column names are strings
    df.columns = df.columns.astype(str)
    
    return df

def calculate_data_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate basic statistics for the dataset."""
    stats = {
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "memory_usage": df.memory_usage(deep=True).sum(),
        "missing_values": df.isnull().sum().to_dict(),
        "missing_percentage": (df.isnull().sum() / len(df) * 100).to_dict()
    }
    
    # Numerical statistics
    numeric_cols = df.select_dtypes(include=['number']).columns
    if len(numeric_cols) > 0:
        stats["numeric_summary"] = df[numeric_cols].describe().to_dict()
    
    # Categorical statistics
    categorical_cols = df.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        stats["categorical_summary"] = {}
        for col in categorical_cols:
            stats["categorical_summary"][col] = {
                "unique_count": df[col].nunique(),
                "most_frequent": df[col].mode().iloc[0] if not df[col].mode().empty else None,
                "frequency": df[col].value_counts().head(5).to_dict()
            }
    
    return stats

def format_response_for_json(response: Dict[str, Any]) -> str:
    """Format response for JSON serialization."""
    def convert_numpy(obj):
        """Convert numpy types to Python types."""
        if hasattr(obj, 'tolist'):
            return obj.tolist()
        elif hasattr(obj, 'item'):
            return obj.item()
        elif isinstance(obj, (pd.Timestamp, pd.Timedelta)):
            return str(obj)
        return obj
    
    return json.dumps(response, default=convert_numpy, indent=2)

def chunk_data(data: pd.DataFrame, chunk_size: int = 1000) -> List[pd.DataFrame]:
    """Split large datasets into smaller chunks for processing."""
    chunks = []
    for i in range(0, len(data), chunk_size):
        chunks.append(data.iloc[i:i + chunk_size])
    return chunks

def merge_analysis_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Merge results from multiple data chunks."""
    if not results:
        return {}
    
    merged = {
        "anomalies": [],
        "duplications": [],
        "total_rows": 0,
        "processing_time": 0
    }
    
    row_offset = 0
    for result in results:
        # Adjust row indices for anomalies
        for anomaly in result.get("anomalies", []):
            anomaly["row_index"] += row_offset
            merged["anomalies"].append(anomaly)
        
        # Adjust row indices for duplications
        for duplication in result.get("duplications", []):
            duplication["row_indices"] = [idx + row_offset for idx in duplication["row_indices"]]
            merged["duplications"].append(duplication)
        
        merged["total_rows"] += result.get("total_rows", 0)
        merged["processing_time"] += result.get("processing_time", 0)
        
        row_offset += result.get("total_rows", 0)
    
    return merged
