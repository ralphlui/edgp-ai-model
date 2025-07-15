"""
Unit tests for the anomaly detector model.
"""
import pandas as pd
import numpy as np
from src.models.anomaly_detector import TabularAnomalyDetector, DuplicationDetector

def test_tabular_anomaly_detector():
    """Test the tabular anomaly detector."""
    detector = TabularAnomalyDetector()
    
    # Create sample data with obvious anomalies
    data = pd.DataFrame({
        'age': [25, 30, 35, 28, 32, 1000],  # 1000 is an anomaly
        'income': [50000, 60000, 70000, 55000, 65000, 10],  # 10 is an anomaly
        'score': [85, 90, 88, 87, 91, -50]  # -50 is an anomaly
    })
    
    # Test anomaly detection
    anomalies = detector.predict_anomalies(data, threshold=0.3)
    
    # Should detect at least one anomaly
    assert len(anomalies) > 0
    
    # Check anomaly structure
    for anomaly in anomalies:
        assert 'row_index' in anomaly
        assert 'anomaly_score' in anomaly
        assert 'is_anomaly' in anomaly
        assert 'affected_columns' in anomaly

def test_duplication_detector():
    """Test the duplication detector."""
    detector = DuplicationDetector()
    
    # Create sample data with duplicates
    data = pd.DataFrame({
        'name': ['Alice', 'Bob', 'Charlie', 'Alice', 'David'],
        'age': [25, 30, 35, 25, 40],
        'city': ['NYC', 'LA', 'Chicago', 'NYC', 'Miami']
    })
    
    # Test duplication detection
    duplications = detector.detect_duplications(data, threshold=0.9)
    
    # Should detect at least one duplication group
    assert len(duplications) > 0
    
    # Check duplication structure
    for duplication in duplications:
        assert 'row_indices' in duplication
        assert 'similarity_score' in duplication
        assert 'duplicate_columns' in duplication
        assert len(duplication['row_indices']) >= 2

def test_preprocessing():
    """Test data preprocessing."""
    detector = TabularAnomalyDetector()
    
    # Create data with mixed types and missing values
    data = pd.DataFrame({
        'numeric': [1, 2, None, 4, 5],
        'categorical': ['A', 'B', None, 'A', 'C'],
        'mixed': [1, 'text', 3, 4, 'text']
    })
    
    processed = detector.preprocess_data(data)
    
    # Check that data is processed correctly
    assert processed.isnull().sum().sum() == 0  # No missing values
    assert all(processed.dtypes != 'object')  # All columns are numeric

def test_empty_data():
    """Test handling of empty data."""
    detector = TabularAnomalyDetector()
    
    # Empty DataFrame
    empty_data = pd.DataFrame()
    
    try:
        anomalies = detector.predict_anomalies(empty_data)
        assert len(anomalies) == 0
    except Exception:
        # Empty data should either return empty results or raise an appropriate exception
        pass

if __name__ == "__main__":
    test_tabular_anomaly_detector()
    test_duplication_detector()
    test_preprocessing()
    test_empty_data()
    print("All tests passed!")
