#!/usr/bin/env python3
"""
Demo script for the EDGP AI Model Service.
This script demonstrates how to use the data quality service programmatically.
"""
import requests
import json
import time

BASE_URL = "http://127.0.0.1:8000"

def test_health_check():
    """Test the health check endpoint."""
    print("🔍 Testing health check...")
    response = requests.get(f"{BASE_URL}/api/v1/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()

def test_data_analysis():
    """Test data quality analysis with sample data."""
    print("🔍 Testing data quality analysis...")
    
    # Sample data with anomalies and duplicates
    sample_data = {
        "data": [
            {"employee_id": 1, "age": 25, "salary": 50000, "department": "Engineering", "performance": 8.5},
            {"employee_id": 2, "age": 30, "salary": 60000, "department": "Marketing", "performance": 9.0},
            {"employee_id": 3, "age": 35, "salary": 70000, "department": "Engineering", "performance": 8.8},
            {"employee_id": 4, "age": 28, "salary": 55000, "department": "Sales", "performance": 8.7},
            {"employee_id": 5, "age": 32, "salary": 65000, "department": "Marketing", "performance": 9.1},
            # Anomaly - very high age and salary
            {"employee_id": 6, "age": 999, "salary": 9999999, "department": "Unknown", "performance": -5.0},
            # Duplicate
            {"employee_id": 7, "age": 25, "salary": 50000, "department": "Engineering", "performance": 8.5},
            # Another anomaly - negative salary
            {"employee_id": 8, "age": 45, "salary": -10000, "department": "Finance", "performance": 10.5}
        ],
        "check_type": "both",
        "anomaly_threshold": 0.3,
        "duplication_threshold": 0.9
    }
    
    response = requests.post(f"{BASE_URL}/api/v1/analyze", json=sample_data)
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Analysis completed successfully!")
        print(f"📊 Total rows analyzed: {result['total_rows']}")
        print(f"⚠️  Anomalies found: {len(result['anomalies'])}")
        print(f"🔄 Duplication groups: {len(result['duplications'])}")
        print(f"⏱️  Processing time: {result['processing_time']:.3f} seconds")
        
        if result['anomalies']:
            print("\n🚨 Detected Anomalies:")
            for anomaly in result['anomalies']:
                print(f"  - Row {anomaly['row_index']}: Score {anomaly['anomaly_score']:.3f}, "
                      f"Affected columns: {anomaly['affected_columns']}")
        
        if result['duplications']:
            print("\n🔄 Detected Duplications:")
            for dup in result['duplications']:
                print(f"  - Rows {dup['row_indices']}: Similarity {dup['similarity_score']:.3f}")
        
        print(f"\n📈 Summary:")
        summary = result['summary']
        print(f"  - Anomaly rate: {summary['anomaly_percentage']:.1f}%")
        print(f"  - Duplication rate: {summary['duplication_percentage']:.1f}%")
    else:
        print(f"❌ Error: {response.text}")
    
    print()

def test_file_upload():
    """Test file upload analysis."""
    print("🔍 Testing file upload analysis...")
    
    try:
        with open("tests/sample_data.csv", "rb") as f:
            files = {"file": ("sample_data.csv", f, "text/csv")}
            data = {
                "check_type": "both",
                "anomaly_threshold": "0.4"
            }
            
            response = requests.post(f"{BASE_URL}/api/v1/analyze-file", files=files, data=data)
            
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ File analysis completed successfully!")
            print(f"📊 Total rows: {result['total_rows']}")
            print(f"⚠️  Anomalies: {len(result['anomalies'])}")
            print(f"🔄 Duplications: {len(result['duplications'])}")
        else:
            print(f"❌ Error: {response.text}")
    except FileNotFoundError:
        print("❌ Sample file not found. Please ensure tests/sample_data.csv exists.")
    
    print()

def test_service_info():
    """Test service information endpoint."""
    print("🔍 Testing service information...")
    response = requests.get(f"{BASE_URL}/api/v1/info")
    print(f"Status: {response.status_code}")
    print(f"Service Info: {json.dumps(response.json(), indent=2)}")
    print()

def main():
    """Run all tests."""
    print("🚀 EDGP AI Model Service Demo")
    print("=" * 50)
    
    try:
        test_health_check()
        test_service_info()
        test_data_analysis()
        test_file_upload()
        
        print("✅ All tests completed!")
        
    except requests.exceptions.ConnectionError:
        print("❌ Error: Cannot connect to the service.")
        print("Make sure the service is running on http://127.0.0.1:8000")
        print("Run: python main.py")

if __name__ == "__main__":
    main()
