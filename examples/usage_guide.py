#!/usr/bin/env python3
"""
Complete Usage Guide for EDGP AI Model Service
This script demonstrates all the ways to use the AI model service.
"""

import requests
import json
import pandas as pd
import time
from typing import Dict, Any

class EDGPAIModelClient:
    """Client for interacting with the EDGP AI Model Service."""
    
    def __init__(self, base_url: str = "http://127.0.0.1:8000"):
        self.base_url = base_url
        self.api_base = f"{base_url}/api/v1"
    
    def check_health(self) -> Dict[str, Any]:
        """Check if the service is healthy."""
        response = requests.get(f"{self.api_base}/health")
        return response.json()
    
    def get_service_info(self) -> Dict[str, Any]:
        """Get service information."""
        response = requests.get(f"{self.api_base}/info")
        return response.json()
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get current AI model information."""
        response = requests.get(f"{self.api_base}/model-info")
        return response.json()
    
    def analyze_data(self, data, check_type="both", anomaly_threshold=None, duplication_threshold=None):
        """Analyze data for quality issues."""
        payload = {
            "data": data,
            "check_type": check_type
        }
        
        if anomaly_threshold is not None:
            payload["anomaly_threshold"] = anomaly_threshold
        if duplication_threshold is not None:
            payload["duplication_threshold"] = duplication_threshold
        
        response = requests.post(f"{self.api_base}/analyze", json=payload)
        return response.json()
    
    def analyze_file(self, file_path, check_type="both", anomaly_threshold=None):
        """Analyze a CSV or JSON file."""
        with open(file_path, 'rb') as f:
            files = {"file": f}
            data = {"check_type": check_type}
            
            if anomaly_threshold is not None:
                data["anomaly_threshold"] = str(anomaly_threshold)
            
            response = requests.post(f"{self.api_base}/analyze-file", files=files, data=data)
            return response.json()

def print_section(title: str):
    """Print a formatted section header."""
    print(f"\n{'='*60}")
    print(f"🔍 {title}")
    print(f"{'='*60}")

def print_results(result: Dict[str, Any]):
    """Print analysis results in a formatted way."""
    print(f"📊 Total rows analyzed: {result.get('total_rows', 0)}")
    print(f"⚠️  Anomalies found: {len(result.get('anomalies', []))}")
    print(f"🔄 Duplication groups: {len(result.get('duplications', []))}")
    print(f"⏱️  Processing time: {result.get('processing_time', 0):.3f} seconds")
    
    # Show anomalies
    if result.get('anomalies'):
        print(f"\n🚨 Anomaly Details:")
        for i, anomaly in enumerate(result['anomalies'][:3]):  # Show first 3
            print(f"  {i+1}. Row {anomaly['row_index']}: Score {anomaly['anomaly_score']:.3f}")
            print(f"     Affected columns: {anomaly['affected_columns']}")
    
    # Show duplications
    if result.get('duplications'):
        print(f"\n🔄 Duplication Details:")
        for i, dup in enumerate(result['duplications'][:3]):  # Show first 3
            print(f"  {i+1}. Rows {dup['row_indices']}: Similarity {dup['similarity_score']:.3f}")
    
    # Show summary
    if result.get('summary'):
        summary = result['summary']
        print(f"\n📈 Summary:")
        print(f"  - Anomaly rate: {summary.get('anomaly_percentage', 0):.1f}%")
        print(f"  - Duplication rate: {summary.get('duplication_percentage', 0):.1f}%")

def example_1_basic_usage():
    """Example 1: Basic data quality analysis."""
    print_section("Example 1: Basic Data Quality Analysis")
    
    client = EDGPAIModelClient()
    
    # Sample employee data with anomalies and duplicates
    employee_data = [
        {"id": 1, "name": "Alice", "age": 25, "salary": 50000, "department": "Engineering"},
        {"id": 2, "name": "Bob", "age": 30, "salary": 60000, "department": "Marketing"},
        {"id": 3, "name": "Charlie", "age": 35, "salary": 70000, "department": "Engineering"},
        {"id": 4, "name": "Diana", "age": 28, "salary": 55000, "department": "Sales"},
        # Anomaly: Very high age and salary
        {"id": 5, "name": "Outlier", "age": 999, "salary": 9999999, "department": "Unknown"},
        # Duplicate of Alice
        {"id": 6, "name": "Alice", "age": 25, "salary": 50000, "department": "Engineering"},
        # Another anomaly: Negative salary
        {"id": 7, "name": "Error", "age": 45, "salary": -10000, "department": "Finance"}
    ]
    
    print("Analyzing employee data...")
    result = client.analyze_data(employee_data, check_type="both")
    print_results(result)

def example_2_anomaly_only():
    """Example 2: Anomaly detection only."""
    print_section("Example 2: Anomaly Detection Only")
    
    client = EDGPAIModelClient()
    
    # Financial transaction data
    transaction_data = [
        {"amount": 100.50, "merchant": "Grocery Store", "category": "Food"},
        {"amount": 50.25, "merchant": "Gas Station", "category": "Transport"},
        {"amount": 200.00, "merchant": "Restaurant", "category": "Food"},
        {"amount": 1500.00, "merchant": "Electronics Store", "category": "Shopping"},
        # Suspicious transactions (anomalies)
        {"amount": 50000.00, "merchant": "Unknown", "category": "Transfer"},
        {"amount": -500.00, "merchant": "Refund", "category": "Error"},
    ]
    
    print("Analyzing transaction data for anomalies...")
    result = client.analyze_data(
        transaction_data, 
        check_type="anomaly",
        anomaly_threshold=0.3  # Lower threshold = more sensitive
    )
    print_results(result)

def example_3_duplication_only():
    """Example 3: Duplication detection only."""
    print_section("Example 3: Duplication Detection Only")
    
    client = EDGPAIModelClient()
    
    # Customer data with duplicates
    customer_data = [
        {"email": "alice@email.com", "name": "Alice Johnson", "phone": "123-456-7890"},
        {"email": "bob@email.com", "name": "Bob Smith", "phone": "234-567-8901"},
        {"email": "charlie@email.com", "name": "Charlie Brown", "phone": "345-678-9012"},
        # Exact duplicate
        {"email": "alice@email.com", "name": "Alice Johnson", "phone": "123-456-7890"},
        # Near duplicate (different email format)
        {"email": "alice.johnson@email.com", "name": "Alice Johnson", "phone": "123-456-7890"},
        # Different person with same name
        {"email": "alice.different@email.com", "name": "Alice Johnson", "phone": "999-888-7777"},
    ]
    
    print("Analyzing customer data for duplicates...")
    result = client.analyze_data(
        customer_data, 
        check_type="duplication",
        duplication_threshold=0.8  # Lower threshold = more sensitive to similarity
    )
    print_results(result)

def example_4_file_analysis():
    """Example 4: Analyzing files."""
    print_section("Example 4: File Analysis")
    
    client = EDGPAIModelClient()
    
    # Analyze the sample CSV file
    print("Analyzing sample CSV file...")
    try:
        result = client.analyze_file("tests/sample_data.csv", check_type="both")
        print_results(result)
    except FileNotFoundError:
        print("❌ Sample file not found. Creating a sample file...")
        
        # Create sample data
        sample_df = pd.DataFrame([
            {"name": "John", "age": 25, "income": 50000, "score": 85},
            {"name": "Jane", "age": 30, "income": 60000, "score": 90},
            {"name": "Bob", "age": 35, "income": 70000, "score": 88},
            {"name": "Alice", "age": 28, "income": 55000, "score": 87},
            {"name": "John", "age": 25, "income": 50000, "score": 85},  # Duplicate
            {"name": "Outlier", "age": 999, "income": 999999, "score": -50},  # Anomaly
        ])
        
        sample_df.to_csv("sample_analysis.csv", index=False)
        print("✅ Created sample_analysis.csv")
        
        result = client.analyze_file("sample_analysis.csv", check_type="both")
        print_results(result)

def example_5_custom_thresholds():
    """Example 5: Using custom thresholds."""
    print_section("Example 5: Custom Thresholds")
    
    client = EDGPAIModelClient()
    
    # IoT sensor data
    sensor_data = [
        {"sensor_id": "S001", "temperature": 22.5, "humidity": 45.0, "pressure": 1013.2},
        {"sensor_id": "S002", "temperature": 23.1, "humidity": 47.2, "pressure": 1012.8},
        {"sensor_id": "S003", "temperature": 21.9, "humidity": 44.1, "pressure": 1013.5},
        {"sensor_id": "S004", "temperature": 22.8, "humidity": 46.8, "pressure": 1013.0},
        # Anomalous readings
        {"sensor_id": "S005", "temperature": 150.0, "humidity": 95.0, "pressure": 800.0},
        {"sensor_id": "S006", "temperature": -50.0, "humidity": 5.0, "pressure": 1200.0},
    ]
    
    print("Testing different threshold settings...")
    
    # High sensitivity (low threshold)
    print("\n🔍 High Sensitivity (threshold=0.2):")
    result1 = client.analyze_data(sensor_data, anomaly_threshold=0.2)
    print(f"Anomalies found: {len(result1.get('anomalies', []))}")
    
    # Medium sensitivity
    print("\n🔍 Medium Sensitivity (threshold=0.5):")
    result2 = client.analyze_data(sensor_data, anomaly_threshold=0.5)
    print(f"Anomalies found: {len(result2.get('anomalies', []))}")
    
    # Low sensitivity (high threshold)
    print("\n🔍 Low Sensitivity (threshold=0.8):")
    result3 = client.analyze_data(sensor_data, anomaly_threshold=0.8)
    print(f"Anomalies found: {len(result3.get('anomalies', []))}")

def example_6_real_world_scenarios():
    """Example 6: Real-world data quality scenarios."""
    print_section("Example 6: Real-World Scenarios")
    
    client = EDGPAIModelClient()
    
    # Scenario 1: E-commerce order data
    print("\n📦 E-commerce Order Analysis:")
    order_data = [
        {"order_id": "ORD001", "customer_id": "C001", "amount": 29.99, "items": 2},
        {"order_id": "ORD002", "customer_id": "C002", "amount": 15.50, "items": 1},
        {"order_id": "ORD003", "customer_id": "C003", "amount": 89.99, "items": 3},
        {"order_id": "ORD004", "customer_id": "C004", "amount": 45.00, "items": 2},
        # Suspicious order (anomaly)
        {"order_id": "ORD005", "customer_id": "C005", "amount": 50000.00, "items": 1},
        # Duplicate order
        {"order_id": "ORD006", "customer_id": "C001", "amount": 29.99, "items": 2},
    ]
    
    result = client.analyze_data(order_data, check_type="both")
    print_results(result)
    
    # Scenario 2: User registration data
    print("\n👤 User Registration Analysis:")
    user_data = [
        {"username": "user1", "email": "user1@email.com", "age": 25, "country": "US"},
        {"username": "user2", "email": "user2@email.com", "age": 30, "country": "UK"},
        {"username": "user3", "email": "user3@email.com", "age": 28, "country": "CA"},
        # Suspicious registration (anomaly)
        {"username": "bot123", "email": "fake@temp.com", "age": 999, "country": "XX"},
        # Duplicate user
        {"username": "user1_duplicate", "email": "user1@email.com", "age": 25, "country": "US"},
    ]
    
    result = client.analyze_data(user_data, check_type="both")
    print_results(result)

def demonstrate_api_endpoints():
    """Demonstrate all available API endpoints."""
    print_section("API Endpoints Demonstration")
    
    client = EDGPAIModelClient()
    
    # Health check
    print("🏥 Health Check:")
    health = client.check_health()
    print(f"Status: {health.get('status')}")
    print(f"Version: {health.get('version')}")
    print(f"Model loaded: {health.get('model_loaded')}")
    
    # Service info
    print("\nℹ️  Service Information:")
    info = client.get_service_info()
    print(f"Service: {info.get('service')}")
    print(f"Supported checks: {info.get('supported_checks')}")
    print(f"Max file size: {info.get('max_file_size')} bytes")
    
    # Model info
    print("\n🤖 Model Information:")
    model_info = client.get_model_info()
    print(f"Model name: {model_info.get('model_name')}")
    print(f"Implementation: {model_info.get('current_implementation')}")
    print(f"HuggingFace: {model_info.get('huggingface_integration')}")

def main():
    """Main function to run all examples."""
    print("🚀 EDGP AI Model Service - Complete Usage Guide")
    print("="*60)
    
    client = EDGPAIModelClient()
    
    # Check if service is running
    try:
        health = client.check_health()
        print(f"✅ Service is running: {health.get('status')}")
    except requests.exceptions.ConnectionError:
        print("❌ Error: Service is not running!")
        print("Please start the service first:")
        print("  1. cd /path/to/edgp-ai-model")
        print("  2. python main.py")
        return
    
    # Run all examples
    demonstrate_api_endpoints()
    example_1_basic_usage()
    example_2_anomaly_only()
    example_3_duplication_only()
    example_4_file_analysis()
    example_5_custom_thresholds()
    example_6_real_world_scenarios()
    
    print(f"\n{'='*60}")
    print("🎉 All examples completed successfully!")
    print("💡 Tips:")
    print("  - Adjust thresholds based on your data sensitivity needs")
    print("  - Use 'anomaly' or 'duplication' for specific checks")
    print("  - Check the web interface at: http://127.0.0.1:8000/docs")
    print("  - View service status at: http://127.0.0.1:8000")

if __name__ == "__main__":
    main()
