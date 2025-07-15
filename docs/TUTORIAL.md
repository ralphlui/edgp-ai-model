# 🚀 EDGP AI Model Service - Step-by-Step Tutorial

## How to Run and Use the AI Model Service

### Step 1: Start the Service

```bash
# Navigate to the project directory
cd /Users/jjfwang/Documents/02-NUS/01-Capstone/edgp-ai-model

# Start the service
python main.py
```

**You should see:**
```
INFO: Starting EDGP AI Model Service v1.0.0
INFO: Uvicorn running on http://127.0.0.1:8000
INFO: Application startup complete.
```

✅ **Service is now running at: http://127.0.0.1:8000**

### Step 2: Verify the Service is Working

Open a new terminal and test:

```bash
# Health check
curl http://127.0.0.1:8000/api/v1/health

# Expected response:
# {"status":"healthy","version":"1.0.0","model_loaded":true,"timestamp":"2025-07-14T..."}
```

### Step 3: View the Interactive API Documentation

Open your browser and go to:
- **Swagger UI**: http://127.0.0.1:8000/docs
- **ReDoc**: http://127.0.0.1:8000/redoc

### Step 4: Test with Sample Data

#### Method 1: Using curl (Terminal)

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/analyze" \
     -H "Content-Type: application/json" \
     -d '{
       "data": [
         {"name": "Alice", "age": 25, "salary": 50000},
         {"name": "Bob", "age": 30, "salary": 60000},
         {"name": "Outlier", "age": 999, "salary": 999999},
         {"name": "Alice", "age": 25, "salary": 50000}
       ],
       "check_type": "both"
     }'
```

#### Method 2: Using Python

```python
import requests

# Data with anomalies and duplicates
data = {
    "data": [
        {"name": "Alice", "age": 25, "salary": 50000},
        {"name": "Bob", "age": 30, "salary": 60000},
        {"name": "Outlier", "age": 999, "salary": 999999},  # Anomaly
        {"name": "Alice", "age": 25, "salary": 50000}       # Duplicate
    ],
    "check_type": "both"
}

response = requests.post("http://127.0.0.1:8000/api/v1/analyze", json=data)
result = response.json()

print(f"Anomalies: {len(result['anomalies'])}")
print(f"Duplicates: {len(result['duplications'])}")
```

#### Method 3: Using the Web Interface

1. Go to http://127.0.0.1:8000/docs
2. Click on "POST /api/v1/analyze"
3. Click "Try it out"
4. Paste this JSON in the request body:

```json
{
  "data": [
    {"name": "Alice", "age": 25, "salary": 50000},
    {"name": "Bob", "age": 30, "salary": 60000},
    {"name": "Outlier", "age": 999, "salary": 999999},
    {"name": "Alice", "age": 25, "salary": 50000}
  ],
  "check_type": "both"
}
```

5. Click "Execute"

### Step 5: Upload and Analyze Files

#### Create a sample CSV file:

```bash
# Create sample data file
cat > my_data.csv << EOF
name,age,salary,department
Alice,25,50000,Engineering
Bob,30,60000,Marketing
Charlie,35,70000,Engineering
Outlier,999,999999,Unknown
Alice,25,50000,Engineering
EOF
```

#### Analyze the file:

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/analyze-file" \
     -F "file=@my_data.csv" \
     -F "check_type=both"
```

### Step 6: Run the Complete Usage Guide

```bash
python usage_guide.py
```

This will run comprehensive examples showing:
- ✅ Basic data quality analysis
- ✅ Anomaly detection only
- ✅ Duplication detection only
- ✅ File analysis
- ✅ Custom thresholds
- ✅ Real-world scenarios

### Step 7: Understanding the Results

#### Sample Response:
```json
{
  "total_rows": 4,
  "check_type": "both",
  "anomalies": [
    {
      "row_index": 2,
      "anomaly_score": 1.0,
      "is_anomaly": true,
      "affected_columns": ["age", "salary"]
    }
  ],
  "duplications": [
    {
      "row_indices": [0, 3],
      "similarity_score": 1.0,
      "duplicate_columns": ["name", "age", "salary"]
    }
  ],
  "summary": {
    "total_rows": 4,
    "anomaly_count": 1,
    "anomaly_percentage": 25.0,
    "duplication_groups": 1,
    "duplicate_rows": 2,
    "duplication_percentage": 50.0
  },
  "processing_time": 0.145
}
```

#### What this means:
- **Row 2** is an anomaly (age: 999, salary: 999999)
- **Rows 0 and 3** are duplicates (identical Alice records)
- **25%** of data contains anomalies
- **50%** of data is duplicated

### Step 8: Advanced Usage

#### Check Current AI Model:
```bash
curl http://127.0.0.1:8000/api/v1/model-info
```

#### Adjust Sensitivity:
```python
# High sensitivity (detects more anomalies)
response = requests.post("http://127.0.0.1:8000/api/v1/analyze", json={
    "data": your_data,
    "check_type": "anomaly",
    "anomaly_threshold": 0.2  # Lower = more sensitive
})

# Low sensitivity (detects fewer duplicates)
response = requests.post("http://127.0.0.1:8000/api/v1/analyze", json={
    "data": your_data,
    "check_type": "duplication", 
    "duplication_threshold": 0.95  # Higher = more strict
})
```

#### Analyze Specific Columns:
```python
response = requests.post("http://127.0.0.1:8000/api/v1/analyze", json={
    "data": your_data,
    "check_type": "both",
    "columns_to_check": ["age", "salary"]  # Only check these columns
})
```

## 🎯 Real-World Use Cases

### 1. **Data Cleaning Pipeline**
```python
# Clean customer database
customer_data = load_customer_data()
result = analyze_data_quality(customer_data)

# Remove duplicates
clean_data = remove_rows(customer_data, result['duplications'])

# Flag anomalies for review
flagged_data = flag_rows(customer_data, result['anomalies'])
```

### 2. **Fraud Detection**
```python
# Analyze transactions
transactions = load_transaction_data()
result = analyze_data_quality(transactions, check_type="anomaly", threshold=0.1)

# Investigate suspicious transactions
suspicious = [transactions[a['row_index']] for a in result['anomalies']]
```

### 3. **Data Quality Monitoring**
```python
# Daily data quality check
daily_data = load_daily_data()
result = analyze_data_quality(daily_data)

# Alert if quality drops
if result['summary']['anomaly_percentage'] > 10:
    send_alert("Data quality alert: High anomaly rate detected")
```

## 🛠️ Development Commands

```bash
# Run with auto-reload for development
uvicorn main:app --reload

# Run tests
python -m pytest tests/ -v

# Build and run with Docker
docker-compose up --build

# Check all available endpoints
curl http://127.0.0.1:8000/api/v1/info
```

## 🎉 You're Ready!

The EDGP AI Model Service is now running and ready to analyze your data for:
- **Anomalies**: Outliers, suspicious patterns, data errors
- **Duplicates**: Identical or similar records
- **Data Quality**: Overall assessment and statistics

**Next Steps:**
1. Integrate with your data pipeline
2. Customize thresholds for your use case
3. Set up monitoring and alerts
4. Explore advanced features and customization

**Need Help?**
- Check the interactive docs: http://127.0.0.1:8000/docs
- Review the usage guide: `python usage_guide.py`
- View service status: http://127.0.0.1:8000
