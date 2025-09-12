"""
Message Queue Configuration Examples

Shows how to configure and use the message queue service with various deployment scenarios.
"""

import os
from typing import Dict, Any

# Development configuration example
DEV_CONFIG = {
    "APP_ENV": "DEV",
    "AWS_REGION": "ap-southeast-1",
    "AWS_ACCESS_KEY_ID": "AKIAXXXXXXXXXXXXXXXX",
    "AWS_SECRET_ACCESS_KEY": "your_secret_access_key_here",
    "WORKFLOW_RULES_REQUEST_SQS_URL": "https://sqs.ap-southeast-1.amazonaws.com/891377185630/test_sqs_queue_input",
    "WORKFLOW_RULES_RESPONSE_SQS_URL": "https://sqs.ap-southeast-1.amazonaws.com/891377185630/test_sqs_queue_output",
    "API_HOST": "0.0.0.0",
    "API_PORT": "8000",
    "LOG_LEVEL": "DEBUG"
}

# Production configuration example
PROD_CONFIG = {
    "APP_ENV": "PROD",
    "AWS_REGION": "us-east-1",
    "AWS_ACCESS_KEY_ID": "AKIAXXXXXXXXXXXXXXXX",
    "AWS_SECRET_ACCESS_KEY": "your_production_secret_key",
    "WORKFLOW_RULES_REQUEST_SQS_URL": "https://sqs.us-east-1.amazonaws.com/123456789012/prod_workflow_requests",
    "WORKFLOW_RULES_RESPONSE_SQS_URL": "https://sqs.us-east-1.amazonaws.com/123456789012/prod_workflow_responses",
    "API_HOST": "0.0.0.0",
    "API_PORT": "8000", 
    "LOG_LEVEL": "INFO"
}

# Using AWS Profile instead of credentials
PROFILE_CONFIG = {
    "APP_ENV": "STAGING",
    "AWS_REGION": "eu-west-1",
    "AWS_PROFILE": "edgp-staging",
    "WORKFLOW_RULES_REQUEST_SQS_URL": "https://sqs.eu-west-1.amazonaws.com/456789012345/staging_requests",
    "WORKFLOW_RULES_RESPONSE_SQS_URL": "https://sqs.eu-west-1.amazonaws.com/456789012345/staging_responses"
}

def print_startup_commands():
    """Print example startup commands for different configurations."""
    
    print("="*80)
    print("EDGP AI Model - Message Queue Service Startup Examples")
    print("="*80)
    
    print("\n1. Development Environment (Runtime Environment Variables):")
    print("-"*60)
    dev_cmd = " ".join([f'{key}="{value}"' for key, value in DEV_CONFIG.items()])
    print(f'{dev_cmd} python -m app.main')
    
    print("\n2. Production Environment (Command Line Arguments):")
    print("-"*60)
    prod_args = []
    for key, value in PROD_CONFIG.items():
        arg_name = key.lower().replace('_', '-')
        prod_args.append(f'--{arg_name} "{value}"')
    print(f'python start.py {" ".join(prod_args)}')
    
    print("\n3. Using AWS Profile (Mixed Approach):")
    print("-"*60)
    profile_cmd_parts = []
    for key, value in PROFILE_CONFIG.items():
        if key.startswith('AWS_') and key != 'AWS_REGION':
            continue
        profile_cmd_parts.append(f'{key}="{value}"')
    print(f'{" ".join(profile_cmd_parts)} python -m app.main')
    
    print("\n4. Docker Environment:")
    print("-"*60)
    print("docker run -e APP_ENV=PROD \\")
    for key, value in PROD_CONFIG.items():
        print(f"           -e {key}='{value}' \\")
    print("           edgp-ai-model:latest")
    
    print("\n5. Kubernetes ConfigMap:")
    print("-"*60)
    print("apiVersion: v1")
    print("kind: ConfigMap")
    print("metadata:")
    print("  name: edgp-ai-config")
    print("data:")
    for key, value in PROD_CONFIG.items():
        if 'SECRET' not in key and 'KEY' not in key:
            print(f"  {key}: \"{value}\"")
    
    print("\n6. Testing Message Queue:")
    print("-"*60)
    print("# Start the service")
    print("python start.py --env DEV")
    print()
    print("# Send a test message")
    print('curl -X POST "http://localhost:8000/message-queue/trigger-data-quality-assessment" \\')
    print('     -H "Content-Type: application/json" \\')
    print('     -d \'{"dataset_id": "test_dataset", "assessment_type": "full"}\'')
    print()
    print("# Check queue status")
    print('curl "http://localhost:8000/message-queue/status"')
    
    print("\n" + "="*80)

if __name__ == "__main__":
    print_startup_commands()
