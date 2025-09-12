#!/usr/bin/env python3
"""
EDGP AI Model Startup Script

Supports runtime environment variables similar to the EDGP rules engine.
Usage:
    APP_ENV="DEV" AWS_REGION="ap-southeast-1" AWS_ACCESS_KEY_ID="..." AWS_SECRET_ACCESS_KEY="..." python -m app.main
    
or:
    python start.py --env DEV --aws-region ap-southeast-1 --aws-access-key-id "..." --aws-secret-access-key "..."
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add the current directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

def setup_environment_variables(args):
    """Setup environment variables from command line arguments or existing environment."""
    
    # Core application settings
    if args.env:
        os.environ['APP_ENV'] = args.env
    if args.debug is not None:
        os.environ['DEBUG'] = str(args.debug).lower()
    if args.log_level:
        os.environ['LOG_LEVEL'] = args.log_level
    
    # API settings
    if args.host:
        os.environ['API_HOST'] = args.host
    if args.port:
        os.environ['API_PORT'] = str(args.port)
    
    # AWS configuration
    if args.aws_region:
        os.environ['AWS_REGION'] = args.aws_region
    if args.aws_access_key_id:
        os.environ['AWS_ACCESS_KEY_ID'] = args.aws_access_key_id
    if args.aws_secret_access_key:
        os.environ['AWS_SECRET_ACCESS_KEY'] = args.aws_secret_access_key
    if args.aws_profile:
        os.environ['AWS_PROFILE'] = args.aws_profile
    
    # SQS configuration
    if args.workflow_request_sqs_url:
        os.environ['WORKFLOW_RULES_REQUEST_SQS_URL'] = args.workflow_request_sqs_url
    if args.workflow_response_sqs_url:
        os.environ['WORKFLOW_RULES_RESPONSE_SQS_URL'] = args.workflow_response_sqs_url
    
    # LLM configuration
    if args.openai_api_key:
        os.environ['OPENAI_API_KEY'] = args.openai_api_key
    if args.anthropic_api_key:
        os.environ['ANTHROPIC_API_KEY'] = args.anthropic_api_key
    if args.llm_provider:
        os.environ['DEFAULT_LLM_PROVIDER'] = args.llm_provider


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="EDGP AI Model - Agentic Master Data Management",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run with environment variables
    APP_ENV="DEV" AWS_REGION="ap-southeast-1" python start.py
    
    # Run with command line arguments
    python start.py --env DEV --aws-region ap-southeast-1 --aws-access-key-id "AKIAXXXXXXXX" --aws-secret-access-key "XXXXXXXX"
    
    # Run with SQS queues
    python start.py --env PROD --workflow-request-sqs-url "https://sqs.ap-southeast-1.amazonaws.com/123456789/input" --workflow-response-sqs-url "https://sqs.ap-southeast-1.amazonaws.com/123456789/output"
        """
    )
    
    # Core application arguments
    parser.add_argument('--env', help='Application environment (DEV, STAGING, PROD)')
    parser.add_argument('--debug', type=bool, help='Enable debug mode')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], help='Logging level')
    
    # API arguments
    parser.add_argument('--host', default='0.0.0.0', help='API host address')
    parser.add_argument('--port', type=int, default=8000, help='API port number')
    
    # AWS arguments
    parser.add_argument('--aws-region', help='AWS region')
    parser.add_argument('--aws-access-key-id', help='AWS access key ID')
    parser.add_argument('--aws-secret-access-key', help='AWS secret access key')
    parser.add_argument('--aws-profile', help='AWS profile name')
    
    # SQS arguments
    parser.add_argument('--workflow-request-sqs-url', help='SQS URL for workflow requests')
    parser.add_argument('--workflow-response-sqs-url', help='SQS URL for workflow responses')
    
    # LLM arguments
    parser.add_argument('--openai-api-key', help='OpenAI API key')
    parser.add_argument('--anthropic-api-key', help='Anthropic API key')
    parser.add_argument('--llm-provider', choices=['openai', 'anthropic', 'bedrock'], help='Default LLM provider')
    
    return parser.parse_args()


def validate_configuration():
    """Validate the configuration before starting the application."""
    errors = []
    warnings = []
    
    # Check AWS configuration
    aws_access_key = os.getenv('AWS_ACCESS_KEY_ID')
    aws_secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
    aws_region = os.getenv('AWS_REGION')
    
    if aws_access_key and aws_secret_key and not aws_region:
        warnings.append("AWS credentials provided but no region specified. Using default region.")
    
    # Check SQS configuration
    request_sqs = os.getenv('WORKFLOW_RULES_REQUEST_SQS_URL')
    response_sqs = os.getenv('WORKFLOW_RULES_RESPONSE_SQS_URL')
    
    if request_sqs and not (aws_access_key or os.getenv('AWS_PROFILE')):
        warnings.append("SQS URLs provided but no AWS credentials found. Make sure AWS credentials are configured.")
    
    # Check LLM configuration
    openai_key = os.getenv('OPENAI_API_KEY')
    anthropic_key = os.getenv('ANTHROPIC_API_KEY')
    llm_provider = os.getenv('DEFAULT_LLM_PROVIDER', 'bedrock')
    
    if llm_provider == 'openai' and not openai_key:
        warnings.append("OpenAI provider selected but no API key provided.")
    elif llm_provider == 'anthropic' and not anthropic_key:
        warnings.append("Anthropic provider selected but no API key provided.")
    elif llm_provider == 'bedrock' and not (aws_access_key or os.getenv('AWS_PROFILE')):
        warnings.append("Bedrock provider selected but no AWS credentials found.")
    
    return errors, warnings


def print_configuration_summary():
    """Print a summary of the current configuration."""
    print("\n" + "="*60)
    print("EDGP AI Model - Configuration Summary")
    print("="*60)
    
    # Application settings
    print(f"Environment: {os.getenv('APP_ENV', 'development')}")
    print(f"Debug Mode: {os.getenv('DEBUG', 'true')}")
    print(f"Log Level: {os.getenv('LOG_LEVEL', 'INFO')}")
    print(f"API Host: {os.getenv('API_HOST', '0.0.0.0')}:{os.getenv('API_PORT', '8000')}")
    
    # AWS settings
    aws_region = os.getenv('AWS_REGION', 'us-east-1')
    aws_profile = os.getenv('AWS_PROFILE')
    aws_key_id = os.getenv('AWS_ACCESS_KEY_ID')
    
    print(f"\nAWS Region: {aws_region}")
    if aws_profile:
        print(f"AWS Profile: {aws_profile}")
    if aws_key_id:
        print(f"AWS Access Key: {aws_key_id[:8]}...")
    
    # SQS settings
    request_sqs = os.getenv('WORKFLOW_RULES_REQUEST_SQS_URL')
    response_sqs = os.getenv('WORKFLOW_RULES_RESPONSE_SQS_URL')
    
    if request_sqs:
        print(f"\nRequest SQS: {request_sqs}")
    if response_sqs:
        print(f"Response SQS: {response_sqs}")
    
    # LLM settings
    llm_provider = os.getenv('DEFAULT_LLM_PROVIDER', 'bedrock')
    print(f"\nLLM Provider: {llm_provider}")
    
    if os.getenv('OPENAI_API_KEY'):
        print("OpenAI API Key: configured")
    if os.getenv('ANTHROPIC_API_KEY'):
        print("Anthropic API Key: configured")
    
    print("="*60 + "\n")


def main():
    """Main entry point."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Setup environment variables
    setup_environment_variables(args)
    
    # Validate configuration
    errors, warnings = validate_configuration()
    
    # Print configuration summary
    print_configuration_summary()
    
    # Display warnings
    if warnings:
        print("⚠️  Warnings:")
        for warning in warnings:
            print(f"   - {warning}")
        print()
    
    # Check for errors
    if errors:
        print("❌ Configuration Errors:")
        for error in errors:
            print(f"   - {error}")
        sys.exit(1)
    
    # Import and run the main application
    try:
        print("🚀 Starting EDGP AI Model...")
        from main import app
        import uvicorn
        
        # Run the application
        uvicorn.run(
            app,
            host=os.getenv('API_HOST', '0.0.0.0'),
            port=int(os.getenv('API_PORT', '8000')),
            log_level=os.getenv('LOG_LEVEL', 'info').lower(),
            access_log=True
        )
        
    except ImportError as e:
        print(f"❌ Failed to import application: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Failed to start application: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
