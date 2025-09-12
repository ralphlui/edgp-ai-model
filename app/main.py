"""
Main entry point for EDGP AI Model when run as module

Usage:
    APP_ENV="DEV" AWS_REGION="ap-southeast-1" AWS_ACCESS_KEY_ID="..." AWS_SECRET_ACCESS_KEY="..." python -m app.main
"""

import os
import sys
from pathlib import Path

# Add parent directory to path to import main application
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

def main():
    """Main entry point for module execution."""
    print("🚀 Starting EDGP AI Model via module execution...")
    
    # Print environment configuration
    print("\n📋 Environment Configuration:")
    env_vars = [
        'APP_ENV', 'AWS_REGION', 'AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY',
        'WORKFLOW_RULES_REQUEST_SQS_URL', 'WORKFLOW_RULES_RESPONSE_SQS_URL',
        'DEBUG', 'LOG_LEVEL', 'API_HOST', 'API_PORT'
    ]
    
    for var in env_vars:
        value = os.getenv(var)
        if value:
            # Mask sensitive values
            if 'SECRET' in var or 'KEY' in var:
                display_value = value[:8] + "..." if len(value) > 8 else "***"
            else:
                display_value = value
            print(f"   {var}: {display_value}")
    
    # Import and run the FastAPI application
    import uvicorn
    from main import app
    
    # Get configuration from environment
    host = os.getenv('API_HOST', '0.0.0.0')
    port = int(os.getenv('API_PORT', '8000'))
    log_level = os.getenv('LOG_LEVEL', 'info').lower()
    
    print(f"\n🌐 Starting server at http://{host}:{port}")
    print("📚 API Documentation available at:")
    print(f"   - Swagger UI: http://{host}:{port}/docs")
    print(f"   - ReDoc: http://{host}:{port}/redoc")
    print("\n🔗 Available endpoints:")
    print(f"   - Health Check: http://{host}:{port}/health")
    print(f"   - Queue Status: http://{host}:{port}/message-queue/status")
    print(f"   - Send Message: http://{host}:{port}/message-queue/send-message")
    print()
    
    # Run the application
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level=log_level,
        access_log=True
    )


if __name__ == "__main__":
    main()
