"""
Configuration settings for the EDGP AI Model service.
"""
import os
from typing import Dict, Any
from pydantic_settings import BaseSettings
from pydantic import ConfigDict

class Settings(BaseSettings):
    """Application settings."""
    
    # API Configuration
    app_name: str = "EDGP AI Model Service"
    app_version: str = "1.0.0"
    app_description: str = "AI-powered data quality checking service for anomaly and duplication detection"
    
    # Server Configuration
    host: str = "127.0.0.1"
    port: int = 8000
    debug: bool = True
    
    # AI Model Configuration
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    model_cache_dir: str = "./models"
    use_local_models: bool = True
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    
    # Data Quality Thresholds
    anomaly_threshold: float = 0.5
    duplication_threshold: float = 0.95
    
    # Logging Configuration
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    model_config = ConfigDict(
        env_file=".env",
        case_sensitive=False,
        protected_namespaces=('settings_',)
    )

# Global settings instance
settings = Settings()
