"""
API routes for data quality checking endpoints.
"""
import logging
from datetime import datetime
from typing import Dict, Any
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse
import pandas as pd
import json

from ..models.schemas import (
    DataQualityRequest, 
    DataQualityResponse, 
    HealthCheck,
    DataQualityCheckType
)
from ..services.data_quality_service import DataQualityService
from config.settings import settings

logger = logging.getLogger(__name__)

# Initialize router
router = APIRouter()

# Initialize service
data_quality_service = DataQualityService()

@router.get("/health", response_model=HealthCheck)
async def health_check():
    """Health check endpoint."""
    try:
        service_status = await data_quality_service.health_check()
        
        return HealthCheck(
            status=service_status["status"],
            version=settings.app_version,
            model_loaded=service_status["model_loaded"],
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

@router.post("/analyze", response_model=DataQualityResponse)
async def analyze_data_quality(request: DataQualityRequest):
    """Analyze data quality for anomalies and duplications."""
    try:
        logger.info(f"Received data quality analysis request with {len(request.data)} rows")
        
        response = await data_quality_service.analyze_data_quality(request)
        
        logger.info(f"Analysis completed: {len(response.anomalies)} anomalies, "
                   f"{len(response.duplications)} duplication groups found")
        
        return response
        
    except ValueError as e:
        logger.error(f"Invalid request data: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@router.post("/analyze-file", response_model=DataQualityResponse)
async def analyze_file(
    file: UploadFile = File(...),
    check_type: str = Form(default="both"),
    columns_to_check: str = Form(default=""),
    anomaly_threshold: float = Form(default=None),
    duplication_threshold: float = Form(default=None)
):
    """Analyze data quality from uploaded file."""
    try:
        # Validate file size
        if file.size > settings.max_file_size:
            raise HTTPException(
                status_code=413, 
                detail=f"File too large. Maximum size: {settings.max_file_size} bytes"
            )
        
        # Read file content
        content = await file.read()
        
        # Parse file based on extension
        if file.filename.endswith('.csv'):
            df = pd.read_csv(pd.io.common.StringIO(content.decode('utf-8')))
        elif file.filename.endswith('.json'):
            data = json.loads(content.decode('utf-8'))
            df = pd.DataFrame(data)
        else:
            raise HTTPException(
                status_code=400, 
                detail="Unsupported file format. Please use CSV or JSON."
            )
        
        # Convert DataFrame to list of dictionaries
        data_list = df.to_dict('records')
        
        # Parse columns to check
        columns_list = None
        if columns_to_check.strip():
            columns_list = [col.strip() for col in columns_to_check.split(',')]
        
        # Create request object
        request = DataQualityRequest(
            data=data_list,
            check_type=DataQualityCheckType(check_type),
            columns_to_check=columns_list,
            anomaly_threshold=anomaly_threshold,
            duplication_threshold=duplication_threshold
        )
        
        logger.info(f"Analyzing uploaded file: {file.filename} with {len(data_list)} rows")
        
        response = await data_quality_service.analyze_data_quality(request)
        
        logger.info(f"File analysis completed: {len(response.anomalies)} anomalies, "
                   f"{len(response.duplications)} duplication groups found")
        
        return response
        
    except pd.errors.EmptyDataError:
        raise HTTPException(status_code=400, detail="The uploaded file is empty")
    except pd.errors.ParserError as e:
        raise HTTPException(status_code=400, detail=f"Error parsing file: {str(e)}")
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON format: {str(e)}")
    except ValueError as e:
        logger.error(f"Invalid request data: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"File analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"File analysis failed: {str(e)}")

@router.get("/info")
async def get_service_info():
    """Get service information and available endpoints."""
    return {
        "service": settings.app_name,
        "version": settings.app_version,
        "description": settings.app_description,
        "endpoints": {
            "/health": "Service health check",
            "/analyze": "Analyze data quality from JSON payload",
            "/analyze-file": "Analyze data quality from uploaded file",
            "/info": "Service information"
        },
        "supported_checks": [check.value for check in DataQualityCheckType],
        "supported_formats": ["CSV", "JSON"],
        "max_file_size": settings.max_file_size,
        "default_thresholds": {
            "anomaly": settings.anomaly_threshold,
            "duplication": settings.duplication_threshold
        }
    }

@router.get("/model-info")
async def get_model_info():
    """Get information about the current AI model being used."""
    try:
        model_info = data_quality_service.anomaly_detector.get_model_info()
        return {
            "model_name": settings.model_name,
            "model_cache_dir": settings.model_cache_dir,
            "current_implementation": "sklearn-based (Isolation Forest + DBSCAN)",
            "huggingface_integration": "available_but_not_configured",
            "fallback_models": ["IsolationForest", "DBSCAN"],
            "note": "To use Amazon Science model, install transformers and torch packages"
        }
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        return {
            "error": str(e),
            "model_name": settings.model_name,
            "status": "error"
        }
