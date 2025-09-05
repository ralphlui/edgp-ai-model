"""
Comprehensive error handling and validation system for EDGP AI Model.
"""

import traceback
from typing import Any, Dict, List, Optional, Union, Type
from datetime import datetime
from enum import Enum
from fastapi import HTTPException, Request, Response, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError, ResponseValidationError
from pydantic import BaseModel, ValidationError
import logging

from .monitoring import structured_logger, metrics


class ErrorCode(str, Enum):
    """Standard error codes for the system."""
    
    # Authentication & Authorization
    UNAUTHORIZED = "UNAUTHORIZED"
    FORBIDDEN = "FORBIDDEN"
    INVALID_CREDENTIALS = "INVALID_CREDENTIALS"
    TOKEN_EXPIRED = "TOKEN_EXPIRED"
    INSUFFICIENT_PERMISSIONS = "INSUFFICIENT_PERMISSIONS"
    
    # Validation Errors
    VALIDATION_ERROR = "VALIDATION_ERROR"
    INVALID_INPUT = "INVALID_INPUT"
    MISSING_REQUIRED_FIELD = "MISSING_REQUIRED_FIELD"
    INVALID_DATA_TYPE = "INVALID_DATA_TYPE"
    INVALID_FORMAT = "INVALID_FORMAT"
    
    # Business Logic Errors
    AGENT_EXECUTION_FAILED = "AGENT_EXECUTION_FAILED"
    DATA_QUALITY_CHECK_FAILED = "DATA_QUALITY_CHECK_FAILED"
    COMPLIANCE_CHECK_FAILED = "COMPLIANCE_CHECK_FAILED"
    REMEDIATION_TASK_FAILED = "REMEDIATION_TASK_FAILED"
    
    # Resource Errors
    RESOURCE_NOT_FOUND = "RESOURCE_NOT_FOUND"
    RESOURCE_ALREADY_EXISTS = "RESOURCE_ALREADY_EXISTS"
    RESOURCE_LOCKED = "RESOURCE_LOCKED"
    
    # External Dependencies
    EXTERNAL_SERVICE_ERROR = "EXTERNAL_SERVICE_ERROR"
    LLM_SERVICE_ERROR = "LLM_SERVICE_ERROR"
    NETWORK_ERROR = "NETWORK_ERROR"
    
    # System Errors
    INTERNAL_SERVER_ERROR = "INTERNAL_SERVER_ERROR"
    SERVICE_UNAVAILABLE = "SERVICE_UNAVAILABLE"
    RATE_LIMIT_EXCEEDED = "RATE_LIMIT_EXCEEDED"
    CONFIGURATION_ERROR = "CONFIGURATION_ERROR"


class ErrorSeverity(str, Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class EdgeError(Exception):
    """Base exception class for EDGP AI Model."""
    
    def __init__(
        self,
        message: str,
        error_code: ErrorCode,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None
    ):
        self.message = message
        self.error_code = error_code
        self.severity = severity
        self.details = details or {}
        self.cause = cause
        self.timestamp = datetime.utcnow()
        super().__init__(self.message)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary."""
        return {
            "error_code": self.error_code.value,
            "message": self.message,
            "severity": self.severity.value,
            "details": self.details,
            "timestamp": self.timestamp.isoformat(),
            "cause": str(self.cause) if self.cause else None
        }
    
    def to_http_response(self) -> JSONResponse:
        """Convert to HTTP response."""
        status_code = self._get_http_status_code()
        
        return JSONResponse(
            status_code=status_code,
            content={
                "success": False,
                "error": self.to_dict(),
                "timestamp": datetime.utcnow().isoformat()
            }
        )
    
    def _get_http_status_code(self) -> int:
        """Get appropriate HTTP status code for error."""
        status_map = {
            ErrorCode.UNAUTHORIZED: 401,
            ErrorCode.FORBIDDEN: 403,
            ErrorCode.INVALID_CREDENTIALS: 401,
            ErrorCode.TOKEN_EXPIRED: 401,
            ErrorCode.INSUFFICIENT_PERMISSIONS: 403,
            ErrorCode.VALIDATION_ERROR: 400,
            ErrorCode.INVALID_INPUT: 400,
            ErrorCode.MISSING_REQUIRED_FIELD: 400,
            ErrorCode.INVALID_DATA_TYPE: 400,
            ErrorCode.INVALID_FORMAT: 400,
            ErrorCode.RESOURCE_NOT_FOUND: 404,
            ErrorCode.RESOURCE_ALREADY_EXISTS: 409,
            ErrorCode.RESOURCE_LOCKED: 423,
            ErrorCode.RATE_LIMIT_EXCEEDED: 429,
            ErrorCode.SERVICE_UNAVAILABLE: 503,
        }
        
        return status_map.get(self.error_code, 500)


class AuthenticationError(EdgeError):
    """Authentication-related errors."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code=ErrorCode.UNAUTHORIZED,
            severity=ErrorSeverity.HIGH,
            details=details
        )


class AuthorizationError(EdgeError):
    """Authorization-related errors."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code=ErrorCode.FORBIDDEN,
            severity=ErrorSeverity.HIGH,
            details=details
        )


class ValidationError(EdgeError):
    """Validation-related errors."""
    
    def __init__(self, message: str, field: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        error_details = details or {}
        if field:
            error_details["field"] = field
        
        super().__init__(
            message=message,
            error_code=ErrorCode.VALIDATION_ERROR,
            severity=ErrorSeverity.LOW,
            details=error_details
        )


class BusinessLogicError(EdgeError):
    """Business logic errors."""
    
    def __init__(self, message: str, error_code: ErrorCode, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code=error_code,
            severity=ErrorSeverity.MEDIUM,
            details=details
        )


class ExternalServiceError(EdgeError):
    """External service errors."""
    
    def __init__(self, service: str, message: str, details: Optional[Dict[str, Any]] = None):
        error_details = details or {}
        error_details["service"] = service
        
        super().__init__(
            message=f"External service error ({service}): {message}",
            error_code=ErrorCode.EXTERNAL_SERVICE_ERROR,
            severity=ErrorSeverity.HIGH,
            details=error_details
        )


class ErrorResponse(BaseModel):
    """Standard error response model."""
    
    success: bool = False
    error: Dict[str, Any]
    timestamp: str
    request_id: Optional[str] = None


class ValidationErrorDetail(BaseModel):
    """Detailed validation error information."""
    
    field: str
    message: str
    value: Any = None
    error_type: str


class ErrorHandler:
    """Central error handler for the application."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
    
    async def handle_edge_error(self, request: Request, exc: EdgeError) -> JSONResponse:
        """Handle EdgeError exceptions."""
        request_id = getattr(request.state, 'request_id', None)
        
        # Log the error
        self.logger.error(
            f"EdgeError: {exc.message}",
            error_code=exc.error_code.value,
            severity=exc.severity.value,
            details=exc.details,
            request_id=request_id,
            traceback=traceback.format_exc() if exc.severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL] else None
        )
        
        # Record error metrics
        metrics.increment("errors_total", labels={
            "error_code": exc.error_code.value,
            "severity": exc.severity.value
        })
        
        # Create response
        response_data = exc.to_dict()
        if request_id:
            response_data["request_id"] = request_id
        
        return JSONResponse(
            status_code=exc._get_http_status_code(),
            content={
                "success": False,
                "error": response_data,
                "timestamp": datetime.utcnow().isoformat(),
                "request_id": request_id
            }
        )
    
    async def handle_validation_error(self, request: Request, exc: RequestValidationError) -> JSONResponse:
        """Handle Pydantic validation errors."""
        request_id = getattr(request.state, 'request_id', None)
        
        # Parse validation errors
        validation_errors = []
        for error in exc.errors():
            validation_errors.append({
                "field": ".".join(str(loc) for loc in error["loc"]),
                "message": error["msg"],
                "error_type": error["type"],
                "value": error.get("input")
            })
        
        self.logger.warning(
            f"Validation error: {len(validation_errors)} fields failed validation",
            validation_errors=validation_errors,
            request_id=request_id
        )
        
        metrics.increment("validation_errors_total")
        
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "success": False,
                "error": {
                    "error_code": ErrorCode.VALIDATION_ERROR.value,
                    "message": "Request validation failed",
                    "details": {
                        "validation_errors": validation_errors
                    },
                    "timestamp": datetime.utcnow().isoformat()
                },
                "request_id": request_id
            }
        )
    
    async def handle_http_exception(self, request: Request, exc: HTTPException) -> JSONResponse:
        """Handle FastAPI HTTP exceptions."""
        request_id = getattr(request.state, 'request_id', None)
        
        self.logger.warning(
            f"HTTP Exception: {exc.detail}",
            status_code=exc.status_code,
            request_id=request_id
        )
        
        metrics.increment("http_exceptions_total", labels={
            "status_code": str(exc.status_code)
        })
        
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "success": False,
                "error": {
                    "error_code": self._status_code_to_error_code(exc.status_code).value,
                    "message": str(exc.detail),
                    "timestamp": datetime.utcnow().isoformat()
                },
                "request_id": request_id
            }
        )
    
    async def handle_unexpected_error(self, request: Request, exc: Exception) -> JSONResponse:
        """Handle unexpected errors."""
        request_id = getattr(request.state, 'request_id', None)
        
        # Log the full traceback for unexpected errors
        self.logger.critical(
            f"Unexpected error: {str(exc)}",
            error_type=type(exc).__name__,
            traceback=traceback.format_exc(),
            request_id=request_id
        )
        
        metrics.increment("unexpected_errors_total", labels={
            "error_type": type(exc).__name__
        })
        
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "success": False,
                "error": {
                    "error_code": ErrorCode.INTERNAL_SERVER_ERROR.value,
                    "message": "An unexpected error occurred",
                    "timestamp": datetime.utcnow().isoformat()
                },
                "request_id": request_id
            }
        )
    
    def _status_code_to_error_code(self, status_code: int) -> ErrorCode:
        """Map HTTP status codes to error codes."""
        mapping = {
            401: ErrorCode.UNAUTHORIZED,
            403: ErrorCode.FORBIDDEN,
            404: ErrorCode.RESOURCE_NOT_FOUND,
            409: ErrorCode.RESOURCE_ALREADY_EXISTS,
            429: ErrorCode.RATE_LIMIT_EXCEEDED,
            503: ErrorCode.SERVICE_UNAVAILABLE
        }
        
        return mapping.get(status_code, ErrorCode.INTERNAL_SERVER_ERROR)


class InputValidator:
    """Input validation utilities."""
    
    @staticmethod
    def validate_agent_type(agent_type: str) -> str:
        """Validate agent type parameter."""
        valid_types = ["data_quality", "compliance", "remediation", "analytics", "policy_suggestion"]
        
        if agent_type not in valid_types:
            raise ValidationError(
                message=f"Invalid agent type: {agent_type}",
                field="agent_type",
                details={
                    "valid_types": valid_types,
                    "provided": agent_type
                }
            )
        
        return agent_type
    
    @staticmethod
    def validate_dataset_id(dataset_id: str) -> str:
        """Validate dataset ID format."""
        if not dataset_id or len(dataset_id) < 3:
            raise ValidationError(
                message="Dataset ID must be at least 3 characters long",
                field="dataset_id",
                details={"min_length": 3}
            )
        
        if not dataset_id.replace("_", "").replace("-", "").isalnum():
            raise ValidationError(
                message="Dataset ID must contain only alphanumeric characters, hyphens, and underscores",
                field="dataset_id"
            )
        
        return dataset_id
    
    @staticmethod
    def validate_pagination(page: int, size: int) -> tuple[int, int]:
        """Validate pagination parameters."""
        if page < 1:
            raise ValidationError(
                message="Page number must be greater than 0",
                field="page",
                details={"min_value": 1}
            )
        
        if size < 1 or size > 1000:
            raise ValidationError(
                message="Page size must be between 1 and 1000",
                field="size",
                details={"min_value": 1, "max_value": 1000}
            )
        
        return page, size
    
    @staticmethod
    def validate_score_range(score: float, field_name: str = "score") -> float:
        """Validate score is between 0 and 1."""
        if not 0 <= score <= 1:
            raise ValidationError(
                message=f"{field_name} must be between 0 and 1",
                field=field_name,
                details={"min_value": 0, "max_value": 1}
            )
        
        return score


# Global error handler instance
error_handler = ErrorHandler(structured_logger.logger)


# Exception handler decorators
def handle_external_service_errors(service_name: str):
    """Decorator to handle external service errors."""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                if not isinstance(e, EdgeError):
                    raise ExternalServiceError(
                        service=service_name,
                        message=str(e),
                        details={"operation": func.__name__}
                    )
                raise
        
        return wrapper
    return decorator


def setup_error_handlers(app):
    """Setup FastAPI error handlers for the application."""
    
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        """Handle HTTP exceptions."""
        structured_logger.error(
            "HTTP exception occurred",
            status_code=exc.status_code,
            detail=exc.detail,
            path=str(request.url)
        )
        
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "code": f"HTTP_{exc.status_code}",
                    "message": exc.detail,
                    "timestamp": datetime.utcnow().isoformat()
                }
            }
        )
    
    @app.exception_handler(ValidationError)
    async def validation_exception_handler(request: Request, exc: ValidationError):
        """Handle Pydantic validation errors."""
        structured_logger.warning(
            "Validation error occurred",
            errors=exc.errors(),
            path=str(request.url)
        )
        
        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "code": "VALIDATION_ERROR",
                    "message": "Request validation failed",
                    "details": exc.errors(),
                    "timestamp": datetime.utcnow().isoformat()
                }
            }
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        """Handle all other exceptions."""
        structured_logger.error(
            "Unexpected error occurred",
            error=str(exc),
            error_type=type(exc).__name__,
            path=str(request.url),
            traceback=traceback.format_exc()
        )
        
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "code": "INTERNAL_SERVER_ERROR",
                    "message": "An unexpected error occurred",
                    "timestamp": datetime.utcnow().isoformat()
                }
            }
        )
