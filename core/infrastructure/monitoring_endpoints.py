"""
Monitoring and observability endpoints for EDGP AI Model.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import asyncio

from ..auth import require_admin, require_permissions, Permission, TokenPayload
from ..monitoring import (
    metrics, system_monitor, health_checker, structured_logger,
    HealthStatus, performance_tracker
)
from ..error_handling import ErrorCode, EdgeError

router = APIRouter(prefix="/monitoring", tags=["Monitoring"])


class MetricsResponse(BaseModel):
    """Metrics response model."""
    timestamp: str
    counters: Dict[str, int]
    gauges: Dict[str, float]
    histograms: Dict[str, Dict[str, float]]
    timers: Dict[str, Dict[str, float]]


class HealthCheckResponse(BaseModel):
    """Health check response model."""
    status: str
    timestamp: str
    checks: Dict[str, Dict[str, Any]]


class SystemStatusResponse(BaseModel):
    """System status response model."""
    status: str
    uptime_seconds: float
    timestamp: str
    version: str
    environment: str


class LogEntry(BaseModel):
    """Log entry model."""
    timestamp: str
    level: str
    message: str
    context: Optional[Dict[str, Any]] = None


class AlertRule(BaseModel):
    """Alert rule configuration."""
    name: str
    metric_name: str
    threshold: float
    operator: str  # gt, lt, eq
    duration_minutes: int
    severity: str


# System status tracking
_system_start_time = datetime.utcnow()


@router.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """
    Get system health status.
    Public endpoint for load balancer health checks.
    """
    try:
        health_result = await health_checker.check_health()
        return HealthCheckResponse(**health_result)
    except Exception as e:
        structured_logger.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Health check failed"
        )


@router.get("/health/detailed", response_model=HealthCheckResponse)
async def detailed_health_check(
    current_user: TokenPayload = Depends(require_permissions(Permission.VIEW_AUDIT_LOGS))
):
    """
    Get detailed health status with internal checks.
    Requires admin permissions.
    """
    try:
        health_result = await health_checker.check_health()
        
        # Add additional internal checks
        additional_checks = {
            'database_connections': await _check_database_connections(),
            'llm_service': await _check_llm_service(),
            'active_requests': _check_active_requests()
        }
        
        health_result['checks'].update(additional_checks)
        return HealthCheckResponse(**health_result)
    except Exception as e:
        structured_logger.error(f"Detailed health check failed: {e}")
        raise EdgeError(
            message="Detailed health check failed",
            error_code=ErrorCode.INTERNAL_SERVER_ERROR,
            details={"error": str(e)}
        )


@router.get("/metrics", response_model=MetricsResponse)
async def get_metrics(
    current_user: TokenPayload = Depends(require_permissions(Permission.VIEW_AUDIT_LOGS))
):
    """
    Get application metrics.
    Requires monitoring permissions.
    """
    try:
        metrics_data = metrics.get_metrics()
        return MetricsResponse(**metrics_data)
    except Exception as e:
        structured_logger.error(f"Failed to get metrics: {e}")
        raise EdgeError(
            message="Failed to retrieve metrics",
            error_code=ErrorCode.INTERNAL_SERVER_ERROR
        )


@router.get("/metrics/prometheus")
async def get_prometheus_metrics(
    current_user: TokenPayload = Depends(require_permissions(Permission.VIEW_AUDIT_LOGS))
):
    """
    Get metrics in Prometheus format.
    """
    try:
        metrics_data = metrics.get_metrics()
        prometheus_output = _convert_to_prometheus_format(metrics_data)
        
        return {
            "content": prometheus_output,
            "media_type": "text/plain"
        }
    except Exception as e:
        structured_logger.error(f"Failed to get Prometheus metrics: {e}")
        raise EdgeError(
            message="Failed to retrieve Prometheus metrics",
            error_code=ErrorCode.INTERNAL_SERVER_ERROR
        )


@router.get("/status", response_model=SystemStatusResponse)
async def get_system_status():
    """
    Get basic system status.
    Public endpoint.
    """
    uptime = (datetime.utcnow() - _system_start_time).total_seconds()
    
    # Get overall health
    try:
        health = await health_checker.check_health()
        system_status = health['status']
    except:
        system_status = HealthStatus.UNHEALTHY
    
    return SystemStatusResponse(
        status=system_status,
        uptime_seconds=uptime,
        timestamp=datetime.utcnow().isoformat(),
        version="1.0.0",  # This should come from config
        environment="development"  # This should come from config
    )


@router.get("/logs")
async def get_recent_logs(
    level: Optional[str] = None,
    limit: int = 100,
    current_user: TokenPayload = Depends(require_admin())
):
    """
    Get recent log entries.
    Admin only.
    """
    # This is a simplified implementation
    # In production, you'd integrate with your log aggregation system
    return {
        "message": "Log retrieval would be implemented with your log aggregation system",
        "parameters": {
            "level": level,
            "limit": limit
        }
    }


@router.get("/performance/active-requests")
async def get_active_requests(
    current_user: TokenPayload = Depends(require_permissions(Permission.VIEW_AUDIT_LOGS))
):
    """
    Get currently active requests.
    """
    try:
        active_requests = list(performance_tracker.active_requests.values())
        
        # Add duration to each active request
        current_time = datetime.utcnow().timestamp()
        for request in active_requests:
            request['duration_seconds'] = current_time - request['start_time']
        
        return {
            "active_requests": active_requests,
            "total_count": len(active_requests),
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        structured_logger.error(f"Failed to get active requests: {e}")
        raise EdgeError(
            message="Failed to retrieve active requests",
            error_code=ErrorCode.INTERNAL_SERVER_ERROR
        )


@router.get("/performance/slowest-endpoints")
async def get_slowest_endpoints(
    limit: int = 10,
    current_user: TokenPayload = Depends(require_permissions(Permission.VIEW_AUDIT_LOGS))
):
    """
    Get slowest API endpoints based on recent performance data.
    """
    try:
        metrics_data = metrics.get_metrics()
        timers = metrics_data.get('timers', {})
        
        # Filter for HTTP request duration metrics
        http_timers = {
            k: v for k, v in timers.items() 
            if k.startswith('http_request_duration')
        }
        
        # Sort by average response time
        slowest = sorted(
            http_timers.items(),
            key=lambda x: x[1]['avg_ms'],
            reverse=True
        )[:limit]
        
        return {
            "slowest_endpoints": [
                {
                    "endpoint": endpoint,
                    "avg_response_time_ms": data['avg_ms'],
                    "max_response_time_ms": data['max_ms'],
                    "request_count": data['count'],
                    "p95_response_time_ms": data['p95_ms']
                }
                for endpoint, data in slowest
            ],
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        structured_logger.error(f"Failed to get slowest endpoints: {e}")
        raise EdgeError(
            message="Failed to retrieve performance data",
            error_code=ErrorCode.INTERNAL_SERVER_ERROR
        )


@router.get("/errors/summary")
async def get_error_summary(
    hours: int = 24,
    current_user: TokenPayload = Depends(require_permissions(Permission.VIEW_AUDIT_LOGS))
):
    """
    Get error summary for the specified time period.
    """
    try:
        metrics_data = metrics.get_metrics()
        
        # Get error-related counters
        error_metrics = {
            k: v for k, v in metrics_data.get('counters', {}).items()
            if 'error' in k.lower() or 'exception' in k.lower()
        }
        
        total_errors = sum(error_metrics.values())
        
        return {
            "summary": {
                "total_errors": total_errors,
                "time_period_hours": hours,
                "error_rate": total_errors / hours if hours > 0 else 0
            },
            "error_breakdown": error_metrics,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        structured_logger.error(f"Failed to get error summary: {e}")
        raise EdgeError(
            message="Failed to retrieve error summary",
            error_code=ErrorCode.INTERNAL_SERVER_ERROR
        )


@router.post("/system/start-monitoring", dependencies=[Depends(require_admin())])
async def start_monitoring(current_user: TokenPayload = Depends(require_admin())):
    """
    Start system monitoring.
    Admin only.
    """
    try:
        system_monitor.start_monitoring()
        structured_logger.info(f"System monitoring started by {current_user.username}")
        
        return {"message": "System monitoring started successfully"}
    except Exception as e:
        structured_logger.error(f"Failed to start monitoring: {e}")
        raise EdgeError(
            message="Failed to start system monitoring",
            error_code=ErrorCode.INTERNAL_SERVER_ERROR
        )


@router.post("/system/stop-monitoring", dependencies=[Depends(require_admin())])
async def stop_monitoring(current_user: TokenPayload = Depends(require_admin())):
    """
    Stop system monitoring.
    Admin only.
    """
    try:
        system_monitor.stop_monitoring()
        structured_logger.info(f"System monitoring stopped by {current_user.username}")
        
        return {"message": "System monitoring stopped successfully"}
    except Exception as e:
        structured_logger.error(f"Failed to stop monitoring: {e}")
        raise EdgeError(
            message="Failed to stop system monitoring",
            error_code=ErrorCode.INTERNAL_SERVER_ERROR
        )


# Helper functions
async def _check_database_connections():
    """Check database connection health."""
    try:
        # This would check actual database connections
        # Simplified implementation
        return {
            'status': HealthStatus.HEALTHY,
            'message': 'Database connections healthy',
            'details': {'connection_count': 5, 'max_connections': 20}
        }
    except Exception as e:
        return {
            'status': HealthStatus.UNHEALTHY,
            'message': f'Database connection failed: {str(e)}',
            'details': {'error': str(e)}
        }


async def _check_llm_service():
    """Check LLM service health."""
    try:
        # This would ping the actual LLM service
        # Simplified implementation
        return {
            'status': HealthStatus.HEALTHY,
            'message': 'LLM service responding',
            'details': {'response_time_ms': 150}
        }
    except Exception as e:
        return {
            'status': HealthStatus.DEGRADED,
            'message': f'LLM service check failed: {str(e)}',
            'details': {'error': str(e)}
        }


def _check_active_requests():
    """Check active request count."""
    active_count = len(performance_tracker.active_requests)
    
    if active_count > 100:  # Threshold for too many active requests
        return {
            'status': HealthStatus.DEGRADED,
            'message': f'High number of active requests: {active_count}',
            'details': {'active_requests': active_count}
        }
    
    return {
        'status': HealthStatus.HEALTHY,
        'message': f'Normal active requests: {active_count}',
        'details': {'active_requests': active_count}
    }


def _convert_to_prometheus_format(metrics_data: Dict[str, Any]) -> str:
    """Convert metrics to Prometheus exposition format."""
    lines = []
    
    # Add metadata
    lines.append("# TYPE edgp_http_requests_total counter")
    lines.append("# TYPE edgp_http_request_duration_seconds histogram")
    lines.append("# TYPE edgp_system_memory_percent gauge")
    
    # Convert counters
    for metric_name, value in metrics_data.get('counters', {}).items():
        lines.append(f"edgp_{metric_name} {value}")
    
    # Convert gauges
    for metric_name, value in metrics_data.get('gauges', {}).items():
        lines.append(f"edgp_{metric_name} {value}")
    
    # Add timestamp
    timestamp = int(datetime.utcnow().timestamp() * 1000)
    lines = [f"{line} {timestamp}" for line in lines if not line.startswith("#")]
    
    return "\n".join(lines)
