"""
System monitoring and health check functionality.
Provides metrics collection, performance monitoring, and health status.
"""

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
import threading

# Optional system monitoring
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    psutil = None

try:
    from fastapi.middleware.base import BaseHTTPMiddleware
    FASTAPI_MIDDLEWARE_AVAILABLE = True
except ImportError:
    # Handle older FastAPI versions
    try:
        from starlette.middleware.base import BaseHTTPMiddleware  
        FASTAPI_MIDDLEWARE_AVAILABLE = True
    except ImportError:
        FASTAPI_MIDDLEWARE_AVAILABLE = False
        BaseHTTPMiddleware = object

import time
import uuid
import logging
import json
from datetime import datetime
from typing import Dict, Any, Optional, List
from contextlib import asynccontextmanager
from fastapi import Request, Response
import asyncio
from enum import Enum
import threading
from collections import defaultdict, deque

# Configure structured logging
import os
os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/edgp_ai_model.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


class MetricType(str, Enum):
    """Types of metrics we collect."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


class LogLevel(str, Enum):
    """Log levels with structured context."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class HealthStatus(str, Enum):
    """System health status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class StructuredLogger:
    """Structured logger with context support."""
    
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self.context = {}
    
    def set_context(self, **kwargs):
        """Set context for all log messages."""
        self.context.update(kwargs)
    
    def clear_context(self):
        """Clear logging context."""
        self.context.clear()
    
    def _log(self, level: LogLevel, message: str, **kwargs):
        """Log with structured data."""
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": level.value,
            "message": message,
            "context": self.context.copy(),
            **kwargs
        }
        
        extra = {"extra": json.dumps(log_data)}
        getattr(self.logger, level.value.lower())(message, extra=extra)
    
    def debug(self, message: str, **kwargs):
        self._log(LogLevel.DEBUG, message, **kwargs)
    
    def info(self, message: str, **kwargs):
        self._log(LogLevel.INFO, message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        self._log(LogLevel.WARNING, message, **kwargs)
    
    def error(self, message: str, **kwargs):
        self._log(LogLevel.ERROR, message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        self._log(LogLevel.CRITICAL, message, **kwargs)


class Metrics:
    """In-memory metrics collection system."""
    
    def __init__(self):
        self._metrics: Dict[str, Any] = {
            'counters': defaultdict(int),
            'gauges': defaultdict(float),
            'histograms': defaultdict(list),
            'timers': defaultdict(list)
        }
        self._lock = threading.Lock()
    
    def increment(self, name: str, value: int = 1, labels: Optional[Dict[str, str]] = None):
        """Increment a counter metric."""
        key = self._make_key(name, labels)
        with self._lock:
            self._metrics['counters'][key] += value
    
    def set_gauge(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Set a gauge metric."""
        key = self._make_key(name, labels)
        with self._lock:
            self._metrics['gauges'][key] = value
    
    def observe(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Add an observation to a histogram."""
        key = self._make_key(name, labels)
        with self._lock:
            self._metrics['histograms'][key].append(value)
            # Keep only last 1000 observations
            if len(self._metrics['histograms'][key]) > 1000:
                self._metrics['histograms'][key] = self._metrics['histograms'][key][-1000:]
    
    def record_time(self, name: str, duration: float, labels: Optional[Dict[str, str]] = None):
        """Record timing information."""
        key = self._make_key(name, labels)
        with self._lock:
            self._metrics['timers'][key].append(duration)
            # Keep only last 1000 measurements
            if len(self._metrics['timers'][key]) > 1000:
                self._metrics['timers'][key] = self._metrics['timers'][key][-1000:]
    
    def _make_key(self, name: str, labels: Optional[Dict[str, str]] = None) -> str:
        """Create a unique key for a metric."""
        if not labels:
            return name
        label_str = ",".join([f"{k}={v}" for k, v in sorted(labels.items())])
        return f"{name}[{label_str}]"
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics snapshot."""
        with self._lock:
            return {
                'timestamp': datetime.utcnow().isoformat(),
                'counters': dict(self._metrics['counters']),
                'gauges': dict(self._metrics['gauges']),
                'histograms': {
                    k: {
                        'count': len(v),
                        'sum': sum(v),
                        'avg': sum(v) / len(v) if v else 0,
                        'min': min(v) if v else 0,
                        'max': max(v) if v else 0,
                        'p50': self._percentile(v, 0.5),
                        'p95': self._percentile(v, 0.95),
                        'p99': self._percentile(v, 0.99)
                    }
                    for k, v in self._metrics['histograms'].items()
                },
                'timers': {
                    k: {
                        'count': len(v),
                        'avg_ms': sum(v) / len(v) * 1000 if v else 0,
                        'min_ms': min(v) * 1000 if v else 0,
                        'max_ms': max(v) * 1000 if v else 0,
                        'p50_ms': self._percentile(v, 0.5) * 1000,
                        'p95_ms': self._percentile(v, 0.95) * 1000,
                        'p99_ms': self._percentile(v, 0.99) * 1000
                    }
                    for k, v in self._metrics['timers'].items()
                }
            }
    
    def _percentile(self, values: List[float], p: float) -> float:
        """Calculate percentile of values."""
        if not values:
            return 0.0
        
        sorted_values = sorted(values)
        index = int(len(sorted_values) * p)
        if index >= len(sorted_values):
            return sorted_values[-1]
        return sorted_values[index]


class PerformanceTracker:
    """Track performance metrics and request traces."""
    
    def __init__(self, metrics: Metrics, logger: StructuredLogger):
        self.metrics = metrics
        self.logger = logger
        self.active_requests: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
    
    @asynccontextmanager
    async def trace_request(self, request_id: str, operation: str, **context):
        """Context manager for tracing requests."""
        start_time = time.time()
        
        trace_data = {
            'request_id': request_id,
            'operation': operation,
            'start_time': start_time,
            'context': context
        }
        
        with self._lock:
            self.active_requests[request_id] = trace_data
        
        self.logger.set_context(request_id=request_id, operation=operation)
        self.logger.info(f"Started {operation}", **context)
        
        try:
            yield trace_data
            
            # Success metrics
            duration = time.time() - start_time
            self.metrics.record_time(f"operation_duration", duration, {"operation": operation})
            self.metrics.increment(f"operation_success", labels={"operation": operation})
            
            self.logger.info(f"Completed {operation}", duration_seconds=duration, **context)
            
        except Exception as e:
            # Error metrics
            duration = time.time() - start_time
            self.metrics.increment(f"operation_error", labels={"operation": operation})
            
            self.logger.error(
                f"Failed {operation}", 
                error=str(e), 
                duration_seconds=duration, 
                **context
            )
            raise
            
        finally:
            with self._lock:
                self.active_requests.pop(request_id, None)
            self.logger.clear_context()


class SystemMonitor:
    """Monitor system resources and health."""
    
    def __init__(self, metrics: Metrics):
        self.metrics = metrics
        self.is_monitoring = False
        self._monitor_task = None
    
    def start_monitoring(self):
        """Start system monitoring in background."""
        if not self.is_monitoring:
            self.is_monitoring = True
            self._monitor_task = asyncio.create_task(self._monitor_loop())
    
    def stop_monitoring(self):
        """Stop system monitoring."""
        self.is_monitoring = False
        if self._monitor_task:
            self._monitor_task.cancel()
    
    async def _monitor_loop(self):
        """Main monitoring loop."""
        while self.is_monitoring:
            try:
                await self._collect_system_metrics()
                await asyncio.sleep(10)  # Collect every 10 seconds
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                await asyncio.sleep(10)
    
    async def _collect_system_metrics(self):
        """Collect system resource metrics."""
        if not PSUTIL_AVAILABLE:
            return  # Skip system metrics collection if psutil not available
            
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        self.metrics.set_gauge("system_cpu_percent", cpu_percent)
        
        # Memory metrics
        memory = psutil.virtual_memory()
        self.metrics.set_gauge("system_memory_percent", memory.percent)
        self.metrics.set_gauge("system_memory_used_bytes", memory.used)
        self.metrics.set_gauge("system_memory_available_bytes", memory.available)
        
        # Disk metrics
        disk = psutil.disk_usage('/')
        self.metrics.set_gauge("system_disk_percent", (disk.used / disk.total) * 100)
        self.metrics.set_gauge("system_disk_used_bytes", disk.used)
        self.metrics.set_gauge("system_disk_free_bytes", disk.free)
        
        # Network metrics (if available)
        try:
            network = psutil.net_io_counters()
            self.metrics.set_gauge("system_network_bytes_sent", network.bytes_sent)
            self.metrics.set_gauge("system_network_bytes_recv", network.bytes_recv)
        except:
            pass  # Network stats might not be available in all environments
        
        # Process metrics
        process = psutil.Process()
        self.metrics.set_gauge("process_memory_rss", process.memory_info().rss)
        self.metrics.set_gauge("process_cpu_percent", process.cpu_percent())
        self.metrics.set_gauge("process_num_threads", process.num_threads())


class HealthChecker:
    """Health check system for dependencies."""
    
    def __init__(self):
        self.checks: Dict[str, callable] = {}
        self.status_cache: Dict[str, Dict[str, Any]] = {}
        self.cache_ttl = 30  # Cache health status for 30 seconds
    
    def register_check(self, name: str, check_func: callable):
        """Register a health check function."""
        self.checks[name] = check_func
    
    async def check_health(self) -> Dict[str, Any]:
        """Run all health checks."""
        results = {}
        overall_status = HealthStatus.HEALTHY
        
        for name, check_func in self.checks.items():
            try:
                # Check cache first
                cached_result = self._get_cached_result(name)
                if cached_result:
                    results[name] = cached_result
                    continue
                
                start_time = time.time()
                if asyncio.iscoroutinefunction(check_func):
                    result = await check_func()
                else:
                    result = check_func()
                
                duration = time.time() - start_time
                
                check_result = {
                    'status': result.get('status', HealthStatus.HEALTHY),
                    'message': result.get('message', 'OK'),
                    'duration_seconds': duration,
                    'timestamp': datetime.utcnow().isoformat(),
                    'details': result.get('details', {})
                }
                
                results[name] = check_result
                self._cache_result(name, check_result)
                
                # Update overall status
                if check_result['status'] == HealthStatus.UNHEALTHY:
                    overall_status = HealthStatus.UNHEALTHY
                elif check_result['status'] == HealthStatus.DEGRADED and overall_status == HealthStatus.HEALTHY:
                    overall_status = HealthStatus.DEGRADED
                    
            except Exception as e:
                error_result = {
                    'status': HealthStatus.UNHEALTHY,
                    'message': f'Health check failed: {str(e)}',
                    'timestamp': datetime.utcnow().isoformat(),
                    'error': str(e)
                }
                results[name] = error_result
                overall_status = HealthStatus.UNHEALTHY
        
        return {
            'status': overall_status,
            'timestamp': datetime.utcnow().isoformat(),
            'checks': results
        }
    
    def _get_cached_result(self, name: str) -> Optional[Dict[str, Any]]:
        """Get cached health check result if still valid."""
        if name in self.status_cache:
            cached = self.status_cache[name]
            cache_time = datetime.fromisoformat(cached['timestamp'].replace('Z', '+00:00'))
            if (datetime.utcnow() - cache_time.replace(tzinfo=None)).seconds < self.cache_ttl:
                return cached
        return None
    
    def _cache_result(self, name: str, result: Dict[str, Any]):
        """Cache health check result."""
        self.status_cache[name] = result


if FASTAPI_MIDDLEWARE_AVAILABLE:
    class RequestTracingMiddleware(BaseHTTPMiddleware):
        """Middleware for request tracing and performance monitoring."""
        
        def __init__(self, app, monitor: 'SystemMonitor'):
            super().__init__(app)
            self.monitor = monitor
        
        async def dispatch(self, request, call_next):
            """Process request with performance tracking."""
            start_time = time.time()
            
            # Generate request ID
            request_id = f"req_{int(time.time() * 1000)}"
            
            # Track request start
            self.monitor.track_request_start(request_id, request.method, str(request.url))
            
            try:
                response = await call_next(request)
                
                # Calculate processing time
                process_time = time.time() - start_time
                
                # Track request completion
                self.monitor.track_request_end(
                    request_id, 
                    response.status_code, 
                    process_time
                )
                
                # Add performance headers
                response.headers["X-Process-Time"] = str(process_time)
                response.headers["X-Request-ID"] = request_id
                
                return response
                
            except Exception as e:
                # Track request error
                process_time = time.time() - start_time
                self.monitor.track_request_error(request_id, str(e), process_time)
                raise
else:
    # Placeholder when middleware isn't available
    class RequestTracingMiddleware:
        def __init__(self, app, monitor):
            self.app = app
            self.monitor = monitor


# Global monitoring instances
metrics = Metrics()
structured_logger = StructuredLogger("edgp_ai_model")
performance_tracker = PerformanceTracker(metrics, structured_logger)
system_monitor = SystemMonitor(metrics)
health_checker = HealthChecker()


# Default health checks
def check_memory_usage():
    """Check memory usage."""
    if not PSUTIL_AVAILABLE:
        return {
            'status': HealthStatus.HEALTHY,
            'message': 'Memory check skipped - psutil not available',
            'details': {'psutil_available': False}
        }
    
    memory = psutil.virtual_memory()
    if memory.percent > 90:
        return {
            'status': HealthStatus.UNHEALTHY,
            'message': f'Memory usage critical: {memory.percent}%',
            'details': {'memory_percent': memory.percent}
        }
    elif memory.percent > 80:
        return {
            'status': HealthStatus.DEGRADED,
            'message': f'Memory usage high: {memory.percent}%',
            'details': {'memory_percent': memory.percent}
        }
    
    return {
        'status': HealthStatus.HEALTHY,
        'message': f'Memory usage normal: {memory.percent}%',
        'details': {'memory_percent': memory.percent}
    }


def check_disk_space():
    """Check disk space."""
    if not PSUTIL_AVAILABLE:
        return {
            'status': HealthStatus.HEALTHY,
            'message': 'Disk check skipped - psutil not available',
            'details': {'psutil_available': False}
        }
        
    disk = psutil.disk_usage('/')
    usage_percent = (disk.used / disk.total) * 100
    
    if usage_percent > 95:
        return {
            'status': HealthStatus.UNHEALTHY,
            'message': f'Disk space critical: {usage_percent:.1f}%',
            'details': {'disk_percent': usage_percent}
        }
    elif usage_percent > 85:
        return {
            'status': HealthStatus.DEGRADED,
            'message': f'Disk space low: {usage_percent:.1f}%',
            'details': {'disk_percent': usage_percent}
        }
    
    return {
        'status': HealthStatus.HEALTHY,
        'message': f'Disk space normal: {usage_percent:.1f}%',
        'details': {'disk_percent': usage_percent}
    }


# Register default health checks
health_checker.register_check('memory', check_memory_usage)
health_checker.register_check('disk', check_disk_space)
