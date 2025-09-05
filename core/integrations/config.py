"""
External Service Integration Configuration
Defines patterns, mappings, and configurations for external microservice interactions.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime
import os

from .patterns import IntegrationPattern


class ExternalServiceType(Enum):
    """Types of external services."""
    DATA_VALIDATION = "data_validation"
    REGULATORY_COMPLIANCE = "regulatory_compliance"
    AUTOMATION_ENGINE = "automation_engine"
    BUSINESS_INTELLIGENCE = "business_intelligence"
    GOVERNANCE_PLATFORM = "governance_platform"
    NOTIFICATION_SERVICE = "notification_service"
    AUDIT_LOGGING = "audit_logging"


@dataclass
class ExternalServiceConfig:
    """Configuration for external service integration."""
    service_name: str
    service_type: ExternalServiceType
    base_url: str
    integration_pattern: IntegrationPattern
    endpoints: Dict[str, str]
    headers: Dict[str, str]
    authentication: Dict[str, Any]
    timeout_seconds: int = 300
    retry_count: int = 3
    health_check_endpoint: str = "/health"
    webhook_config: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.webhook_config is None:
            self.webhook_config = {}


class IntegrationConfigManager:
    """Manages configuration for external service integrations."""
    
    def __init__(self):
        self.service_configs: Dict[str, ExternalServiceConfig] = {}
        self.agent_service_mappings: Dict[str, List[str]] = {}
        self.load_default_configurations()
    
    def load_default_configurations(self):
        """Load default external service configurations."""
        
        # Data Validation Service
        self.register_service_config(ExternalServiceConfig(
            service_name="data_validator_service",
            service_type=ExternalServiceType.DATA_VALIDATION,
            base_url=os.getenv("DATA_VALIDATOR_URL", "http://localhost:8001"),
            integration_pattern=IntegrationPattern.SYNC_API,
            endpoints={
                "validate_schema": "/api/v1/validate/schema",
                "validate_quality": "/api/v1/validate/quality",
                "validate_integrity": "/api/v1/validate/integrity"
            },
            headers={
                "Content-Type": "application/json",
                "X-Service-Name": "edgp-ai-model"
            },
            authentication={
                "type": "api_key",
                "header": "X-API-Key",
                "value": os.getenv("DATA_VALIDATOR_API_KEY", "")
            }
        ))
        
        # Regulatory Compliance Service
        self.register_service_config(ExternalServiceConfig(
            service_name="regulatory_service",
            service_type=ExternalServiceType.REGULATORY_COMPLIANCE,
            base_url=os.getenv("REGULATORY_SERVICE_URL", "http://localhost:8002"),
            integration_pattern=IntegrationPattern.ASYNC_API,
            endpoints={
                "compliance_check": "/api/v1/compliance/check",
                "regulation_lookup": "/api/v1/regulations/lookup",
                "violation_report": "/api/v1/violations/report"
            },
            headers={
                "Content-Type": "application/json",
                "X-Service-Name": "edgp-ai-model"
            },
            authentication={
                "type": "bearer_token",
                "header": "Authorization",
                "value": os.getenv("REGULATORY_SERVICE_TOKEN", "")
            },
            webhook_config={
                "callback_path": "/webhooks/regulatory_service/callback",
                "authentication_required": True
            }
        ))
        
        # Automation Engine Service
        self.register_service_config(ExternalServiceConfig(
            service_name="automation_engine",
            service_type=ExternalServiceType.AUTOMATION_ENGINE,
            base_url=os.getenv("AUTOMATION_ENGINE_URL", "http://localhost:8003"),
            integration_pattern=IntegrationPattern.MESSAGE_QUEUE,
            endpoints={
                "execute_remediation": "/api/v1/remediation/execute",
                "schedule_task": "/api/v1/tasks/schedule",
                "get_task_status": "/api/v1/tasks/status"
            },
            headers={
                "Content-Type": "application/json",
                "X-Service-Name": "edgp-ai-model"
            },
            authentication={
                "type": "api_key",
                "header": "X-API-Key",
                "value": os.getenv("AUTOMATION_ENGINE_API_KEY", "")
            }
        ))
        
        # Business Intelligence Service
        self.register_service_config(ExternalServiceConfig(
            service_name="bi_analytics_service",
            service_type=ExternalServiceType.BUSINESS_INTELLIGENCE,
            base_url=os.getenv("BI_SERVICE_URL", "http://localhost:8004"),
            integration_pattern=IntegrationPattern.SYNC_API,
            endpoints={
                "generate_report": "/api/v1/reports/generate",
                "create_dashboard": "/api/v1/dashboards/create",
                "export_data": "/api/v1/data/export"
            },
            headers={
                "Content-Type": "application/json",
                "X-Service-Name": "edgp-ai-model"
            },
            authentication={
                "type": "oauth2",
                "token_url": "/api/v1/auth/token",
                "client_id": os.getenv("BI_SERVICE_CLIENT_ID", ""),
                "client_secret": os.getenv("BI_SERVICE_CLIENT_SECRET", "")
            }
        ))
        
        # Governance Platform Service
        self.register_service_config(ExternalServiceConfig(
            service_name="governance_platform",
            service_type=ExternalServiceType.GOVERNANCE_PLATFORM,
            base_url=os.getenv("GOVERNANCE_PLATFORM_URL", "http://localhost:8005"),
            integration_pattern=IntegrationPattern.ASYNC_API,
            endpoints={
                "validate_policy": "/api/v1/policies/validate",
                "register_policy": "/api/v1/policies/register",
                "compliance_scan": "/api/v1/compliance/scan"
            },
            headers={
                "Content-Type": "application/json",
                "X-Service-Name": "edgp-ai-model"
            },
            authentication={
                "type": "mutual_tls",
                "cert_path": os.getenv("GOVERNANCE_CERT_PATH", ""),
                "key_path": os.getenv("GOVERNANCE_KEY_PATH", "")
            },
            webhook_config={
                "callback_path": "/webhooks/governance_platform/callback",
                "authentication_required": True
            }
        ))
        
        # Notification Service
        self.register_service_config(ExternalServiceConfig(
            service_name="notification_service",
            service_type=ExternalServiceType.NOTIFICATION_SERVICE,
            base_url=os.getenv("NOTIFICATION_SERVICE_URL", "http://localhost:8006"),
            integration_pattern=IntegrationPattern.EVENT_STREAMING,
            endpoints={
                "send_alert": "/api/v1/alerts/send",
                "create_notification": "/api/v1/notifications/create",
                "manage_subscriptions": "/api/v1/subscriptions/manage"
            },
            headers={
                "Content-Type": "application/json",
                "X-Service-Name": "edgp-ai-model"
            },
            authentication={
                "type": "api_key",
                "header": "X-API-Key",
                "value": os.getenv("NOTIFICATION_SERVICE_API_KEY", "")
            }
        ))
        
        # Setup agent-service mappings
        self._setup_agent_service_mappings()
    
    def _setup_agent_service_mappings(self):
        """Setup which services each agent type should use."""
        self.agent_service_mappings = {
            "data_quality_agent": [
                "data_validator_service",
                "notification_service"
            ],
            "compliance_agent": [
                "regulatory_service",
                "governance_platform",
                "notification_service"
            ],
            "remediation_agent": [
                "automation_engine",
                "notification_service"
            ],
            "analytics_agent": [
                "bi_analytics_service",
                "notification_service"
            ],
            "policy_agent": [
                "governance_platform",
                "regulatory_service",
                "notification_service"
            ]
        }
    
    def register_service_config(self, config: ExternalServiceConfig):
        """Register a new service configuration."""
        self.service_configs[config.service_name] = config
    
    def get_service_config(self, service_name: str) -> Optional[ExternalServiceConfig]:
        """Get service configuration by name."""
        return self.service_configs.get(service_name)
    
    def get_agent_services(self, agent_id: str) -> List[str]:
        """Get list of services for a specific agent."""
        return self.agent_service_mappings.get(agent_id, [])
    
    def get_all_service_configs(self) -> Dict[str, ExternalServiceConfig]:
        """Get all service configurations."""
        return self.service_configs.copy()
    
    def generate_service_registry_data(self) -> List[Dict[str, Any]]:
        """Generate data for service registry initialization."""
        registry_data = []
        
        for service_name, config in self.service_configs.items():
            registry_data.append({
                'service_name': service_name,
                'base_url': config.base_url,
                'pattern': config.integration_pattern,
                'config': {
                    'headers': config.headers,
                    'authentication': config.authentication,
                    'timeout': config.timeout_seconds,
                    'retry_count': config.retry_count,
                    'endpoints': config.endpoints,
                    'webhook_config': config.webhook_config
                }
            })
        
        return registry_data


class IntegrationPatternTemplates:
    """Templates for common integration patterns."""
    
    @staticmethod
    def get_sync_api_template() -> Dict[str, Any]:
        """Template for synchronous API integration."""
        return {
            "pattern": IntegrationPattern.SYNC_API.value,
            "timeout": 30,
            "retry_count": 3,
            "response_format": "json",
            "error_handling": {
                "retry_on_status": [500, 502, 503, 504],
                "circuit_breaker_threshold": 5
            }
        }
    
    @staticmethod
    def get_async_api_template() -> Dict[str, Any]:
        """Template for asynchronous API integration."""
        return {
            "pattern": IntegrationPattern.ASYNC_API.value,
            "initiation_timeout": 30,
            "callback_timeout": 300,
            "callback_required": True,
            "status_polling": {
                "enabled": True,
                "interval_seconds": 10,
                "max_polls": 30
            }
        }
    
    @staticmethod
    def get_message_queue_template() -> Dict[str, Any]:
        """Template for message queue integration."""
        return {
            "pattern": IntegrationPattern.MESSAGE_QUEUE.value,
            "queue_config": {
                "visibility_timeout": 300,
                "message_retention_period": 1209600,  # 14 days
                "delay_seconds": 0,
                "max_receive_count": 3
            },
            "response_queue": True,
            "dlq_enabled": True
        }
    
    @staticmethod
    def get_webhook_template() -> Dict[str, Any]:
        """Template for webhook integration."""
        return {
            "pattern": IntegrationPattern.WEBHOOK_CALLBACK.value,
            "webhook_config": {
                "authentication_required": True,
                "signature_verification": True,
                "retry_failed_webhooks": True,
                "max_retries": 3
            }
        }


# Default configuration for EDGP AI Model
DEFAULT_EXTERNAL_SERVICES_CONFIG = {
    "data_quality_agent": {
        "type": "data_quality",
        "external_services": ["data_validator_service", "notification_service"],
        "llm_preferences": {
            "temperature": 0.3,
            "max_tokens": 1500,
            "preferred_model": "claude-3-sonnet"
        },
        "external_service_configs": {
            "data_validator_service": {
                "validation_endpoint": "/api/v1/validate/quality",
                "pattern": "sync_api"
            }
        }
    },
    
    "compliance_agent": {
        "type": "compliance",
        "external_services": ["regulatory_service", "governance_platform", "notification_service"],
        "llm_preferences": {
            "temperature": 0.2,
            "max_tokens": 2000,
            "preferred_model": "claude-3-sonnet"
        },
        "external_service_configs": {
            "regulatory_service": {
                "compliance_endpoint": "/api/v1/compliance/check",
                "pattern": "async_api"
            },
            "governance_platform": {
                "policy_validation_endpoint": "/api/v1/policies/validate",
                "pattern": "async_api"
            }
        }
    },
    
    "remediation_agent": {
        "type": "remediation",
        "external_services": ["automation_engine", "notification_service"],
        "llm_preferences": {
            "temperature": 0.2,
            "max_tokens": 1800,
            "preferred_model": "claude-3-sonnet"
        },
        "external_service_configs": {
            "automation_engine": {
                "remediation_endpoint": "/api/v1/remediation/execute",
                "pattern": "message_queue"
            }
        }
    },
    
    "analytics_agent": {
        "type": "analytics",
        "external_services": ["bi_analytics_service", "notification_service"],
        "llm_preferences": {
            "temperature": 0.4,
            "max_tokens": 2500,
            "preferred_model": "claude-3-sonnet"
        },
        "external_service_configs": {
            "bi_analytics_service": {
                "analytics_endpoint": "/api/v1/reports/generate",
                "pattern": "sync_api"
            }
        }
    },
    
    "policy_agent": {
        "type": "policy",
        "external_services": ["governance_platform", "regulatory_service", "notification_service"],
        "llm_preferences": {
            "temperature": 0.3,
            "max_tokens": 3000,
            "preferred_model": "claude-3-sonnet"
        },
        "external_service_configs": {
            "governance_platform": {
                "policy_validation_endpoint": "/api/v1/policies/validate",
                "pattern": "async_api"
            }
        }
    }
}


class IntegrationWorkflowTemplates:
    """Pre-defined workflow templates for common integration scenarios."""
    
    @staticmethod
    def get_data_quality_assessment_workflow() -> Dict[str, Any]:
        """Workflow for comprehensive data quality assessment."""
        return {
            "name": "comprehensive_data_quality_assessment",
            "description": "Multi-step data quality assessment using LLM and external validators",
            "steps": [
                {
                    "name": "initial_llm_assessment",
                    "type": "llm",
                    "agent": "data_quality_agent",
                    "operation": "data_quality_assessment",
                    "config": {"temperature": 0.3}
                },
                {
                    "name": "external_schema_validation",
                    "type": "external_service",
                    "service": "data_validator_service",
                    "endpoint": "validate_schema",
                    "pattern": "sync_api"
                },
                {
                    "name": "external_quality_validation",
                    "type": "external_service",
                    "service": "data_validator_service",
                    "endpoint": "validate_quality",
                    "pattern": "sync_api"
                },
                {
                    "name": "final_llm_synthesis",
                    "type": "llm",
                    "agent": "data_quality_agent",
                    "operation": "quality_synthesis",
                    "config": {"temperature": 0.2}
                }
            ],
            "error_handling": {
                "continue_on_external_failure": True,
                "fallback_to_llm_only": True
            }
        }
    
    @staticmethod
    def get_compliance_assessment_workflow() -> Dict[str, Any]:
        """Workflow for regulatory compliance assessment."""
        return {
            "name": "regulatory_compliance_assessment",
            "description": "Multi-service compliance assessment with regulatory APIs",
            "steps": [
                {
                    "name": "llm_initial_analysis",
                    "type": "llm",
                    "agent": "compliance_agent",
                    "operation": "compliance_assessment"
                },
                {
                    "name": "regulatory_api_check",
                    "type": "external_service",
                    "service": "regulatory_service",
                    "endpoint": "compliance_check",
                    "pattern": "async_api"
                },
                {
                    "name": "governance_validation",
                    "type": "external_service",
                    "service": "governance_platform",
                    "endpoint": "compliance_scan",
                    "pattern": "async_api"
                },
                {
                    "name": "compliance_report_generation",
                    "type": "llm",
                    "agent": "compliance_agent",
                    "operation": "compliance_reporting"
                }
            ]
        }
    
    @staticmethod
    def get_remediation_execution_workflow() -> Dict[str, Any]:
        """Workflow for automated remediation execution."""
        return {
            "name": "automated_remediation_execution",
            "description": "LLM-planned remediation with automation engine execution",
            "steps": [
                {
                    "name": "remediation_planning",
                    "type": "llm",
                    "agent": "remediation_agent",
                    "operation": "remediation_planning"
                },
                {
                    "name": "automation_execution",
                    "type": "external_service",
                    "service": "automation_engine",
                    "endpoint": "execute_remediation",
                    "pattern": "message_queue"
                },
                {
                    "name": "execution_monitoring",
                    "type": "external_service",
                    "service": "automation_engine",
                    "endpoint": "get_task_status",
                    "pattern": "sync_api"
                }
            ]
        }


class SharedUtilityFunctions:
    """Shared utility functions for external service integration."""
    
    @staticmethod
    def create_service_request_payload(
        agent_id: str,
        operation: str,
        data: Dict[str, Any],
        correlation_id: str,
        metadata: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Create standardized payload for external service requests."""
        return {
            "request_metadata": {
                "agent_id": agent_id,
                "operation": operation,
                "correlation_id": correlation_id,
                "timestamp": datetime.utcnow().isoformat(),
                "service_version": "2.0.0"
            },
            "data": data,
            "additional_metadata": metadata or {}
        }
    
    @staticmethod
    def extract_service_response_data(response: Dict[str, Any]) -> Dict[str, Any]:
        """Extract and standardize response data from external services."""
        if "data" in response:
            return response["data"]
        elif "result" in response:
            return response["result"]
        elif "response" in response:
            return response["response"]
        else:
            return response
    
    @staticmethod
    def create_error_response(
        agent_id: str,
        operation: str,
        error_message: str,
        error_code: str = "INTEGRATION_ERROR"
    ) -> Dict[str, Any]:
        """Create standardized error response."""
        return {
            "success": False,
            "error": {
                "code": error_code,
                "message": error_message,
                "agent_id": agent_id,
                "operation": operation,
                "timestamp": datetime.utcnow().isoformat()
            }
        }
    
    @staticmethod
    def validate_service_response_schema(
        response: Dict[str, Any],
        expected_fields: List[str],
        optional_fields: List[str] = None
    ) -> Dict[str, Any]:
        """Validate external service response schema."""
        optional_fields = optional_fields or []
        validation_result = {
            "valid": True,
            "missing_required_fields": [],
            "missing_optional_fields": [],
            "extra_fields": []
        }
        
        response_keys = set(response.keys())
        required_keys = set(expected_fields)
        optional_keys = set(optional_fields)
        
        # Check required fields
        missing_required = required_keys - response_keys
        if missing_required:
            validation_result["valid"] = False
            validation_result["missing_required_fields"] = list(missing_required)
        
        # Check optional fields
        missing_optional = optional_keys - response_keys
        validation_result["missing_optional_fields"] = list(missing_optional)
        
        # Check for extra fields
        expected_keys = required_keys | optional_keys
        extra_fields = response_keys - expected_keys
        validation_result["extra_fields"] = list(extra_fields)
        
        return validation_result


# Global configuration manager
integration_config_manager = IntegrationConfigManager()

# Export configuration and utilities
__all__ = [
    'ExternalServiceType',
    'ExternalServiceConfig', 
    'IntegrationConfigManager',
    'IntegrationWorkflowTemplates',
    'SharedUtilityFunctions',
    'DEFAULT_EXTERNAL_SERVICES_CONFIG',
    'integration_config_manager'
]
