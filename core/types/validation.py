"""
Type validation utilities for ensuring data consistency across agents.
"""

from typing import Any, Dict, List, Optional, Type, TypeVar, Union
from pydantic import BaseModel, ValidationError
import logging

logger = logging.getLogger(__name__)

T = TypeVar('T', bound=BaseModel)


class TypeValidator:
    """Utility class for validating and converting types across agents."""
    
    @staticmethod
    def validate_input(data: Dict[str, Any], model_class: Type[T]) -> T:
        """
        Validate input data against a Pydantic model.
        
        Args:
            data: Input data dictionary
            model_class: Pydantic model class to validate against
            
        Returns:
            Validated model instance
            
        Raises:
            ValidationError: If validation fails
        """
        try:
            return model_class(**data)
        except ValidationError as e:
            logger.error(f"Validation failed for {model_class.__name__}: {e}")
            raise
    
    @staticmethod
    def validate_output(instance: BaseModel) -> Dict[str, Any]:
        """
        Validate and serialize output from an agent.
        
        Args:
            instance: Pydantic model instance
            
        Returns:
            Serialized dictionary
        """
        try:
            return instance.dict(exclude_unset=True)
        except Exception as e:
            logger.error(f"Output serialization failed: {e}")
            raise
    
    @staticmethod
    def convert_legacy_format(
        legacy_data: Dict[str, Any], 
        target_model: Type[T],
        field_mapping: Optional[Dict[str, str]] = None
    ) -> T:
        """
        Convert legacy data format to standardized types.
        
        Args:
            legacy_data: Data in legacy format
            target_model: Target Pydantic model
            field_mapping: Optional mapping of old field names to new ones
            
        Returns:
            Converted model instance
        """
        if field_mapping:
            converted_data = {}
            for old_field, new_field in field_mapping.items():
                if old_field in legacy_data:
                    converted_data[new_field] = legacy_data[old_field]
            
            # Add unmapped fields
            for key, value in legacy_data.items():
                if key not in field_mapping:
                    converted_data[key] = value
        else:
            converted_data = legacy_data
        
        return TypeValidator.validate_input(converted_data, target_model)
    
    @staticmethod
    def ensure_agent_compatibility(
        source_agent_output: Dict[str, Any],
        target_agent_input_type: Type[T]
    ) -> T:
        """
        Ensure output from one agent is compatible with input to another agent.
        
        Args:
            source_agent_output: Output from source agent
            target_agent_input_type: Expected input type for target agent
            
        Returns:
            Compatible input for target agent
        """
        try:
            return TypeValidator.validate_input(source_agent_output, target_agent_input_type)
        except ValidationError as e:
            logger.warning(f"Direct conversion failed, attempting field mapping: {e}")
            
            # Attempt common field mappings
            common_mappings = {
                "data": "input_data",
                "results": "data",
                "output": "data",
                "response": "data"
            }
            
            return TypeValidator.convert_legacy_format(
                source_agent_output, 
                target_agent_input_type, 
                common_mappings
            )


class AgentTypeRegistry:
    """Registry for agent input/output types."""
    
    _agent_types = {
        "policy_suggestion": {
            "request": "PolicySuggestionRequest",
            "response": "PolicySuggestionResponse"
        },
        "data_privacy_compliance": {
            "request": "ComplianceRequest", 
            "response": "ComplianceResponse"
        },
        "data_quality": {
            "request": "DataQualityRequest",
            "response": "DataQualityResponse"
        },
        "data_remediation": {
            "request": "RemediationRequest",
            "response": "RemediationResponse"
        },
        "analytics": {
            "request": "AnalyticsRequest",
            "response": "AnalyticsResponse"
        }
    }
    
    @classmethod
    def get_request_type(cls, agent_name: str) -> str:
        """Get request type for an agent."""
        return cls._agent_types.get(agent_name, {}).get("request")
    
    @classmethod
    def get_response_type(cls, agent_name: str) -> str:
        """Get response type for an agent."""
        return cls._agent_types.get(agent_name, {}).get("response")
    
    @classmethod
    def list_agents(cls) -> List[str]:
        """List all registered agent types."""
        return list(cls._agent_types.keys())


# Helper functions for common type operations

def create_standard_request(
    agent_type: str,
    request_data: Dict[str, Any],
    user_context: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Create a standardized request for any agent.
    
    Args:
        agent_type: Type of agent
        request_data: Request payload
        user_context: Optional user context
        
    Returns:
        Standardized request dictionary
    """
    import uuid
    from datetime import datetime
    
    return {
        "request_id": str(uuid.uuid4()),
        "agent_type": agent_type,
        "timestamp": datetime.utcnow().isoformat(),
        "user_id": user_context.get("user_id") if user_context else None,
        "session_id": user_context.get("session_id") if user_context else None,
        "context": user_context or {},
        **request_data
    }


def create_standard_response(
    request_id: str,
    agent_type: str,
    success: bool,
    data: Any,
    processing_time_ms: Optional[float] = None,
    confidence: Optional[str] = None,
    errors: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Create a standardized response from any agent.
    
    Args:
        request_id: Original request ID
        agent_type: Type of agent responding
        success: Whether the operation was successful
        data: Response data
        processing_time_ms: Processing time in milliseconds
        confidence: Confidence level of the response
        errors: List of errors if any
        
    Returns:
        Standardized response dictionary
    """
    from datetime import datetime
    
    base_response = {
        "request_id": request_id,
        "agent_type": agent_type,
        "timestamp": datetime.utcnow().isoformat(),
        "success": success,
        "processing_time_ms": processing_time_ms,
        "confidence": confidence,
        "metadata": {
            "errors": errors or []
        }
    }
    
    # Merge with agent-specific data
    if isinstance(data, dict):
        base_response.update(data)
    else:
        base_response["data"] = data
    
    return base_response
