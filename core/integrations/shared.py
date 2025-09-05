"""
Shared Agent Integration Utilities
Common functions and patterns for agent interactions with external services and LLM gateway.
"""

import asyncio
import json
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass
from datetime import datetime
import logging

from ..services.llm_bridge import (
    LLMGatewayBridge,
    AgentLLMInterface,
    LLMRequest,
    PromptTemplate,
    ResponseFormat
)
from .patterns import (
    IntegrationOrchestrator,
    IntegrationPattern,
    IntegrationRequest,
    SharedIntegrationFunctions
)
from ..communication.external import ExternalCommunicationManager

logger = logging.getLogger(__name__)


@dataclass
class AgentContext:
    """Context information for agent operations."""
    agent_id: str
    agent_type: str
    current_task: Optional[str] = None
    external_services: List[str] = None
    llm_preferences: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.external_services is None:
            self.external_services = []
        if self.llm_preferences is None:
            self.llm_preferences = {}


class AgentIntegrationHelper:
    """Helper class for agent integration with external services and LLM."""
    
    def __init__(
        self,
        agent_context: AgentContext,
        llm_bridge: LLMGatewayBridge,
        integration_orchestrator: IntegrationOrchestrator,
        external_comm: ExternalCommunicationManager
    ):
        self.context = agent_context
        self.llm_interface = AgentLLMInterface(agent_context.agent_id, llm_bridge)
        self.integration_orchestrator = integration_orchestrator
        self.external_comm = external_comm
        self.operation_history: List[Dict[str, Any]] = []
    
    async def process_with_llm(
        self,
        data: Dict[str, Any],
        operation_type: str,
        custom_prompt: Optional[str] = None,
        **llm_kwargs
    ) -> Dict[str, Any]:
        """Process data using LLM with agent-specific patterns."""
        try:
            # Determine appropriate template
            template = self._get_template_for_operation(operation_type)
            
            # Prepare prompt
            if custom_prompt:
                prompt = custom_prompt
            else:
                prompt = self._generate_default_prompt(data, operation_type)
            
            # Set LLM preferences from context
            llm_params = {
                'temperature': self.context.llm_preferences.get('temperature', 0.7),
                'max_tokens': self.context.llm_preferences.get('max_tokens', 1000),
                **llm_kwargs
            }
            
            # Generate response
            response = await self.llm_interface.generate_response(
                prompt=prompt,
                template=template,
                response_format=ResponseFormat.JSON,
                **llm_params
            )
            
            # Log operation
            self._log_operation('llm_processing', {
                'operation_type': operation_type,
                'success': response.success,
                'processing_time_ms': response.processing_time_ms
            })
            
            if not response.success:
                raise Exception(f"LLM processing failed: {response.error}")
            
            # Parse JSON response
            try:
                result = json.loads(response.content)
            except json.JSONDecodeError:
                result = {"raw_response": response.content}
            
            return {
                'success': True,
                'data': result,
                'metadata': {
                    'agent_id': self.context.agent_id,
                    'operation_type': operation_type,
                    'request_id': response.request_id,
                    'processing_time_ms': response.processing_time_ms
                }
            }
            
        except Exception as e:
            logger.error(f"LLM processing failed for {self.context.agent_id}: {e}")
            self._log_operation('llm_processing', {
                'operation_type': operation_type,
                'success': False,
                'error': str(e)
            })
            
            return {
                'success': False,
                'error': str(e),
                'metadata': {'agent_id': self.context.agent_id, 'operation_type': operation_type}
            }
    
    async def call_external_service(
        self,
        service_name: str,
        endpoint: str,
        payload: Dict[str, Any],
        pattern: IntegrationPattern = IntegrationPattern.SYNC_API,
        **kwargs
    ) -> Dict[str, Any]:
        """Call external service with standardized error handling."""
        try:
            # Standardize payload
            standardized_payload = SharedIntegrationFunctions.standardize_payload(
                payload,
                kwargs.get('service_schema')
            )
            
            # Create correlation ID
            correlation_id = SharedIntegrationFunctions.create_correlation_id()
            
            # Make request
            result = await self.integration_orchestrator.make_request(
                service_name=service_name,
                pattern=pattern,
                endpoint=endpoint,
                payload=standardized_payload,
                request_id=correlation_id,
                **kwargs
            )
            
            # Handle different response types
            if isinstance(result, str):
                # Async pattern - correlation ID returned
                self._log_operation('external_service_call', {
                    'service_name': service_name,
                    'pattern': pattern.value,
                    'correlation_id': result,
                    'async': True
                })
                
                return {
                    'success': True,
                    'async': True,
                    'correlation_id': result,
                    'service_name': service_name,
                    'pattern': pattern.value
                }
            else:
                # Sync pattern - direct response
                self._log_operation('external_service_call', {
                    'service_name': service_name,
                    'pattern': pattern.value,
                    'success': result.status.value == 'completed',
                    'processing_time_ms': result.processing_time_ms
                })
                
                return {
                    'success': result.status.value == 'completed',
                    'data': result.response_data,
                    'error': result.error_message,
                    'service_name': service_name,
                    'processing_time_ms': result.processing_time_ms,
                    'request_id': result.request_id
                }
                
        except Exception as e:
            logger.error(f"External service call failed for {self.context.agent_id}: {e}")
            self._log_operation('external_service_call', {
                'service_name': service_name,
                'success': False,
                'error': str(e)
            })
            
            return {
                'success': False,
                'error': str(e),
                'service_name': service_name,
                'agent_id': self.context.agent_id
            }
    
    async def send_mq_message(
        self,
        target_service: str,
        message_data: Dict[str, Any],
        priority: str = "normal"
    ) -> Dict[str, Any]:
        """Send message via message queue to external service."""
        try:
            # Use external communication manager for MQ
            await self.external_comm.send_message(target_service, message_data, priority=priority)
            
            message_id = SharedIntegrationFunctions.create_correlation_id()
            
            self._log_operation('mq_message_sent', {
                'target_service': target_service,
                'message_id': message_id,
                'priority': priority
            })
            
            return {
                'success': True,
                'message_id': message_id,
                'target_service': target_service,
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"MQ message sending failed for {self.context.agent_id}: {e}")
            self._log_operation('mq_message_sent', {
                'target_service': target_service,
                'success': False,
                'error': str(e)
            })
            
            return {
                'success': False,
                'error': str(e),
                'target_service': target_service
            }
    
    async def broadcast_event(
        self,
        event_topic: str,
        event_data: Dict[str, Any],
        priority: str = "normal"
    ) -> Dict[str, Any]:
        """Broadcast event to external services via SNS."""
        try:
            await self.external_comm.send_event(event_topic, event_data, priority)
            
            event_id = SharedIntegrationFunctions.create_correlation_id()
            
            self._log_operation('event_broadcast', {
                'event_topic': event_topic,
                'event_id': event_id,
                'priority': priority
            })
            
            return {
                'success': True,
                'event_id': event_id,
                'topic': event_topic,
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Event broadcasting failed for {self.context.agent_id}: {e}")
            self._log_operation('event_broadcast', {
                'event_topic': event_topic,
                'success': False,
                'error': str(e)
            })
            
            return {
                'success': False,
                'error': str(e),
                'event_topic': event_topic
            }
    
    async def coordinate_multi_service_operation(
        self,
        services: List[Dict[str, Any]],
        aggregation_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """Coordinate operation across multiple external services."""
        try:
            task_id = SharedIntegrationFunctions.create_correlation_id()
            request_ids = []
            
            # Set up aggregation
            if aggregation_callback:
                self.integration_orchestrator.result_aggregator.create_aggregation_task(
                    task_id,
                    [],  # Will be populated as requests are made
                    aggregation_callback
                )
            
            # Make requests to all services
            for service_config in services:
                result = await self.call_external_service(
                    service_name=service_config['service_name'],
                    endpoint=service_config['endpoint'],
                    payload=service_config['payload'],
                    pattern=IntegrationPattern(service_config.get('pattern', 'sync_api'))
                )
                
                if result.get('correlation_id'):
                    request_ids.append(result['correlation_id'])
                elif result.get('request_id'):
                    request_ids.append(result['request_id'])
            
            self._log_operation('multi_service_operation', {
                'task_id': task_id,
                'services_count': len(services),
                'request_ids': request_ids
            })
            
            return {
                'success': True,
                'task_id': task_id,
                'service_count': len(services),
                'request_ids': request_ids,
                'aggregation_enabled': bool(aggregation_callback)
            }
            
        except Exception as e:
            logger.error(f"Multi-service coordination failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'task_id': task_id if 'task_id' in locals() else None
            }
    
    def _get_template_for_operation(self, operation_type: str) -> Optional[PromptTemplate]:
        """Get appropriate prompt template for operation type."""
        operation_templates = {
            'data_quality': PromptTemplate.DATA_QUALITY,
            'quality_assessment': PromptTemplate.DATA_QUALITY,
            'compliance_check': PromptTemplate.COMPLIANCE,
            'compliance_assessment': PromptTemplate.COMPLIANCE,
            'data_remediation': PromptTemplate.REMEDIATION,
            'remediation_planning': PromptTemplate.REMEDIATION,
            'analytics': PromptTemplate.ANALYTICS,
            'data_analysis': PromptTemplate.ANALYTICS,
            'policy_generation': PromptTemplate.POLICY,
            'policy_development': PromptTemplate.POLICY
        }
        
        return operation_templates.get(operation_type.lower())
    
    def _generate_default_prompt(self, data: Dict[str, Any], operation_type: str) -> str:
        """Generate default prompt for operation."""
        return f"Operation: {operation_type}\n\nData to process:\n{json.dumps(data, indent=2)}\n\nPlease analyze and provide appropriate response."
    
    def _log_operation(self, operation_type: str, details: Dict[str, Any]):
        """Log operation for audit and debugging."""
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'agent_id': self.context.agent_id,
            'operation_type': operation_type,
            'details': details
        }
        
        self.operation_history.append(log_entry)
        
        # Keep only last 100 operations
        if len(self.operation_history) > 100:
            self.operation_history = self.operation_history[-100:]
    
    def get_operation_history(self) -> List[Dict[str, Any]]:
        """Get operation history for this agent."""
        return self.operation_history.copy()
    
    def get_context_summary(self) -> Dict[str, Any]:
        """Get summary of agent context and recent operations."""
        recent_operations = self.operation_history[-10:] if self.operation_history else []
        
        return {
            'agent_context': {
                'agent_id': self.context.agent_id,
                'agent_type': self.context.agent_type,
                'current_task': self.context.current_task,
                'external_services': self.context.external_services
            },
            'recent_operations': recent_operations,
            'total_operations': len(self.operation_history)
        }


class SharedAgentFunctions:
    """Collection of shared functions for all agents."""
    
    @staticmethod
    def create_agent_helper(
        agent_id: str,
        agent_type: str,
        llm_bridge: LLMGatewayBridge,
        integration_orchestrator: IntegrationOrchestrator,
        external_comm: ExternalCommunicationManager,
        **context_kwargs
    ) -> AgentIntegrationHelper:
        """Create agent integration helper with context."""
        context = AgentContext(
            agent_id=agent_id,
            agent_type=agent_type,
            **context_kwargs
        )
        
        return AgentIntegrationHelper(
            context,
            llm_bridge,
            integration_orchestrator,
            external_comm
        )
    
    @staticmethod
    async def standardize_agent_response(
        agent_id: str,
        operation: str,
        success: bool,
        data: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Standardize agent response format."""
        return {
            'agent_id': agent_id,
            'operation': operation,
            'success': success,
            'timestamp': datetime.utcnow().isoformat(),
            'data': data,
            'error': error,
            'metadata': metadata or {}
        }
    
    @staticmethod
    def create_cross_agent_context(
        agents: List[str],
        shared_data: Dict[str, Any],
        operation: str
    ) -> Dict[str, Any]:
        """Create context for cross-agent operations."""
        return {
            'participating_agents': agents,
            'shared_data': shared_data,
            'operation': operation,
            'correlation_id': SharedIntegrationFunctions.create_correlation_id(),
            'created_at': datetime.utcnow().isoformat()
        }
    
    @staticmethod
    async def validate_external_response(
        response: Dict[str, Any],
        expected_schema: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Validate response from external service."""
        validation_result = {
            'valid': True,
            'errors': [],
            'warnings': []
        }
        
        # Basic validation
        if not isinstance(response, dict):
            validation_result['valid'] = False
            validation_result['errors'].append("Response is not a dictionary")
            return validation_result
        
        # Schema validation if provided
        if expected_schema:
            required_fields = expected_schema.get('required', [])
            for field in required_fields:
                if field not in response:
                    validation_result['valid'] = False
                    validation_result['errors'].append(f"Missing required field: {field}")
        
        # Check for common error indicators
        if 'error' in response or 'errors' in response:
            validation_result['warnings'].append("Response contains error fields")
        
        return validation_result
    
    @staticmethod
    async def merge_agent_responses(
        responses: List[Dict[str, Any]],
        merge_strategy: str = "combine"
    ) -> Dict[str, Any]:
        """Merge responses from multiple agents."""
        if not responses:
            return {'success': False, 'error': 'No responses to merge'}
        
        if merge_strategy == "combine":
            merged = {
                'agents': [],
                'combined_data': {},
                'success': all(r.get('success', False) for r in responses),
                'errors': [],
                'metadata': {
                    'merge_strategy': merge_strategy,
                    'response_count': len(responses),
                    'merged_at': datetime.utcnow().isoformat()
                }
            }
            
            for response in responses:
                merged['agents'].append(response.get('agent_id', 'unknown'))
                
                if response.get('success'):
                    # Merge data
                    if 'data' in response:
                        agent_id = response.get('agent_id', 'unknown')
                        merged['combined_data'][agent_id] = response['data']
                else:
                    # Collect errors
                    error_info = {
                        'agent_id': response.get('agent_id', 'unknown'),
                        'error': response.get('error', 'Unknown error')
                    }
                    merged['errors'].append(error_info)
            
            return merged
        
        elif merge_strategy == "best":
            # Return the best/first successful response
            for response in responses:
                if response.get('success'):
                    return {
                        **response,
                        'metadata': {
                            **response.get('metadata', {}),
                            'merge_strategy': merge_strategy,
                            'selected_from': len(responses)
                        }
                    }
            
            # If no successful response, return first with error info
            return {
                'success': False,
                'error': 'No successful responses available',
                'metadata': {
                    'merge_strategy': merge_strategy,
                    'response_count': len(responses)
                }
            }
        
        else:
            raise ValueError(f"Unknown merge strategy: {merge_strategy}")


class AgentOperationPipeline:
    """Pipeline for complex agent operations involving multiple steps."""
    
    def __init__(self, agent_helper: AgentIntegrationHelper):
        self.agent_helper = agent_helper
        self.pipeline_steps: List[Dict[str, Any]] = []
        self.pipeline_results: Dict[str, Any] = {}
    
    def add_llm_step(
        self,
        step_name: str,
        operation_type: str,
        data_source: Union[str, Callable] = "input",
        **llm_kwargs
    ):
        """Add LLM processing step to pipeline."""
        self.pipeline_steps.append({
            'type': 'llm',
            'name': step_name,
            'operation_type': operation_type,
            'data_source': data_source,
            'params': llm_kwargs
        })
    
    def add_external_service_step(
        self,
        step_name: str,
        service_name: str,
        endpoint: str,
        pattern: IntegrationPattern = IntegrationPattern.SYNC_API,
        data_source: Union[str, Callable] = "input",
        **service_kwargs
    ):
        """Add external service call step to pipeline."""
        self.pipeline_steps.append({
            'type': 'external_service',
            'name': step_name,
            'service_name': service_name,
            'endpoint': endpoint,
            'pattern': pattern,
            'data_source': data_source,
            'params': service_kwargs
        })
    
    def add_validation_step(
        self,
        step_name: str,
        validation_function: Callable,
        data_source: Union[str, Callable] = "previous"
    ):
        """Add validation step to pipeline."""
        self.pipeline_steps.append({
            'type': 'validation',
            'name': step_name,
            'validation_function': validation_function,
            'data_source': data_source
        })
    
    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the complete pipeline."""
        try:
            current_data = input_data
            step_results = {}
            
            for step in self.pipeline_steps:
                step_name = step['name']
                logger.info(f"Executing pipeline step: {step_name}")
                
                # Get data for this step
                if callable(step['data_source']):
                    step_input = step['data_source'](step_results, current_data)
                elif step['data_source'] == "input":
                    step_input = input_data
                elif step['data_source'] == "previous":
                    step_input = current_data
                else:
                    step_input = step_results.get(step['data_source'], current_data)
                
                # Execute step
                if step['type'] == 'llm':
                    result = await self.agent_helper.process_with_llm(
                        step_input,
                        step['operation_type'],
                        **step['params']
                    )
                
                elif step['type'] == 'external_service':
                    result = await self.agent_helper.call_external_service(
                        step['service_name'],
                        step['endpoint'],
                        step_input,
                        step['pattern'],
                        **step['params']
                    )
                
                elif step['type'] == 'validation':
                    validation_func = step['validation_function']
                    validation_result = await validation_func(step_input)
                    result = {
                        'success': True,
                        'data': validation_result,
                        'metadata': {'step_type': 'validation'}
                    }
                
                else:
                    raise ValueError(f"Unknown step type: {step['type']}")
                
                # Store step result
                step_results[step_name] = result
                
                # Update current data for next step
                if result.get('success') and result.get('data'):
                    current_data = result['data']
                
                # Stop pipeline if step failed
                if not result.get('success'):
                    logger.error(f"Pipeline step {step_name} failed: {result.get('error')}")
                    break
            
            # Compile final results
            self.pipeline_results = step_results
            
            return {
                'success': all(r.get('success', False) for r in step_results.values()),
                'step_results': step_results,
                'final_data': current_data,
                'pipeline_metadata': {
                    'agent_id': self.agent_helper.context.agent_id,
                    'steps_executed': len(step_results),
                    'total_steps': len(self.pipeline_steps),
                    'execution_time': datetime.utcnow().isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'step_results': step_results if 'step_results' in locals() else {},
                'pipeline_metadata': {
                    'agent_id': self.agent_helper.context.agent_id,
                    'execution_failed_at': datetime.utcnow().isoformat()
                }
            }


# Factory function for easy integration helper creation
def create_agent_integration_helper(
    agent_id: str,
    agent_type: str,
    llm_bridge: LLMGatewayBridge,
    integration_orchestrator: IntegrationOrchestrator,
    external_comm: ExternalCommunicationManager,
    **context_kwargs
) -> AgentIntegrationHelper:
    """Factory function to create agent integration helper."""
    return SharedAgentFunctions.create_agent_helper(
        agent_id=agent_id,
        agent_type=agent_type,
        llm_bridge=llm_bridge,
        integration_orchestrator=integration_orchestrator,
        external_comm=external_comm,
        **context_kwargs
    )


# Export utilities
__all__ = [
    'AgentIntegrationHelper',
    'AgentOperationPipeline',
    'SharedAgentFunctions',
    'AgentContext',
    'create_agent_integration_helper'
]
