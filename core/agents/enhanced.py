"""
Enhanced Agent Base Class with Shared Integration Capabilities
Extends the existing agent base with shared functions for external service integration.
"""

from typing import Dict, List, Any, Optional, Union
import logging
from abc import ABC, abstractmethod
from datetime import datetime

from .base import BaseAgent
from ..integrations.shared import (
    AgentIntegrationHelper,
    AgentOperationPipeline,
    SharedAgentFunctions,
    create_agent_integration_helper
)
from ..services.llm_bridge import (
    LLMGatewayBridge,
    PromptTemplate,
    ResponseFormat
)
from ..integrations.patterns import (
    integration_orchestrator,
    IntegrationPattern,
    IntegrationOrchestrator,
    SharedIntegrationFunctions
)
from ..communication.external import ExternalCommunicationManager

logger = logging.getLogger(__name__)


class EnhancedAgentBase(BaseAgent):
    """Enhanced agent base class with shared integration capabilities."""
    
    def __init__(
        self,
        name: str,
        description: str,
        config: Dict[str, Any],
        llm_bridge: LLMGatewayBridge,
        integration_orchestrator: IntegrationOrchestrator,
        external_comm: ExternalCommunicationManager
    ):
        super().__init__(name, description, config)
        
        # Create integration helper
        self.integration_helper = create_agent_integration_helper(
            agent_id=self.name,
            agent_type=self.__class__.__name__,
            llm_bridge=llm_bridge,
            integration_orchestrator=integration_orchestrator,
            external_comm=external_comm,
            current_task=None,
            external_services=config.get('external_services', []),
            llm_preferences=config.get('llm_preferences', {})
        )
        
        self.external_service_configs = config.get('external_service_configs', {})
        self.default_templates = self._setup_default_templates()
    
    @abstractmethod
    def _setup_default_templates(self) -> Dict[str, PromptTemplate]:
        """Setup default LLM templates for this agent type."""
        pass
    
    async def process_with_external_llm(
        self,
        data: Dict[str, Any],
        operation_type: str,
        custom_prompt: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Process data using external LLM with agent-specific optimizations."""
        return await self.integration_helper.process_with_llm(
            data=data,
            operation_type=operation_type,
            custom_prompt=custom_prompt,
            **kwargs
        )
    
    async def call_external_microservice(
        self,
        service_name: str,
        endpoint: str,
        payload: Dict[str, Any],
        pattern: IntegrationPattern = IntegrationPattern.SYNC_API,
        **kwargs
    ) -> Dict[str, Any]:
        """Call external microservice with standardized patterns."""
        return await self.integration_helper.call_external_service(
            service_name=service_name,
            endpoint=endpoint,
            payload=payload,
            pattern=pattern,
            **kwargs
        )
    
    async def send_message_to_external_service(
        self,
        service_name: str,
        message_data: Dict[str, Any],
        priority: str = "normal"
    ) -> Dict[str, Any]:
        """Send message to external service via message queue."""
        return await self.integration_helper.send_mq_message(
            target_service=service_name,
            message_data=message_data,
            priority=priority
        )
    
    async def broadcast_agent_event(
        self,
        event_type: str,
        event_data: Dict[str, Any],
        priority: str = "normal"
    ) -> Dict[str, Any]:
        """Broadcast event from agent to external services."""
        topic = f"agent.{self.name}.{event_type}"
        
        return await self.integration_helper.broadcast_event(
            event_topic=topic,
            event_data={
                'agent_id': self.name,
                'event_type': event_type,
                'data': event_data,
                'timestamp': datetime.utcnow().isoformat()
            },
            priority=priority
        )
    
    async def create_operation_pipeline(self) -> AgentOperationPipeline:
        """Create operation pipeline for complex multi-step operations."""
        return AgentOperationPipeline(self.integration_helper)
    
    async def collaborate_with_agent(
        self,
        target_agent_id: str,
        collaboration_prompt: str,
        shared_context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Collaborate with another agent via shared context."""
        if shared_context is None:
            shared_context = {}
        
        # Add agent-specific context
        shared_context.update({
            'initiating_agent_type': self.__class__.__name__,
            'collaboration_timestamp': datetime.utcnow().isoformat()
        })
        
        try:
            from ..services.llm_bridge import cross_agent_bridge
            
            results = await cross_agent_bridge.facilitate_agent_collaboration(
                initiating_agent=self.name,
                target_agent=target_agent_id,
                collaboration_prompt=collaboration_prompt,
                context_data=shared_context
            )
            
            return {
                'success': True,
                'collaboration_results': results,
                'agents_involved': [self.name, target_agent_id]
            }
            
        except Exception as e:
            logger.error(f"Agent collaboration failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'agents_involved': [self.name, target_agent_id]
            }
    
    def get_integration_status(self) -> Dict[str, Any]:
        """Get status of agent's external integrations."""
        context_summary = self.integration_helper.get_context_summary()
        
        return {
            **context_summary,
            'external_service_configs': list(self.external_service_configs.keys()),
            'integration_capabilities': [
                'llm_processing',
                'external_api_calls',
                'message_queue_communication',
                'event_broadcasting',
                'agent_collaboration'
            ]
        }
    
    async def handle_external_callback(
        self,
        correlation_id: str,
        callback_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle callback from external service."""
        try:
            # Process callback data
            processed_data = await self._process_external_callback(callback_data)
            
            # Log the callback
            self.integration_helper._log_operation('external_callback', {
                'correlation_id': correlation_id,
                'success': True,
                'callback_data_size': len(str(callback_data))
            })
            
            return {
                'success': True,
                'correlation_id': correlation_id,
                'processed_data': processed_data,
                'agent_id': self.name
            }
            
        except Exception as e:
            logger.error(f"External callback handling failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'correlation_id': correlation_id,
                'agent_id': self.name
            }
    
    async def _process_external_callback(self, callback_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process external callback data - override in specific agents."""
        # Default processing - can be overridden by specific agent implementations
        return {
            'processed_at': datetime.utcnow().isoformat(),
            'original_data': callback_data,
            'processor': self.name
        }


class DataQualityEnhancedAgent(EnhancedAgentBase):
    """Enhanced Data Quality Agent with external integrations."""
    
    def _setup_default_templates(self) -> Dict[str, PromptTemplate]:
        return {
            'quality_assessment': PromptTemplate.DATA_QUALITY,
            'anomaly_detection': PromptTemplate.DATA_QUALITY,
            'quality_reporting': PromptTemplate.ANALYTICS
        }
    
    async def assess_data_quality_with_external_tools(
        self,
        dataset: Dict[str, Any],
        external_validators: List[str] = None
    ) -> Dict[str, Any]:
        """Assess data quality using both LLM and external validation services."""
        pipeline = await self.create_operation_pipeline()
        
        # Add LLM assessment step
        pipeline.add_llm_step(
            "llm_quality_assessment",
            "data_quality",
            temperature=0.3  # Lower temperature for consistent analysis
        )
        
        # Add external validation steps if specified
        if external_validators:
            for validator_service in external_validators:
                if validator_service in self.external_service_configs:
                    config = self.external_service_configs[validator_service]
                    pipeline.add_external_service_step(
                        f"external_validation_{validator_service}",
                        validator_service,
                        config['validation_endpoint'],
                        IntegrationPattern.SYNC_API
                    )
        
        # Execute pipeline
        result = await pipeline.execute(dataset)
        
        # Broadcast quality assessment event
        if result['success']:
            await self.broadcast_agent_event(
                "quality_assessment_completed",
                {
                    'dataset_id': dataset.get('id', 'unknown'),
                    'quality_score': result.get('final_data', {}).get('quality_score'),
                    'issues_found': len(result.get('final_data', {}).get('issues', []))
                }
            )
        
        return result


class ComplianceEnhancedAgent(EnhancedAgentBase):
    """Enhanced Compliance Agent with regulatory service integrations."""
    
    def _setup_default_templates(self) -> Dict[str, PromptTemplate]:
        return {
            'compliance_assessment': PromptTemplate.COMPLIANCE,
            'risk_evaluation': PromptTemplate.COMPLIANCE,
            'regulatory_mapping': PromptTemplate.POLICY
        }
    
    async def perform_compliance_check_with_regulatory_apis(
        self,
        data: Dict[str, Any],
        regulations: List[str],
        external_regulatory_services: List[str] = None
    ) -> Dict[str, Any]:
        """Perform compliance check using LLM and external regulatory APIs."""
        pipeline = await self.create_operation_pipeline()
        
        # Add LLM compliance assessment
        pipeline.add_llm_step(
            "llm_compliance_check",
            "compliance_assessment",
            data_source="input"
        )
        
        # Add external regulatory service checks
        if external_regulatory_services:
            for reg_service in external_regulatory_services:
                if reg_service in self.external_service_configs:
                    config = self.external_service_configs[reg_service]
                    pipeline.add_external_service_step(
                        f"regulatory_check_{reg_service}",
                        reg_service,
                        config['compliance_endpoint'],
                        IntegrationPattern.ASYNC_API
                    )
        
        # Execute compliance pipeline
        result = await pipeline.execute({
            'data': data,
            'regulations': regulations,
            'check_timestamp': datetime.utcnow().isoformat()
        })
        
        return result


class RemediationEnhancedAgent(EnhancedAgentBase):
    """Enhanced Remediation Agent with automation service integrations."""
    
    def _setup_default_templates(self) -> Dict[str, PromptTemplate]:
        return {
            'remediation_planning': PromptTemplate.REMEDIATION,
            'action_generation': PromptTemplate.REMEDIATION,
            'impact_assessment': PromptTemplate.ANALYTICS
        }
    
    async def execute_remediation_with_automation(
        self,
        issues: List[Dict[str, Any]],
        automation_services: List[str] = None
    ) -> Dict[str, Any]:
        """Execute remediation using LLM planning and automation services."""
        pipeline = await self.create_operation_pipeline()
        
        # Add LLM remediation planning
        pipeline.add_llm_step(
            "remediation_planning",
            "remediation_planning",
            temperature=0.2  # Low temperature for consistent planning
        )
        
        # Add automation service execution steps
        if automation_services:
            for automation_service in automation_services:
                if automation_service in self.external_service_configs:
                    config = self.external_service_configs[automation_service]
                    pipeline.add_external_service_step(
                        f"automation_{automation_service}",
                        automation_service,
                        config['remediation_endpoint'],
                        IntegrationPattern.ASYNC_API
                    )
        
        # Execute remediation pipeline
        result = await pipeline.execute({
            'issues': issues,
            'agent_id': self.name,
            'remediation_timestamp': datetime.utcnow().isoformat()
        })
        
        return result


class AnalyticsEnhancedAgent(EnhancedAgentBase):
    """Enhanced Analytics Agent with BI service integrations."""
    
    def _setup_default_templates(self) -> Dict[str, PromptTemplate]:
        return {
            'data_analysis': PromptTemplate.ANALYTICS,
            'insight_generation': PromptTemplate.ANALYTICS,
            'trend_analysis': PromptTemplate.ANALYTICS
        }
    
    async def generate_analytics_with_bi_tools(
        self,
        dataset: Dict[str, Any],
        analysis_requirements: Dict[str, Any],
        bi_services: List[str] = None
    ) -> Dict[str, Any]:
        """Generate analytics using LLM insights and BI tool integration."""
        pipeline = await self.create_operation_pipeline()
        
        # Add LLM analysis step
        pipeline.add_llm_step(
            "llm_data_analysis",
            "data_analysis",
            temperature=0.4
        )
        
        # Add BI service integrations
        if bi_services:
            for bi_service in bi_services:
                if bi_service in self.external_service_configs:
                    config = self.external_service_configs[bi_service]
                    pipeline.add_external_service_step(
                        f"bi_analysis_{bi_service}",
                        bi_service,
                        config['analytics_endpoint'],
                        IntegrationPattern.SYNC_API
                    )
        
        # Execute analytics pipeline
        result = await pipeline.execute({
            'dataset': dataset,
            'requirements': analysis_requirements,
            'analysis_timestamp': datetime.utcnow().isoformat()
        })
        
        return result


class PolicyEnhancedAgent(EnhancedAgentBase):
    """Enhanced Policy Agent with governance service integrations."""
    
    def _setup_default_templates(self) -> Dict[str, PromptTemplate]:
        return {
            'policy_generation': PromptTemplate.POLICY,
            'policy_validation': PromptTemplate.COMPLIANCE,
            'governance_assessment': PromptTemplate.ANALYTICS
        }
    
    async def develop_policy_with_governance_tools(
        self,
        requirements: Dict[str, Any],
        governance_services: List[str] = None
    ) -> Dict[str, Any]:
        """Develop policy using LLM and external governance tools."""
        pipeline = await self.create_operation_pipeline()
        
        # Add LLM policy generation
        pipeline.add_llm_step(
            "llm_policy_generation",
            "policy_generation",
            temperature=0.3
        )
        
        # Add governance service validations
        if governance_services:
            for gov_service in governance_services:
                if gov_service in self.external_service_configs:
                    config = self.external_service_configs[gov_service]
                    pipeline.add_external_service_step(
                        f"governance_validation_{gov_service}",
                        gov_service,
                        config['policy_validation_endpoint'],
                        IntegrationPattern.SYNC_API
                    )
        
        # Execute policy development pipeline
        result = await pipeline.execute({
            'requirements': requirements,
            'agent_id': self.name,
            'development_timestamp': datetime.utcnow().isoformat()
        })
        
        return result


class AgentFactory:
    """Factory for creating enhanced agents with shared integration capabilities."""
    
    @staticmethod
    def create_enhanced_agent(
        agent_type: str,
        agent_id: str,
        config: Dict[str, Any],
        llm_bridge: LLMGatewayBridge,
        integration_orchestrator: IntegrationOrchestrator,
        external_comm: ExternalCommunicationManager
    ) -> EnhancedAgentBase:
        """Create enhanced agent of specified type."""
        
        agent_classes = {
            'data_quality': DataQualityEnhancedAgent,
            'compliance': ComplianceEnhancedAgent,
            'remediation': RemediationEnhancedAgent,
            'analytics': AnalyticsEnhancedAgent,
            'policy': PolicyEnhancedAgent
        }
        
        agent_class = agent_classes.get(agent_type.lower())
        if not agent_class:
            raise ValueError(f"Unknown agent type: {agent_type}")
        
        description = f"Enhanced {agent_type} agent with external service integration"
        
        return agent_class(
            name=agent_id,
            description=description,
            config=config,
            llm_bridge=llm_bridge,
            integration_orchestrator=integration_orchestrator,
            external_comm=external_comm
        )
    
    @staticmethod
    async def initialize_agent_with_external_services(
        agent: EnhancedAgentBase,
        external_service_mappings: Dict[str, Dict[str, Any]]
    ):
        """Initialize agent with external service mappings."""
        try:
            # Register external services for this agent
            for service_name, service_config in external_service_mappings.items():
                logger.info(f"Registering {service_name} for agent {agent.name}")
                
                # Store service config in agent
                agent.external_service_configs[service_name] = service_config
                
                # Update agent context
                if service_name not in agent.integration_helper.context.external_services:
                    agent.integration_helper.context.external_services.append(service_name)
            
            logger.info(f"Agent {agent.name} initialized with {len(external_service_mappings)} external services")
            
        except Exception as e:
            logger.error(f"Failed to initialize agent with external services: {e}")
            raise


class CrossAgentOperationManager:
    """Manages operations that span multiple agents."""
    
    def __init__(self, agents: List[EnhancedAgentBase]):
        self.agents = {agent.name: agent for agent in agents}
        self.active_operations: Dict[str, Dict[str, Any]] = {}
    
    async def execute_cross_agent_workflow(
        self,
        workflow_name: str,
        agents_sequence: List[str],
        initial_data: Dict[str, Any],
        operation_config: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Execute workflow across multiple agents."""
        operation_id = SharedIntegrationFunctions.create_correlation_id()
        
        try:
            self.active_operations[operation_id] = {
                'workflow_name': workflow_name,
                'agents_sequence': agents_sequence,
                'started_at': datetime.utcnow(),
                'status': 'in_progress'
            }
            
            current_data = initial_data
            results = {}
            
            for i, agent_id in enumerate(agents_sequence):
                if agent_id not in self.agents:
                    raise ValueError(f"Agent {agent_id} not found")
                
                agent = self.agents[agent_id]
                
                # Determine operation for this agent
                operation_type = operation_config.get(agent_id, {}).get('operation', 'default_processing')
                
                # Process with agent
                agent_result = await agent.process_with_external_llm(
                    current_data,
                    operation_type
                )
                
                results[agent_id] = agent_result
                
                # Update data for next agent
                if agent_result.get('success') and agent_result.get('data'):
                    current_data = agent_result['data']
                else:
                    # If agent fails, stop workflow
                    break
            
            # Mark operation as completed
            self.active_operations[operation_id]['status'] = 'completed'
            self.active_operations[operation_id]['completed_at'] = datetime.utcnow()
            
            return {
                'success': True,
                'operation_id': operation_id,
                'workflow_name': workflow_name,
                'agent_results': results,
                'final_data': current_data
            }
            
        except Exception as e:
            self.active_operations[operation_id]['status'] = 'failed'
            self.active_operations[operation_id]['error'] = str(e)
            
            logger.error(f"Cross-agent workflow failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'operation_id': operation_id,
                'workflow_name': workflow_name
            }
    
    def get_operation_status(self, operation_id: str) -> Optional[Dict[str, Any]]:
        """Get status of cross-agent operation."""
        return self.active_operations.get(operation_id)
    
    def list_active_operations(self) -> List[Dict[str, Any]]:
        """List all active cross-agent operations."""
        return [
            {**op, 'operation_id': op_id}
            for op_id, op in self.active_operations.items()
            if op.get('status') == 'in_progress'
        ]


# Global enhanced agents registry
enhanced_agents_registry: Dict[str, EnhancedAgentBase] = {}
cross_agent_manager: Optional[CrossAgentOperationManager] = None


def initialize_enhanced_agents(
    llm_bridge: LLMGatewayBridge,
    integration_orchestrator: IntegrationOrchestrator,
    external_comm: ExternalCommunicationManager,
    agent_configs: Dict[str, Dict[str, Any]]
) -> Dict[str, EnhancedAgentBase]:
    """Initialize all enhanced agents with shared capabilities."""
    global enhanced_agents_registry, cross_agent_manager
    
    enhanced_agents = {}
    
    for agent_id, config in agent_configs.items():
        agent_type = config.get('type', agent_id.split('_')[0])
        
        try:
            agent = AgentFactory.create_enhanced_agent(
                agent_type=agent_type,
                agent_id=agent_id,
                config=config,
                llm_bridge=llm_bridge,
                integration_orchestrator=integration_orchestrator,
                external_comm=external_comm
            )
            
            enhanced_agents[agent_id] = agent
            logger.info(f"Created enhanced agent: {agent_id}")
            
        except Exception as e:
            logger.error(f"Failed to create enhanced agent {agent_id}: {e}")
    
    enhanced_agents_registry = enhanced_agents
    cross_agent_manager = CrossAgentOperationManager(list(enhanced_agents.values()))
    
    logger.info(f"Initialized {len(enhanced_agents)} enhanced agents")
    return enhanced_agents


# Export classes and functions
__all__ = [
    'EnhancedAgentBase',
    'DataQualityEnhancedAgent',
    'ComplianceEnhancedAgent', 
    'RemediationEnhancedAgent',
    'AnalyticsEnhancedAgent',
    'PolicyEnhancedAgent',
    'AgentFactory',
    'CrossAgentOperationManager',
    'initialize_enhanced_agents',
    'enhanced_agents_registry',
    'cross_agent_manager'
]
