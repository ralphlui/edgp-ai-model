"""
LLM Gateway Bridge for Agent Integration
Provides standardized interface between agents and LLM gateway with advanced patterns.
"""

import asyncio
import json
import time
from typing import Dict, List, Any, Optional, Union, AsyncGenerator, Callable
from dataclasses import dataclass
from enum import Enum
import logging

from .llm_gateway import llm_gateway, LLMResponse, SimpleLLMProvider
from ..integrations.patterns import (
    IntegrationRequest, 
    IntegrationResponse,
    IntegrationPattern,
    SharedIntegrationFunctions
)

logger = logging.getLogger(__name__)


class ResponseFormat(Enum):
    """Supported response formats."""
    JSON = "json"
    TEXT = "text"
    MARKDOWN = "markdown"
    STRUCTURED = "structured"


class PromptTemplate(Enum):
    """Standard prompt templates for different agent types."""
    DATA_QUALITY = "data_quality"
    COMPLIANCE = "compliance"
    REMEDIATION = "remediation"
    ANALYTICS = "analytics"
    POLICY = "policy"


@dataclass
class LLMRequest:
    """Standardized LLM request from agents."""
    agent_id: str
    request_id: str
    prompt: str
    system_prompt: Optional[str] = None
    template: Optional[PromptTemplate] = None
    response_format: ResponseFormat = ResponseFormat.JSON
    temperature: float = 0.7
    max_tokens: int = 1000
    streaming: bool = False
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class LLMBridgeResponse:
    """Enhanced response from LLM bridge."""
    request_id: str
    agent_id: str
    content: str
    format: ResponseFormat
    metadata: Dict[str, Any]
    processing_time_ms: int
    success: bool
    error: Optional[str] = None


class PromptTemplateManager:
    """Manages prompt templates for different agent types."""
    
    def __init__(self):
        self.templates = {
            PromptTemplate.DATA_QUALITY: {
                "system": """You are a data quality assessment expert. Analyze the provided data and identify quality issues, anomalies, and recommendations for improvement. 
                
Respond in the following JSON format:
{
    "quality_score": float,
    "issues": [{"type": str, "severity": str, "description": str, "affected_fields": [str]}],
    "recommendations": [{"action": str, "priority": str, "impact": str}],
    "summary": str
}""",
                "user_prefix": "Assess the data quality for the following dataset:\n\n"
            },
            
            PromptTemplate.COMPLIANCE: {
                "system": """You are a compliance assessment expert. Evaluate the provided data against regulatory requirements and identify compliance gaps, risks, and remediation actions.

Respond in the following JSON format:
{
    "compliance_score": float,
    "regulations_checked": [str],
    "violations": [{"regulation": str, "severity": str, "description": str, "affected_data": str}],
    "risk_assessment": {"level": str, "factors": [str]},
    "remediation_actions": [{"action": str, "urgency": str, "regulation": str}]
}""",
                "user_prefix": "Perform compliance assessment for the following data:\n\n"
            },
            
            PromptTemplate.REMEDIATION: {
                "system": """You are a data remediation expert. Analyze data quality issues and compliance violations to provide specific remediation strategies and implementation guidance.

Respond in the following JSON format:
{
    "remediation_plan": {"strategy": str, "timeline": str, "resources_required": [str]},
    "specific_actions": [{"step": int, "action": str, "tool": str, "estimated_time": str}],
    "validation_criteria": [{"check": str, "expected_outcome": str}],
    "risk_mitigation": [{"risk": str, "mitigation": str}]
}""",
                "user_prefix": "Develop remediation strategy for the following issues:\n\n"
            },
            
            PromptTemplate.ANALYTICS: {
                "system": """You are a data analytics expert. Generate insights, trends, and actionable intelligence from the provided data and analysis requirements.

Respond in the following JSON format:
{
    "insights": [{"category": str, "insight": str, "confidence": float, "supporting_data": str}],
    "trends": [{"metric": str, "direction": str, "significance": str, "time_period": str}],
    "recommendations": [{"recommendation": str, "business_impact": str, "implementation": str}],
    "visualizations": [{"type": str, "data_points": [str], "title": str}]
}""",
                "user_prefix": "Analyze the following data and provide insights:\n\n"
            },
            
            PromptTemplate.POLICY: {
                "system": """You are a policy development expert. Create comprehensive data governance policies based on requirements, compliance needs, and organizational context.

Respond in the following JSON format:
{
    "policy_framework": {"title": str, "scope": str, "objectives": [str]},
    "policy_rules": [{"rule_id": str, "category": str, "rule": str, "enforcement": str}],
    "implementation_guide": [{"phase": str, "actions": [str], "timeline": str}],
    "compliance_mapping": [{"regulation": str, "policy_rules": [str]}]
}""",
                "user_prefix": "Develop data governance policy for the following requirements:\n\n"
            }
        }
    
    def get_template(self, template_type: PromptTemplate) -> Dict[str, str]:
        """Get prompt template by type."""
        return self.templates.get(template_type, {})
    
    def format_prompt(self, template_type: PromptTemplate, user_content: str, additional_context: Dict[str, Any] = None) -> tuple[str, str]:
        """Format prompt using template."""
        template = self.get_template(template_type)
        
        if not template:
            return None, user_content
        
        system_prompt = template.get("system", "")
        user_prefix = template.get("user_prefix", "")
        
        # Add additional context if provided
        if additional_context:
            context_str = "\n\nAdditional Context:\n" + json.dumps(additional_context, indent=2)
            user_content += context_str
        
        formatted_user_prompt = user_prefix + user_content
        
        return system_prompt, formatted_user_prompt


class LLMGatewayBridge:
    """Bridge between agents and LLM gateway with advanced integration patterns."""
    
    def __init__(self, llm_provider: Optional[SimpleLLMProvider] = None):
        self.llm_provider = llm_provider or SimpleLLMProvider("bridge_mock")
        self.template_manager = PromptTemplateManager()
        self.request_cache = {}
        self.response_processors: Dict[str, Callable] = {}
        self.metrics = {
            'requests_processed': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'cache_hits': 0,
            'average_response_time': 0.0
        }
    
    def register_response_processor(self, agent_type: str, processor: Callable):
        """Register a response processor for specific agent type."""
        self.response_processors[agent_type] = processor
        logger.info(f"Registered response processor for {agent_type}")
    
    async def process_agent_request(self, request: LLMRequest) -> LLMBridgeResponse:
        """Process LLM request from agent with enhanced features."""
        start_time = time.time()
        
        try:
            # Check cache first
            cache_key = self._generate_cache_key(request)
            if cache_key in self.request_cache:
                cached_response = self.request_cache[cache_key]
                self.metrics['cache_hits'] += 1
                logger.info(f"Cache hit for request {request.request_id}")
                
                return LLMBridgeResponse(
                    request_id=request.request_id,
                    agent_id=request.agent_id,
                    content=cached_response['content'],
                    format=request.response_format,
                    metadata={**cached_response['metadata'], 'cached': True},
                    processing_time_ms=int((time.time() - start_time) * 1000),
                    success=True
                )
            
            # Format prompt using template if specified
            if request.template:
                system_prompt, formatted_prompt = self.template_manager.format_prompt(
                    request.template,
                    request.prompt,
                    request.metadata
                )
                if system_prompt:
                    request.system_prompt = system_prompt
                    request.prompt = formatted_prompt
            
            # Generate LLM response
            if request.streaming:
                # Handle streaming separately
                return await self._handle_streaming_request(request, start_time)
            else:
                llm_response = await self.llm_provider.generate(
                    prompt=request.prompt,
                    system_prompt=request.system_prompt,
                    temperature=request.temperature,
                    max_tokens=request.max_tokens
                )
            
            # Process response based on format
            processed_content = await self._process_response_format(
                llm_response.content,
                request.response_format,
                request.agent_id
            )
            
            # Cache successful response
            self._cache_response(cache_key, processed_content, llm_response.metadata)
            
            # Update metrics
            self.metrics['requests_processed'] += 1
            self.metrics['successful_requests'] += 1
            processing_time = int((time.time() - start_time) * 1000)
            self._update_average_response_time(processing_time)
            
            return LLMBridgeResponse(
                request_id=request.request_id,
                agent_id=request.agent_id,
                content=processed_content,
                format=request.response_format,
                metadata={
                    **llm_response.metadata,
                    'template_used': request.template.value if request.template else None,
                    'cache_key': cache_key
                },
                processing_time_ms=processing_time,
                success=True
            )
            
        except Exception as e:
            self.metrics['requests_processed'] += 1
            self.metrics['failed_requests'] += 1
            
            logger.error(f"LLM bridge request failed for {request.request_id}: {e}")
            
            return LLMBridgeResponse(
                request_id=request.request_id,
                agent_id=request.agent_id,
                content="",
                format=request.response_format,
                metadata={'error_details': str(e)},
                processing_time_ms=int((time.time() - start_time) * 1000),
                success=False,
                error=str(e)
            )
    
    async def _handle_streaming_request(self, request: LLMRequest, start_time: float) -> LLMBridgeResponse:
        """Handle streaming LLM request."""
        try:
            content_chunks = []
            
            async for chunk in self.llm_provider.stream_generate(
                prompt=request.prompt,
                system_prompt=request.system_prompt,
                temperature=request.temperature,
                max_tokens=request.max_tokens
            ):
                content_chunks.append(chunk)
            
            full_content = ''.join(content_chunks)
            processed_content = await self._process_response_format(
                full_content,
                request.response_format,
                request.agent_id
            )
            
            return LLMBridgeResponse(
                request_id=request.request_id,
                agent_id=request.agent_id,
                content=processed_content,
                format=request.response_format,
                metadata={'streaming': True, 'chunks_received': len(content_chunks)},
                processing_time_ms=int((time.time() - start_time) * 1000),
                success=True
            )
            
        except Exception as e:
            raise e
    
    async def _process_response_format(self, content: str, format_type: ResponseFormat, agent_id: str) -> str:
        """Process response according to requested format."""
        # Apply agent-specific response processor if available
        if agent_id in self.response_processors:
            try:
                processor = self.response_processors[agent_id]
                content = await processor(content, format_type)
            except Exception as e:
                logger.warning(f"Response processor failed for {agent_id}: {e}")
        
        # Apply format-specific processing
        if format_type == ResponseFormat.JSON:
            try:
                # Try to parse and reformat JSON
                parsed = json.loads(content)
                return json.dumps(parsed, indent=2)
            except json.JSONDecodeError:
                # If not valid JSON, try to extract JSON from text
                return self._extract_json_from_text(content)
        
        elif format_type == ResponseFormat.STRUCTURED:
            return self._convert_to_structured_format(content)
        
        return content
    
    def _extract_json_from_text(self, text: str) -> str:
        """Extract JSON from text response."""
        # Look for JSON-like content
        start_markers = ['{', '[']
        end_markers = ['}', ']']
        
        for start_marker, end_marker in zip(start_markers, end_markers):
            start_idx = text.find(start_marker)
            if start_idx != -1:
                # Find matching end marker
                bracket_count = 0
                for i, char in enumerate(text[start_idx:], start_idx):
                    if char == start_marker:
                        bracket_count += 1
                    elif char == end_marker:
                        bracket_count -= 1
                        if bracket_count == 0:
                            json_str = text[start_idx:i+1]
                            try:
                                parsed = json.loads(json_str)
                                return json.dumps(parsed, indent=2)
                            except json.JSONDecodeError:
                                continue
        
        # If no valid JSON found, wrap in simple structure
        return json.dumps({"response": text.strip()}, indent=2)
    
    def _convert_to_structured_format(self, content: str) -> str:
        """Convert content to structured format."""
        lines = content.strip().split('\n')
        structured = {
            "summary": lines[0] if lines else "",
            "details": lines[1:] if len(lines) > 1 else [],
            "metadata": {
                "line_count": len(lines),
                "character_count": len(content)
            }
        }
        return json.dumps(structured, indent=2)
    
    def _generate_cache_key(self, request: LLMRequest) -> str:
        """Generate cache key for request."""
        key_data = {
            'prompt_hash': hash(request.prompt),
            'system_prompt_hash': hash(request.system_prompt or ""),
            'template': request.template.value if request.template else None,
            'temperature': request.temperature,
            'response_format': request.response_format.value
        }
        return f"llm_cache_{hash(str(key_data))}"
    
    def _cache_response(self, cache_key: str, content: str, metadata: Dict[str, Any]):
        """Cache successful response."""
        self.request_cache[cache_key] = {
            'content': content,
            'metadata': metadata,
            'cached_at': time.time()
        }
        
        # Simple cache size management (keep last 100 responses)
        if len(self.request_cache) > 100:
            oldest_key = min(self.request_cache.keys(), 
                           key=lambda k: self.request_cache[k]['cached_at'])
            del self.request_cache[oldest_key]
    
    def _update_average_response_time(self, processing_time: int):
        """Update average response time metric."""
        current_avg = self.metrics['average_response_time']
        total_requests = self.metrics['requests_processed']
        
        if total_requests == 1:
            self.metrics['average_response_time'] = processing_time
        else:
            # Weighted average
            self.metrics['average_response_time'] = (
                (current_avg * (total_requests - 1) + processing_time) / total_requests
            )
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get bridge performance metrics."""
        return {
            **self.metrics,
            'cache_size': len(self.request_cache),
            'cache_hit_rate': (
                self.metrics['cache_hits'] / max(self.metrics['requests_processed'], 1)
            ) * 100
        }
    
    def clear_cache(self):
        """Clear response cache."""
        self.request_cache.clear()
        self.metrics['cache_hits'] = 0
        logger.info("LLM bridge cache cleared")


class AgentLLMInterface:
    """Simplified interface for agents to interact with LLM gateway."""
    
    def __init__(self, agent_id: str, bridge: LLMGatewayBridge):
        self.agent_id = agent_id
        self.bridge = bridge
    
    async def generate_response(
        self,
        prompt: str,
        template: Optional[PromptTemplate] = None,
        response_format: ResponseFormat = ResponseFormat.JSON,
        **kwargs
    ) -> LLMBridgeResponse:
        """Generate LLM response for agent."""
        request = LLMRequest(
            agent_id=self.agent_id,
            request_id=SharedIntegrationFunctions.create_correlation_id(),
            prompt=prompt,
            template=template,
            response_format=response_format,
            **kwargs
        )
        
        return await self.bridge.process_agent_request(request)
    
    async def analyze_data(
        self,
        data: Dict[str, Any],
        analysis_type: str,
        context: Optional[Dict[str, Any]] = None
    ) -> LLMBridgeResponse:
        """Perform data analysis using LLM."""
        prompt = f"Analysis Type: {analysis_type}\n\nData to Analyze:\n{json.dumps(data, indent=2)}"
        
        # Add context if provided
        if context:
            prompt += f"\n\nAdditional Context:\n{json.dumps(context, indent=2)}"
        
        # Determine appropriate template based on analysis type
        template = None
        if "quality" in analysis_type.lower():
            template = PromptTemplate.DATA_QUALITY
        elif "compliance" in analysis_type.lower():
            template = PromptTemplate.COMPLIANCE
        elif "remediation" in analysis_type.lower():
            template = PromptTemplate.REMEDIATION
        elif "analytics" in analysis_type.lower():
            template = PromptTemplate.ANALYTICS
        
        return await self.generate_response(
            prompt=prompt,
            template=template,
            response_format=ResponseFormat.JSON,
            metadata={'analysis_type': analysis_type, 'context': context}
        )
    
    async def generate_recommendations(
        self,
        issue_description: str,
        domain: str,
        priority: str = "normal"
    ) -> LLMBridgeResponse:
        """Generate recommendations for specific issues."""
        prompt = f"Domain: {domain}\nPriority: {priority}\n\nIssue Description:\n{issue_description}\n\nProvide actionable recommendations."
        
        template = PromptTemplate.REMEDIATION if domain == "remediation" else PromptTemplate.POLICY
        
        return await self.generate_response(
            prompt=prompt,
            template=template,
            response_format=ResponseFormat.JSON,
            metadata={'domain': domain, 'priority': priority}
        )


class BatchLLMProcessor:
    """Processes multiple LLM requests in batches for efficiency."""
    
    def __init__(self, bridge: LLMGatewayBridge, batch_size: int = 5):
        self.bridge = bridge
        self.batch_size = batch_size
        self.processing_queue: List[LLMRequest] = []
        self.result_handlers: Dict[str, Callable] = {}
    
    def add_request(self, request: LLMRequest, result_handler: Optional[Callable] = None):
        """Add request to batch processing queue."""
        self.processing_queue.append(request)
        if result_handler:
            self.result_handlers[request.request_id] = result_handler
    
    async def process_batch(self) -> List[LLMBridgeResponse]:
        """Process all queued requests in batches."""
        results = []
        
        for i in range(0, len(self.processing_queue), self.batch_size):
            batch = self.processing_queue[i:i + self.batch_size]
            
            # Process batch concurrently
            batch_tasks = [
                self.bridge.process_agent_request(request)
                for request in batch
            ]
            
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            # Handle results and call handlers
            for request, result in zip(batch, batch_results):
                if isinstance(result, Exception):
                    logger.error(f"Batch processing failed for {request.request_id}: {result}")
                    result = LLMBridgeResponse(
                        request_id=request.request_id,
                        agent_id=request.agent_id,
                        content="",
                        format=request.response_format,
                        metadata={},
                        processing_time_ms=0,
                        success=False,
                        error=str(result)
                    )
                
                results.append(result)
                
                # Call result handler if registered
                if request.request_id in self.result_handlers:
                    try:
                        handler = self.result_handlers[request.request_id]
                        await handler(result)
                    except Exception as e:
                        logger.error(f"Result handler failed for {request.request_id}: {e}")
        
        # Clear processed requests
        self.processing_queue.clear()
        self.result_handlers.clear()
        
        return results


class CrossAgentCommunicationBridge:
    """Facilitates communication between agents through LLM gateway."""
    
    def __init__(self, bridge: LLMGatewayBridge):
        self.bridge = bridge
        self.agent_interfaces: Dict[str, AgentLLMInterface] = {}
    
    def register_agent(self, agent_id: str) -> AgentLLMInterface:
        """Register an agent and return its interface."""
        interface = AgentLLMInterface(agent_id, self.bridge)
        self.agent_interfaces[agent_id] = interface
        logger.info(f"Registered agent {agent_id} with LLM bridge")
        return interface
    
    async def facilitate_agent_collaboration(
        self,
        initiating_agent: str,
        target_agent: str,
        collaboration_prompt: str,
        context_data: Dict[str, Any]
    ) -> Dict[str, LLMBridgeResponse]:
        """Facilitate collaboration between two agents."""
        if initiating_agent not in self.agent_interfaces:
            raise ValueError(f"Agent {initiating_agent} not registered")
        
        if target_agent not in self.agent_interfaces:
            raise ValueError(f"Agent {target_agent} not registered")
        
        # Get interfaces
        init_interface = self.agent_interfaces[initiating_agent]
        target_interface = self.agent_interfaces[target_agent]
        
        # Prepare collaboration context
        collaboration_context = {
            'initiating_agent': initiating_agent,
            'target_agent': target_agent,
            'collaboration_prompt': collaboration_prompt,
            'shared_context': context_data
        }
        
        # Generate responses from both agents
        init_task = init_interface.generate_response(
            prompt=f"Collaborate with {target_agent} on: {collaboration_prompt}",
            response_format=ResponseFormat.JSON,
            metadata=collaboration_context
        )
        
        target_task = target_interface.generate_response(
            prompt=f"Respond to collaboration request from {initiating_agent}: {collaboration_prompt}",
            response_format=ResponseFormat.JSON,
            metadata=collaboration_context
        )
        
        # Execute concurrently
        init_response, target_response = await asyncio.gather(init_task, target_task)
        
        return {
            'initiating_agent_response': init_response,
            'target_agent_response': target_response
        }


# Global bridge instance
llm_bridge = LLMGatewayBridge()
cross_agent_bridge = CrossAgentCommunicationBridge(llm_bridge)
batch_processor = BatchLLMProcessor(llm_bridge)
