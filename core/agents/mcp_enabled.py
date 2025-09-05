"""
MCP-enabled agent base classes for internal communication.
"""

import asyncio
import uuid
from typing import Dict, Any, List, Optional
from abc import ABC, abstractmethod
from datetime import datetime
import logging

from ..communication.mcp import (
    MCPResourceProvider, MCPClient, MCPMessage, MessageType, 
    mcp_bus
)
from ..types.agent_types import AgentCapability
from ..communication.external import external_comm, ExternalServiceType, MessagePriority

logger = logging.getLogger(__name__)


class MCPEnabledAgent(MCPResourceProvider, ABC):
    """Base class for MCP-enabled agents."""
    
    def __init__(self, agent_id: str, name: str, description: str):
        self.agent_id = agent_id
        self.name = name
        self.description = description
        self.capabilities: List[AgentCapability] = []
        self.mcp_client = MCPClient(agent_id)
        self.is_initialized = False
    
    async def initialize(self):
        """Initialize the agent with MCP bus."""
        if not self.is_initialized:
            # Register as provider
            await mcp_bus.register_provider(self.agent_id, self)
            
            # Register as client
            mcp_bus.register_client(self.mcp_client)
            
            # Set up message handlers
            self.mcp_client.register_handler(MessageType.REQUEST, self._handle_mcp_request)
            self.mcp_client.register_handler(MessageType.NOTIFICATION, self._handle_mcp_notification)
            
            self.is_initialized = True
            logger.info(f"Agent {self.agent_id} initialized with capabilities: {self.capabilities}")
    
    async def get_capabilities(self) -> List[AgentCapability]:
        """Get agent capabilities for MCP registration."""
        return self.capabilities
    
    async def handle_request(self, message: MCPMessage) -> MCPMessage:
        """Handle incoming MCP request."""
        try:
            # Route to specific capability handler
            if message.capability in self.capabilities:
                result = await self._process_capability_request(message.capability, message.payload)
                
                # Create response message
                response = MCPMessage(
                    message_id=str(uuid.uuid4()),
                    message_type=MessageType.RESPONSE,
                    sender_id=self.agent_id,
                    receiver_id=message.sender_id,
                    capability=message.capability,
                    payload={"result": result, "status": "success"},
                    timestamp=datetime.utcnow().isoformat(),
                    correlation_id=message.correlation_id
                )
                
                return response
            else:
                # Capability not supported
                error_response = MCPMessage(
                    message_id=str(uuid.uuid4()),
                    message_type=MessageType.ERROR,
                    sender_id=self.agent_id,
                    receiver_id=message.sender_id,
                    capability=message.capability,
                    payload={"error": f"Capability {message.capability} not supported"},
                    timestamp=datetime.utcnow().isoformat(),
                    correlation_id=message.correlation_id
                )
                
                return error_response
                
        except Exception as e:
            logger.error(f"Error handling request in {self.agent_id}: {e}")
            
            error_response = MCPMessage(
                message_id=str(uuid.uuid4()),
                message_type=MessageType.ERROR,
                sender_id=self.agent_id,
                receiver_id=message.sender_id,
                capability=message.capability,
                payload={"error": str(e)},
                timestamp=datetime.utcnow().isoformat(),
                correlation_id=message.correlation_id
            )
            
            return error_response
    
    async def get_resources(self) -> List[Dict[str, Any]]:
        """Get available resources from this agent."""
        return [
            {
                "id": f"{self.agent_id}_main",
                "name": self.name,
                "type": "agent",
                "description": self.description,
                "capabilities": [cap.value for cap in self.capabilities],
                "status": "active" if self.is_initialized else "inactive"
            }
        ]
    
    async def request_from_agent(
        self, 
        target_agent_id: str, 
        capability: AgentCapability, 
        payload: Dict[str, Any],
        timeout: float = 30.0
    ) -> Dict[str, Any]:
        """Request processing from another agent."""
        try:
            response = await self.mcp_client.send_request(
                receiver_id=target_agent_id,
                capability=capability,
                payload=payload,
                timeout=timeout
            )
            
            if response.message_type == MessageType.ERROR:
                raise Exception(response.payload.get("error", "Unknown error"))
            
            return response.payload.get("result", {})
            
        except Exception as e:
            logger.error(f"Request to {target_agent_id} failed: {e}")
            raise
    
    async def notify_agents(self, capability: AgentCapability, payload: Dict[str, Any]):
        """Send notification to all agents with specific capability."""
        await self.mcp_client.send_notification(
            receiver_id=None,  # Broadcast
            capability=capability,
            payload=payload
        )
    
    async def send_to_external_service(
        self, 
        service_type: ExternalServiceType, 
        action: str, 
        payload: Dict[str, Any],
        priority: MessagePriority = MessagePriority.NORMAL
    ) -> str:
        """Send data to external service."""
        return await external_comm.send_to_service(
            service_type=service_type,
            action=action,
            payload=payload,
            priority=priority,
            correlation_id=f"{self.agent_id}_{uuid.uuid4()}"
        )
    
    async def broadcast_event(self, event_type: str, subject: str, data: Dict[str, Any]):
        """Broadcast event externally."""
        return await external_comm.broadcast_event(
            event_type=event_type,
            subject=subject,
            data=data,
            attributes={"agent_id": self.agent_id}
        )
    
    async def _handle_mcp_request(self, message: MCPMessage):
        """Handle incoming MCP request message."""
        logger.info(f"Agent {self.agent_id} received request: {message.payload}")
    
    async def _handle_mcp_notification(self, message: MCPMessage):
        """Handle incoming MCP notification message."""
        logger.info(f"Agent {self.agent_id} received notification: {message.payload}")
        await self._process_notification(message.capability, message.payload)
    
    @abstractmethod
    async def _process_capability_request(self, capability: AgentCapability, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Process a request for a specific capability."""
        pass
    
    async def _process_notification(self, capability: AgentCapability, payload: Dict[str, Any]):
        """Process a notification (optional override)."""
        pass


class DataQualityMCPAgent(MCPEnabledAgent):
    """MCP-enabled Data Quality Agent."""
    
    def __init__(self, agent_id: str = "data_quality_agent", mcp_bus=None, external_comm=None):
        super().__init__(
            agent_id=agent_id,
            name="Data Quality Agent",
            description="Performs data quality assessment and anomaly detection"
        )
        self.mcp_bus = mcp_bus
        self.external_comm = external_comm
        self.capabilities = [
            AgentCapability.DATA_QUALITY_ASSESSMENT,
            AgentCapability.ANOMALY_DETECTION,
            AgentCapability.DATA_PROFILING
        ]
    
    async def _process_capability_request(self, capability: AgentCapability, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Process data quality requests."""
        if capability == AgentCapability.DATA_QUALITY_ASSESSMENT:
            return await self._assess_data_quality(payload)
        elif capability == AgentCapability.ANOMALY_DETECTION:
            return await self._detect_anomalies(payload)
        elif capability == AgentCapability.DATA_PROFILING:
            return await self._profile_data(payload)
        else:
            raise ValueError(f"Unsupported capability: {capability}")
    
    async def _assess_data_quality(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Assess data quality."""
        dataset_id = payload.get("dataset_id")
        data_source_id = payload.get("data_source_id")
        
        # Simulate data quality assessment
        quality_results = {
            "dataset_id": dataset_id,
            "data_source_id": data_source_id,
            "overall_quality_score": 0.85,
            "completeness_score": 0.92,
            "accuracy_score": 0.78,
            "consistency_score": 0.88,
            "validity_score": 0.81,
            "uniqueness_score": 0.95,
            "timeliness_score": 0.87,
            "anomalies_detected": 12,
            "duplicates_found": 8,
            "missing_values": 145,
            "invalid_values": 23,
            "assessment_timestamp": datetime.utcnow().isoformat()
        }
        
        # Send results to external analytics platform
        await self.send_to_external_service(
            service_type=ExternalServiceType.ANALYTICS_PLATFORM,
            action="ingest_quality_metrics",
            payload=quality_results
        )
        
        # Broadcast quality event if score is low
        if quality_results["overall_quality_score"] < 0.7:
            await self.broadcast_event(
                event_type="data_quality",
                subject=f"Low Data Quality Detected: {dataset_id}",
                data=quality_results
            )
        
        return quality_results
    
    async def _detect_anomalies(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Detect anomalies in data."""
        # Simulate anomaly detection
        anomalies = {
            "anomalies": [
                {
                    "type": "outlier",
                    "field": "price",
                    "value": 999999,
                    "confidence": 0.95,
                    "record_id": "rec_12345"
                },
                {
                    "type": "pattern_deviation",
                    "field": "email",
                    "value": "invalid@email",
                    "confidence": 0.87,
                    "record_id": "rec_12346"
                }
            ],
            "total_records_analyzed": 10000,
            "anomaly_rate": 0.02,
            "detection_timestamp": datetime.utcnow().isoformat()
        }
        
        return anomalies
    
    async def _validate_data(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Validate data against rules."""
        validation_rules = payload.get("validation_rules", [])
        
        # Simulate data validation
        validation_results = {
            "rules_applied": len(validation_rules),
            "rules_passed": len(validation_rules) - 2,
            "rules_failed": 2,
            "validation_errors": [
                {
                    "rule": "email_format",
                    "field": "email",
                    "error_count": 15,
                    "sample_values": ["invalid@", "@domain.com", "notanemail"]
                }
            ],
            "validation_timestamp": datetime.utcnow().isoformat()
        }
        
        return validation_results


class ComplianceMCPAgent(MCPEnabledAgent):
    """MCP-enabled Compliance Agent."""
    
    def __init__(self, agent_id: str = "compliance_agent", mcp_bus=None, external_comm=None):
        super().__init__(
            agent_id=agent_id,
            name="Data Privacy Compliance Agent", 
            description="Performs privacy compliance checks and risk assessment"
        )
        self.mcp_bus = mcp_bus
        self.external_comm = external_comm
        self.capabilities = [
            AgentCapability.COMPLIANCE_CHECKING,
            AgentCapability.RISK_ASSESSMENT
        ]
    
    async def _process_capability_request(self, capability: AgentCapability, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Process compliance requests."""
        if capability == AgentCapability.COMPLIANCE_CHECKING:
            return await self._check_compliance(payload)
        elif capability == AgentCapability.RISK_ASSESSMENT:
            return await self._assess_risk(payload)
        else:
            raise ValueError(f"Unsupported capability: {capability}")
    
    async def _check_compliance(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Check compliance with regulations."""
        data_sources = payload.get("data_sources", [])
        regulations = payload.get("regulations", ["GDPR", "CCPA"])
        
        # Simulate compliance checking
        compliance_results = {
            "data_sources": data_sources,
            "regulations_checked": regulations,
            "compliance_score": 0.78,
            "overall_risk_level": "medium",
            "gdpr_compliant": True,
            "ccpa_compliant": False,
            "hipaa_compliant": True,
            "pii_types_detected": ["email", "phone", "ssn"],
            "privacy_risks": [
                {
                    "type": "unencrypted_pii",
                    "severity": "high",
                    "description": "SSN stored in plain text",
                    "affected_records": 1500
                }
            ],
            "violations": [
                {
                    "regulation": "CCPA",
                    "type": "data_retention",
                    "description": "Data retained beyond legal limit"
                }
            ],
            "recommendations": [
                "Encrypt sensitive PII data",
                "Implement data retention policy",
                "Add consent management system"
            ],
            "check_timestamp": datetime.utcnow().isoformat()
        }
        
        # Send high-risk compliance alerts
        if compliance_results["overall_risk_level"] in ["high", "critical"]:
            await self.send_to_external_service(
                service_type=ExternalServiceType.NOTIFICATION_SERVICE,
                action="send_compliance_alert",
                payload=compliance_results,
                priority=MessagePriority.HIGH
            )
        
        # Log audit event
        await self.send_to_external_service(
            service_type=ExternalServiceType.AUDIT_SERVICE,
            action="log_compliance_check",
            payload=compliance_results
        )
        
        return compliance_results
    
    async def _assess_risk(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Assess privacy risk."""
        # Simulate risk assessment
        risk_assessment = {
            "risk_score": 0.65,
            "risk_level": "medium",
            "risk_factors": [
                {
                    "factor": "data_sensitivity",
                    "score": 0.8,
                    "weight": 0.3
                },
                {
                    "factor": "access_control",
                    "score": 0.6,
                    "weight": 0.4
                }
            ],
            "mitigation_suggestions": [
                "Implement role-based access control",
                "Add data encryption at rest",
                "Regular compliance audits"
            ],
            "assessment_timestamp": datetime.utcnow().isoformat()
        }
        
        return risk_assessment


class RemediationMCPAgent(MCPEnabledAgent):
    """MCP-enabled Remediation Agent."""
    
    def __init__(self, agent_id: str = "remediation_agent", mcp_bus=None, external_comm=None):
        super().__init__(
            agent_id=agent_id,
            name="Data Remediation Agent",
            description="Generates and executes data remediation plans"
        )
        self.mcp_bus = mcp_bus
        self.external_comm = external_comm
        self.capabilities = [
            AgentCapability.DATA_REMEDIATION
        ]
    
    async def _process_capability_request(self, capability: AgentCapability, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Process remediation requests."""
        if capability == AgentCapability.DATA_REMEDIATION:
            return await self._generate_remediation_plan(payload)
        else:
            raise ValueError(f"Unsupported capability: {capability}")
    
    async def _generate_remediation_plan(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Generate remediation plan."""
        issues = payload.get("issues", [])
        priority = payload.get("priority", "medium")
        
        # Simulate remediation plan generation
        remediation_plan = {
            "plan_id": str(uuid.uuid4()),
            "issues_addressed": len(issues),
            "tasks": [
                {
                    "task_id": str(uuid.uuid4()),
                    "title": "Fix data quality issues",
                    "description": "Clean duplicate records and validate email formats",
                    "priority": "high",
                    "estimated_effort_hours": 8,
                    "affected_records": 1500,
                    "remediation_rules": ["remove_duplicates", "validate_email_format"]
                },
                {
                    "task_id": str(uuid.uuid4()),
                    "title": "Implement data encryption",
                    "description": "Encrypt sensitive PII fields",
                    "priority": "critical",
                    "estimated_effort_hours": 16,
                    "affected_records": 5000,
                    "remediation_rules": ["encrypt_pii"]
                }
            ],
            "total_estimated_hours": 24,
            "expected_quality_improvement": 0.25,
            "plan_timestamp": datetime.utcnow().isoformat()
        }
        
        # Send remediation plan to reporting service
        await self.send_to_external_service(
            service_type=ExternalServiceType.REPORTING_SERVICE,
            action="generate_remediation_report",
            payload=remediation_plan
        )
        
        return remediation_plan


# Note: Agents should be instantiated when needed, not as globals
# This avoids import-time dependencies and circular import issues
