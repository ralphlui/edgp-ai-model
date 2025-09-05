"""
EDGP AI Model - Message-Driven Architecture
Main FastAPI application with MCP internal communication and AWS SQS/SNS external messaging.
"""
import asyncio
import logging
import os
from contextlib import asynccontextmanager
from typing import Dict, List, Any, Optional
from datetime import datetime

from fastapi import FastAPI, HTTPException, status, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from core.infrastructure.config import get_settings
from core.communication.mcp import (
    MCPMessageBus, 
    LangGraphAgentOrchestrator,
    MCPMessage,
    MessagePriority
)
from core.communication.external import ExternalCommunicationManager
from core.agents.mcp_enabled import (
    DataQualityMCPAgent,
    ComplianceMCPAgent, 
    RemediationMCPAgent
)
from core.infrastructure.monitoring import SystemMonitor
from core.types.agent_types import AgentCapability
from core.types.responses import (
    HealthResponse,
    AgentStatusResponse,
    WorkflowResponse,
    MessageResponse
)
from core.types.data import ProcessingRequest
from core.infrastructure.error_handling import setup_error_handlers
from core.integrations.endpoints import (
    external_router,
    webhook_router,
    llm_router,
    integration_router
)
from core.integrations.patterns import (
    integration_orchestrator,
    service_registry
)
from core.services.llm_bridge import llm_bridge, cross_agent_bridge

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/edgp_ai_model.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Global instances
mcp_bus = MCPMessageBus()
external_comm = ExternalCommunicationManager()
orchestrator = LangGraphAgentOrchestrator(mcp_bus)
# Initialize monitoring system
from core.infrastructure.monitoring import Metrics
metrics = Metrics()
system_monitor = SystemMonitor(metrics)

# MCP-enabled agents
data_quality_agent = DataQualityMCPAgent("data_quality_agent", mcp_bus, external_comm)
compliance_agent = ComplianceMCPAgent("compliance_agent", mcp_bus, external_comm)
remediation_agent = RemediationMCPAgent("remediation_agent", mcp_bus, external_comm)

# Track initialized agents
initialized_agents: List[Any] = []

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan with MCP and external communication setup."""
    logger.info("Starting EDGP AI Model with message-driven architecture...")
    
    # Create logs directory
    os.makedirs("logs", exist_ok=True)
    
    try:
        # Initialize MCP message bus
        logger.info("Starting MCP message bus...")
        bus_task = asyncio.create_task(mcp_bus.start_message_processing())
        
        # Initialize external communication
        logger.info("Initializing external communication (AWS SQS/SNS)...")
        await external_comm.initialize()
        
        # Initialize integration orchestrator and service registry
        logger.info("Initializing integration orchestrator...")
        await integration_orchestrator.start()
        await service_registry.start_all_adapters()
        
        # Register external services for integration
        from core.integrations.config import integration_config_manager
        registry_data = integration_config_manager.generate_service_registry_data()
        for service_data in registry_data:
            service_registry.register_service(
                service_data['service_name'],
                service_data['base_url'],
                service_data['pattern'],
                service_data['config']
            )
        
        # Initialize enhanced agents with shared integration capabilities
        logger.info("Initializing enhanced agents with external service integration...")
        from core.agents.enhanced import initialize_enhanced_agents
        from core.integrations.config import DEFAULT_EXTERNAL_SERVICES_CONFIG
        
        enhanced_agents = initialize_enhanced_agents(
            llm_bridge=llm_bridge,
            integration_orchestrator=integration_orchestrator,
            external_comm=external_comm,
            agent_configs=DEFAULT_EXTERNAL_SERVICES_CONFIG
        )
        
        logger.info(f"Initialized {len(enhanced_agents)} enhanced agents")
        
        # Initialize and register MCP agents
        logger.info("Initializing MCP-enabled agents...")
        agents_to_init = [data_quality_agent, compliance_agent, remediation_agent]
        
        for agent in agents_to_init:
            await agent.initialize()
            initialized_agents.append(agent)
        
        # Initialize orchestrator
        logger.info("Initializing LangGraph orchestrator...")
        await orchestrator.initialize()
        
        # Create sample workflows
        await _create_sample_workflows()
        
        # Start system monitoring
        system_monitor.start_monitoring()
        
        logger.info(f"Successfully initialized {len(initialized_agents)} MCP agents")
        logger.info("EDGP AI Model startup complete - message-driven architecture ready")
        
        yield
        
    except Exception as e:
        logger.error(f"Failed to initialize application: {e}")
        raise
    finally:
        # Shutdown
        logger.info("Shutting down EDGP AI Model...")
        
        # Stop integration orchestrator
        await integration_orchestrator.stop()
        await service_registry.stop_all_adapters()
        
        # Stop MCP message bus
        await mcp_bus.stop_message_processing()
        
        # Stop monitoring
        system_monitor.stop_monitoring()
        
        logger.info("Shutdown complete")


async def _create_sample_workflows():
    """Create sample LangGraph workflows for agent orchestration."""
    
    # Data Quality Assessment Workflow
    quality_workflow = {
        "nodes": {
            "assess_quality": {
                "capability": AgentCapability.DATA_QUALITY_ASSESSMENT.value,
                "agent_id": "data_quality_agent",
                "parameters": {"threshold": 0.8}
            },
            "check_compliance": {
                "capability": AgentCapability.COMPLIANCE_CHECKING.value,
                "agent_id": "compliance_agent", 
                "parameters": {"regulations": ["GDPR", "CCPA"]}
            },
            "generate_remediation": {
                "capability": AgentCapability.DATA_REMEDIATION.value,
                "agent_id": "remediation_agent",
                "parameters": {"priority": "high"}
            }
        },
        "edges": [
            {"from": "assess_quality", "to": "check_compliance"},
            {"from": "check_compliance", "to": "generate_remediation"}
        ],
        "entry_point": "assess_quality"
    }
    
    orchestrator.create_workflow("data_quality_pipeline", quality_workflow)
    
    # Compliance Assessment Workflow
    compliance_workflow = {
        "nodes": {
            "risk_assessment": {
                "capability": AgentCapability.RISK_ASSESSMENT.value,
                "agent_id": "compliance_agent",
                "parameters": {"scope": "full"}
            },
            "compliance_check": {
                "capability": AgentCapability.COMPLIANCE_CHECKING.value,
                "agent_id": "compliance_agent",
                "parameters": {"detailed": True}
            }
        },
        "edges": [
            {"from": "risk_assessment", "to": "compliance_check"}
        ],
        "entry_point": "risk_assessment"
    }
    
    orchestrator.create_workflow("compliance_assessment", compliance_workflow)


# Create FastAPI app with message-driven architecture
app = FastAPI(
    title="EDGP AI Model Service",
    description="Enterprise Data Governance Platform with MCP and AWS messaging",
    version="2.0.0",
    lifespan=lifespan
)

# Setup CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup error handlers
setup_error_handlers(app)

# Include external service routers
app.include_router(external_router)
app.include_router(webhook_router)
app.include_router(llm_router)
app.include_router(integration_router)

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint providing basic service information."""
    return {
        "service": "EDGP AI Model",
        "version": "2.0.0",
        "description": "Enterprise Data Governance Platform with MCP and AWS messaging",
        "status": "running"
    }

# Metrics endpoint
@app.get("/metrics")
async def get_metrics():
    """Get basic system metrics."""
    return {
        "service": "edgp-ai-model",
        "uptime": system_monitor.get_uptime() if hasattr(system_monitor, 'get_uptime') else "unknown",
        "agents_count": len(initialized_agents),
        "status": "healthy"
    }

# Health check endpoints
@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint with MCP system status."""
    try:
        # Use safe method calls with fallbacks
        mcp_status = "active" 
        if hasattr(mcp_bus, 'get_status'):
            try:
                mcp_status = await mcp_bus.get_status()
            except:
                mcp_status = "error"
        
        external_status = "active"
        if hasattr(external_comm, 'get_status'):
            try:
                external_status = await external_comm.get_status()
            except:
                external_status = "error"
        
        # Use safe timestamp method
        timestamp = datetime.utcnow().isoformat()
        if hasattr(system_monitor, 'get_current_timestamp'):
            try:
                timestamp = system_monitor.get_current_timestamp()
            except:
                pass
        
        return HealthResponse(
            status="healthy",
            version="2.0.0",
            timestamp=timestamp,
            services={
                "mcp_message_bus": mcp_status,
                "external_communication": external_status,
                "orchestrator": "active" if orchestrator else "inactive",
                "system_monitor": "active"
            },
            agents=[getattr(agent, 'agent_id', 'unknown') for agent in initialized_agents],
            message_driven_architecture=True
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="unhealthy",
            version="2.0.0", 
            timestamp=datetime.utcnow().isoformat(),
            services={},
            agents=[],
            message_driven_architecture=True,
            error=str(e)
        )


@app.get("/agents/status", response_model=List[AgentStatusResponse])
async def get_agents_status() -> List[AgentStatusResponse]:
    """Get status of all MCP-enabled agents."""
    agent_statuses = []
    
    for agent in initialized_agents:
        try:
            capabilities = await agent.get_capabilities()
            status = AgentStatusResponse(
                agent_id=agent.agent_id,
                status="active",
                capabilities=[cap.value for cap in capabilities],
                message_queue_size=len(agent.message_queue),
                last_activity=agent.last_activity_timestamp
            )
            agent_statuses.append(status)
        except Exception as e:
            logger.error(f"Failed to get status for agent {agent.agent_id}: {e}")
            agent_statuses.append(
                AgentStatusResponse(
                    agent_id=agent.agent_id,
                    status="error",
                    capabilities=[],
                    message_queue_size=0,
                    error=str(e)
                )
            )
    
    return agent_statuses


# API v1 endpoints for agent management
@app.get("/api/v1/agents")
async def list_agents():
    """List all available agents."""
    # Create DataQualityAgent for testing
    from agents.data_quality.agent import DataQualityAgent
    test_agent = DataQualityAgent("data_quality_agent")
    
    return {
        "agents": [
            {
                "agent_id": "data_quality_agent",
                "name": "Data Quality Agent",
                "description": "Monitors and detects data quality issues",
                "capabilities": [cap.value for cap in test_agent.capabilities],
                "status": "active"
            }
        ],
        "total_count": 1
    }

@app.get("/api/v1/agents/{agent_id}/status")
async def get_agent_status(agent_id: str):
    """Get status of a specific agent."""
    if agent_id == "data_quality_agent":
        from agents.data_quality.agent import DataQualityAgent
        agent = DataQualityAgent(agent_id)
        return {
            "agent_id": agent_id,
            "status": "active",
            "capabilities": [cap.value for cap in agent.capabilities],
            "last_activity": datetime.utcnow().isoformat()
        }
    else:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent {agent_id} not found"
        )

@app.post("/api/v1/agents/{agent_id}/message")
async def send_agent_message(agent_id: str, message_data: dict):
    """Send a message to a specific agent."""
    if agent_id == "data_quality_agent":
        from agents.data_quality.agent import DataQualityAgent
        agent = DataQualityAgent(agent_id)
        
        # Process the message
        result = await agent.process_message(
            message_data.get("content", ""),
            context=message_data.get("context")
        )
        
        return {
            "agent_id": agent_id,
            "message_processed": True,
            "response": result,
            "timestamp": datetime.utcnow().isoformat()
        }
    else:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent {agent_id} not found"
        )

@app.post("/api/v1/agents/{agent_id}/execute")
async def execute_agent_capability(agent_id: str, capability_data: dict):
    """Execute a capability on a specific agent."""
    if agent_id == "data_quality_agent":
        from agents.data_quality.agent import DataQualityAgent
        from core.types.agent_types import AgentCapability
        
        agent = DataQualityAgent(agent_id)
        
        capability_name = capability_data.get("capability")
        parameters = capability_data.get("parameters", {})
        
        # Convert capability string to enum
        try:
            capability = AgentCapability(capability_name)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid capability: {capability_name}"
            )
        
        # Execute capability
        try:
            result = await agent.execute_capability(capability, parameters)
            return {
                "agent_id": agent_id,
                "capability": capability_name,
                "result": result,
                "timestamp": datetime.utcnow().isoformat()
            }
        except ValueError as e:
            if "not supported" in str(e):
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=str(e)
                )
            raise
    else:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent {agent_id} not found"
        )


# Standardized Agent Communication Test Endpoint
@app.post("/api/v2/agents/{agent_id}/standardized")
async def test_standardized_communication(agent_id: str, request_data: dict):
    """Test the new standardized agent communication interface."""
    try:
        if agent_id == "data_quality_agent":
            from agents.data_quality.agent import DataQualityAgent
            from core.types.communication import (
                StandardAgentInput, DataQualityInput, DataPayload, 
                OperationParameters, QualityDimension, AgentContext,
                create_standard_input
            )
            
            agent = DataQualityAgent(agent_id)
            
            # Create sample data payload
            data_payload = DataPayload(
                data_type=request_data.get("data_type", "tabular"),
                data_format=request_data.get("data_format", "json"),
                content=request_data.get("data", {"sample": "data"}),
                source_system=request_data.get("source", "test_api")
            )
            
            # Create operation parameters
            operation_params = OperationParameters(
                quality_threshold=request_data.get("quality_threshold", 0.8),
                include_recommendations=request_data.get("include_recommendations", True)
            )
            
            # Create data quality input
            quality_input = DataQualityInput(
                data_payload=data_payload,
                operation_parameters=operation_params,
                quality_dimensions=request_data.get("quality_dimensions", [
                    QualityDimension.COMPLETENESS,
                    QualityDimension.ACCURACY,
                    QualityDimension.CONSISTENCY
                ]),
                include_anomaly_detection=request_data.get("include_anomaly_detection", True),
                include_profiling=request_data.get("include_profiling", True)
            )
            
            # Create standardized input
            agent_input = create_standard_input(
                source_agent_id="test_api",
                target_agent_id=agent_id,
                capability_name=request_data.get("capability", "data_quality_assessment"),
                data=quality_input,
                context=AgentContext()
            )
            
            # Process using new standardized interface
            result = await agent.process_standardized_input(agent_input)
            
            return {
                "agent_id": agent_id,
                "interface": "standardized_v2",
                "result": result.dict(),
                "timestamp": datetime.utcnow().isoformat(),
                "success": True
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Agent {agent_id} not found or not supporting standardized interface"
            )
            
    except Exception as e:
        logger.error(f"Standardized communication test failed: {e}")
        return {
            "agent_id": agent_id,
            "interface": "standardized_v2",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat(),
            "success": False
        }


# MCP Message endpoints
@app.post("/messages/send", response_model=MessageResponse)
async def send_mcp_message(
    recipient: str,
    capability: str, 
    content: Dict[str, Any],
    priority: str = "normal"
) -> MessageResponse:
    """Send a message via MCP message bus."""
    try:
        message = MCPMessage(
            sender="api",
            recipient=recipient,
            message_type="capability_request",
            content={
                "capability": capability,
                "parameters": content
            },
            priority=MessagePriority(priority)
        )
        
        success = await mcp_bus.send_message(message)
        
        return MessageResponse(
            success=success,
            message_id=message.message_id,
            recipient=recipient,
            timestamp=message.timestamp
        )
    except Exception as e:
        logger.error(f"Failed to send MCP message: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Message sending failed: {str(e)}"
        )


# Workflow endpoints
@app.post("/workflows/{workflow_name}/execute", response_model=WorkflowResponse)
async def execute_workflow(
    workflow_name: str,
    initial_data: Dict[str, Any],
    background_tasks: BackgroundTasks
) -> WorkflowResponse:
    """Execute a LangGraph workflow."""
    try:
        # Start workflow execution in background
        execution_id = f"exec_{workflow_name}_{system_monitor.get_current_timestamp()}"
        
        background_tasks.add_task(
            _execute_workflow_background,
            workflow_name,
            initial_data,
            execution_id
        )
        
        return WorkflowResponse(
            execution_id=execution_id,
            workflow_name=workflow_name,
            status="started",
            timestamp=system_monitor.get_current_timestamp()
        )
    except Exception as e:
        logger.error(f"Failed to execute workflow {workflow_name}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Workflow execution failed: {str(e)}"
        )


async def _execute_workflow_background(
    workflow_name: str,
    initial_data: Dict[str, Any],
    execution_id: str
):
    """Execute workflow in background task."""
    try:
        logger.info(f"Starting workflow {workflow_name} with execution ID {execution_id}")
        
        result = await orchestrator.execute_workflow(workflow_name, initial_data)
        
        # Optionally send results via external communication
        if result.get("notify_external"):
            await external_comm.send_event(
                topic="workflow_completed",
                data={
                    "execution_id": execution_id,
                    "workflow_name": workflow_name,
                    "result": result
                }
            )
        
        logger.info(f"Workflow {workflow_name} completed successfully")
        
    except Exception as e:
        logger.error(f"Workflow {workflow_name} failed: {e}")
        
        # Send failure notification
        await external_comm.send_event(
            topic="workflow_failed",
            data={
                "execution_id": execution_id,
                "workflow_name": workflow_name,
                "error": str(e)
            }
        )


# Data processing endpoints
@app.post("/process/data-quality")
async def process_data_quality(request: ProcessingRequest):
    """Process data quality assessment via MCP messaging."""
    try:
        message = MCPMessage(
            sender="api",
            recipient="data_quality_agent",
            message_type="capability_request",
            content={
                "capability": AgentCapability.DATA_QUALITY_ASSESSMENT.value,
                "parameters": request.dict()
            },
            priority=MessagePriority.NORMAL
        )
        
        success = await mcp_bus.send_message(message)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to send data quality assessment request"
            )
        
        return {"status": "request_sent", "message_id": message.message_id}
        
    except Exception as e:
        logger.error(f"Data quality processing failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.post("/process/compliance-check")
async def process_compliance_check(request: ProcessingRequest):
    """Process compliance check via MCP messaging."""
    try:
        message = MCPMessage(
            sender="api",
            recipient="compliance_agent",
            message_type="capability_request",
            content={
                "capability": AgentCapability.COMPLIANCE_CHECKING.value,
                "parameters": request.dict()
            },
            priority=MessagePriority.HIGH
        )
        
        success = await mcp_bus.send_message(message)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to send compliance check request"
            )
        
        return {"status": "request_sent", "message_id": message.message_id}
        
    except Exception as e:
        logger.error(f"Compliance check processing failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.post("/process/data-remediation")
async def process_data_remediation(request: ProcessingRequest):
    """Process data remediation via MCP messaging."""
    try:
        message = MCPMessage(
            sender="api",
            recipient="remediation_agent",
            message_type="capability_request",
            content={
                "capability": AgentCapability.DATA_REMEDIATION.value,
                "parameters": request.dict()
            },
            priority=MessagePriority.NORMAL
        )
        
        success = await mcp_bus.send_message(message)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to send data remediation request"
            )
        
        return {"status": "request_sent", "message_id": message.message_id}
        
    except Exception as e:
        logger.error(f"Data remediation processing failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


# External communication endpoints
@app.post("/external/send-event")
async def send_external_event(
    topic: str,
    data: Dict[str, Any],
    priority: str = "normal"
):
    """Send event via external communication (AWS SNS)."""
    try:
        await external_comm.send_event(topic, data, priority)
        return {"status": "event_sent", "topic": topic}
    except Exception as e:
        logger.error(f"Failed to send external event: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.post("/external/queue-message")
async def queue_external_message(
    queue_name: str,
    message: Dict[str, Any],
    delay_seconds: int = 0
):
    """Queue message via external communication (AWS SQS)."""
    try:
        await external_comm.send_message(queue_name, message, delay_seconds)
        return {"status": "message_queued", "queue": queue_name}
    except Exception as e:
        logger.error(f"Failed to queue external message: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


# Monitoring endpoints
@app.get("/monitoring/metrics")
async def get_monitoring_metrics():
    """Get system monitoring metrics."""
    try:
        return {
            "system_metrics": system_monitor.get_metrics(),
            "mcp_bus_metrics": await mcp_bus.get_metrics(),
            "external_comm_metrics": await external_comm.get_metrics(),
            "agent_metrics": {
                agent.agent_id: await agent.get_metrics()
                for agent in initialized_agents
            }
        }
    except Exception as e:
        logger.error(f"Failed to get monitoring metrics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


# Enhanced agent operation endpoints
@app.post("/enhanced-agents/{agent_id}/process")
async def enhanced_agent_process(
    agent_id: str,
    data: Dict[str, Any],
    operation_type: str,
    use_external_services: bool = True
):
    """Process data using enhanced agent with external service integration."""
    try:
        from core.agents.enhanced import enhanced_agents_registry
        
        if agent_id not in enhanced_agents_registry:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Enhanced agent {agent_id} not found"
            )
        
        agent = enhanced_agents_registry[agent_id]
        
        if use_external_services:
            # Use full integration capabilities
            result = await agent.process_with_external_llm(
                data=data,
                operation_type=operation_type
            )
        else:
            # Use LLM only
            result = await agent.process_with_external_llm(
                data=data,
                operation_type=operation_type
            )
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Enhanced agent processing failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.post("/enhanced-agents/collaborate")
async def facilitate_enhanced_agent_collaboration(
    initiating_agent: str,
    target_agent: str,
    collaboration_prompt: str,
    shared_context: Dict[str, Any] = {}
):
    """Facilitate collaboration between enhanced agents."""
    try:
        from core.agents.enhanced import enhanced_agents_registry
        
        if initiating_agent not in enhanced_agents_registry:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Initiating agent {initiating_agent} not found"
            )
        
        agent = enhanced_agents_registry[initiating_agent]
        
        result = await agent.collaborate_with_agent(
            target_agent_id=target_agent,
            collaboration_prompt=collaboration_prompt,
            shared_context=shared_context
        )
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Enhanced agent collaboration failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.post("/cross-agent-workflows/{workflow_name}")
async def execute_cross_agent_workflow(
    workflow_name: str,
    agents_sequence: List[str],
    initial_data: Dict[str, Any],
    operation_config: Dict[str, Any] = {}
):
    """Execute workflow across multiple enhanced agents."""
    try:
        from core.agents.enhanced import cross_agent_manager
        
        if not cross_agent_manager:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Cross-agent manager not initialized"
            )
        
        result = await cross_agent_manager.execute_cross_agent_workflow(
            workflow_name=workflow_name,
            agents_sequence=agents_sequence,
            initial_data=initial_data,
            operation_config=operation_config
        )
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Cross-agent workflow execution failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.get("/cross-agent-workflows/{operation_id}/status")
async def get_cross_agent_workflow_status(operation_id: str):
    """Get status of cross-agent workflow operation."""
    try:
        from core.agents.enhanced import cross_agent_manager
        
        if not cross_agent_manager:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Cross-agent manager not initialized"
            )
        
        status_info = cross_agent_manager.get_operation_status(operation_id)
        
        if not status_info:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Operation {operation_id} not found"
            )
        
        return status_info
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get workflow status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


if __name__ == "__main__":
    settings = get_settings()
    
    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
        log_level="info"
    )
