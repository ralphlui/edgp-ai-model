"""
Main FastAPI application for EDGP AI Model service.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging

from core.config import get_settings
from core.llm_gateway import LLMGateway
from agents import (
    PolicySuggestionAgent,
    DataPrivacyComplianceAgent,
    DataQualityAgent,
    DataRemediationAgent,
    AnalyticsAgent
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for dependency injection
llm_gateway = None
agents = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    global llm_gateway, agents
    
    settings = get_settings()
    logger.info("Starting EDGP AI Model service...")
    
    # Initialize LLM Gateway
    try:
        # TODO: Initialize LLM Gateway when implementation is complete
        # llm_gateway = LLMGateway(settings)
        logger.info("LLM Gateway initialization skipped (placeholder mode)")
    except Exception as e:
        logger.error(f"Failed to initialize LLM Gateway: {e}")
        raise
    
    # Initialize agents
    try:
        agent_config = {"llm_gateway": llm_gateway}
        
        agents = {
            "policy_suggestion": PolicySuggestionAgent(config=agent_config),
            "data_privacy_compliance": DataPrivacyComplianceAgent(config=agent_config),
            "data_quality": DataQualityAgent(config=agent_config),
            "data_remediation": DataRemediationAgent(config=agent_config),
            "analytics": AnalyticsAgent(config=agent_config)
        }
        
        logger.info(f"Initialized {len(agents)} agents successfully")
    except Exception as e:
        logger.error(f"Failed to initialize agents: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down EDGP AI Model service...")


# Create FastAPI application
app = FastAPI(
    title="EDGP AI Model Service",
    description="Agentic AI microservice for master data management",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "EDGP AI Model",
        "version": "1.0.0",
        "status": "running",
        "agents": list(agents.keys()) if agents else []
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "agents_initialized": len(agents) if agents else 0,
        "llm_gateway_status": "initialized" if llm_gateway else "not_initialized"
    }


@app.get("/agents")
async def list_agents():
    """List all available agents."""
    if not agents:
        raise HTTPException(status_code=503, detail="Agents not initialized")
    
    return {
        "agents": [
            {
                "name": agent.name,
                "description": agent.description,
                "status": "active"
            }
            for agent in agents.values()
        ]
    }


# Agent-specific endpoints (placeholders)

@app.post("/agents/policy-suggestion/suggest-policies")
async def suggest_policies(request: dict):
    """Suggest data validation policies."""
    if "policy_suggestion" not in agents:
        raise HTTPException(status_code=503, detail="Policy Suggestion Agent not available")
    
    agent = agents["policy_suggestion"]
    
    # Placeholder response
    return await agent.suggest_data_validation_policies(
        data_schema=request.get("data_schema", {}),
        business_context=request.get("business_context", ""),
        compliance_requirements=request.get("compliance_requirements", [])
    )


@app.post("/agents/data-privacy-compliance/scan-risks")
async def scan_privacy_risks(request: dict):
    """Scan for data privacy risks."""
    if "data_privacy_compliance" not in agents:
        raise HTTPException(status_code=503, detail="Data Privacy Compliance Agent not available")
    
    agent = agents["data_privacy_compliance"]
    
    return await agent.scan_privacy_risks(
        data_sources=request.get("data_sources", []),
        data_schema=request.get("data_schema", {}),
        processing_context=request.get("processing_context", "")
    )


@app.post("/agents/data-quality/detect-anomalies")
async def detect_anomalies(request: dict):
    """Detect data quality anomalies."""
    if "data_quality" not in agents:
        raise HTTPException(status_code=503, detail="Data Quality Agent not available")
    
    agent = agents["data_quality"]
    
    return await agent.detect_anomalies(
        dataset=request.get("dataset", ""),
        data_schema=request.get("data_schema", {}),
        quality_rules=request.get("quality_rules", [])
    )


@app.post("/agents/data-remediation/generate-plan")
async def generate_remediation_plan(request: dict):
    """Generate data remediation plan."""
    if "data_remediation" not in agents:
        raise HTTPException(status_code=503, detail="Data Remediation Agent not available")
    
    agent = agents["data_remediation"]
    
    return await agent.generate_remediation_plan(
        issues=request.get("issues", []),
        business_rules=request.get("business_rules", {}),
        resource_constraints=request.get("resource_constraints", {})
    )


@app.post("/agents/analytics/generate-dashboard")
async def generate_dashboard(request: dict):
    """Generate analytics dashboard."""
    if "analytics" not in agents:
        raise HTTPException(status_code=503, detail="Analytics Agent not available")
    
    agent = agents["analytics"]
    
    return await agent.generate_quality_dashboard(
        datasets=request.get("datasets", []),
        time_range=request.get("time_range", "30d"),
        visualization_type=request.get("visualization_type", "comprehensive")
    )


if __name__ == "__main__":
    import uvicorn
    settings = get_settings()
    
    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
        log_level=settings.log_level.lower()
    )
