"""
Complete LangChain/LangGraph Integration Example

Demonstrates the full integration of LangChain/LangGraph with the
collaborative AI platform's shared services.
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional

# LangChain imports
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate

# Our components
from core.shared import create_shared_services, get_feature_status
from core.integrations.langchain_integration import create_langchain_integration
from core.agents.enhanced_base import ExampleAnalyticsAgent, LangChainAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ComplianceAgent(LangChainAgent):
    """Example compliance agent with LangChain integration."""
    
    def __init__(self, shared_services):
        super().__init__(
            agent_id="compliance_agent",
            agent_type="compliance",
            name="Compliance Agent",
            description="Performs compliance checks and regulatory analysis",
            shared_services=shared_services,
            capabilities=["compliance_check", "regulatory_scan", "risk_assessment"],
            system_prompt="You are a compliance agent that helps ensure regulatory compliance."
        )
    
    async def _analyze_input(self, input_text: str) -> Dict[str, Any]:
        input_lower = input_text.lower()
        return {
            "requires_compliance": any(word in input_lower for word in ["comply", "regulation", "legal"]),
            "requires_risk_assessment": any(word in input_lower for word in ["risk", "assess", "danger"]),
            "requires_scan": any(word in input_lower for word in ["scan", "check", "review"]),
            "urgency": "high" if any(word in input_lower for word in ["urgent", "critical"]) else "normal"
        }
    
    def _determine_required_capability(self, input_analysis: Dict[str, Any]) -> Optional[str]:
        if input_analysis.get("requires_risk_assessment"):
            return "risk_assessment"
        elif input_analysis.get("requires_scan"):
            return "regulatory_scan"
        else:
            return "compliance_check"
    
    async def _prepare_capability_context(self, capability: str, state) -> Dict[str, Any]:
        return {
            "capability": capability,
            "compliance_rules": ["GDPR", "CCPA", "SOX", "HIPAA"],
            "timestamp": datetime.now().isoformat()
        }
    
    async def _execute_capability(self, capability: str, state) -> Any:
        if capability == "compliance_check":
            return {
                "compliance_status": "COMPLIANT",
                "checked_regulations": ["GDPR", "CCPA"],
                "violations_found": 0,
                "recommendations": ["Continue current practices"]
            }
        elif capability == "regulatory_scan":
            return {
                "scan_completed": True,
                "regulations_scanned": 15,
                "issues_found": 2,
                "severity": "LOW"
            }
        elif capability == "risk_assessment":
            return {
                "risk_score": 25,
                "risk_level": "LOW",
                "risk_factors": ["Data exposure potential", "Access control gaps"],
                "mitigation_steps": ["Implement additional encryption", "Review access policies"]
            }
        return {"error": "Unknown capability"}
    
    async def _compile_response(self, state) -> str:
        if state.errors:
            return f"Compliance check encountered issues: {'; '.join(state.errors)}"
        
        response_parts = []
        for capability, result in state.agent_outputs.items():
            if isinstance(result, dict):
                if capability == "compliance_check":
                    status = result.get("compliance_status", "UNKNOWN")
                    violations = result.get("violations_found", 0)
                    response_parts.append(f"**Compliance Status:** {status}")
                    response_parts.append(f"**Violations Found:** {violations}")
                elif capability == "risk_assessment":
                    score = result.get("risk_score", 0)
                    level = result.get("risk_level", "UNKNOWN")
                    response_parts.append(f"**Risk Assessment:** {level} (Score: {score})")
                    if result.get("mitigation_steps"):
                        response_parts.append("**Recommended Actions:**")
                        for step in result["mitigation_steps"]:
                            response_parts.append(f"- {step}")
        
        return "\n".join(response_parts) if response_parts else "Compliance analysis completed."


class DataQualityAgent(LangChainAgent):
    """Example data quality agent with LangChain integration."""
    
    def __init__(self, shared_services):
        super().__init__(
            agent_id="data_quality_agent",
            agent_type="data_quality",
            name="Data Quality Agent",
            description="Performs data quality assessment and validation",
            shared_services=shared_services,
            capabilities=["quality_assessment", "data_validation", "anomaly_detection"],
            system_prompt="You are a data quality agent that helps ensure data integrity and quality."
        )
    
    async def _create_agent_tools(self):
        """Create data quality specific tools."""
        
        @tool
        async def validate_data_format(data_sample: str) -> str:
            """Validate data format and structure."""
            # Simplified validation logic
            if not data_sample:
                return "Invalid: Empty data"
            if len(data_sample) < 10:
                return "Warning: Data sample too small"
            return "Valid: Data format acceptable"
        
        @tool 
        async def detect_anomalies(data_points: str) -> str:
            """Detect anomalies in data points."""
            # Simplified anomaly detection
            try:
                points = [float(x.strip()) for x in data_points.split(",")]
                avg = sum(points) / len(points)
                anomalies = [p for p in points if abs(p - avg) > avg * 0.5]
                return f"Found {len(anomalies)} potential anomalies out of {len(points)} data points"
            except:
                return "Error: Unable to parse data points"
        
        return [validate_data_format, detect_anomalies]
    
    async def _analyze_input(self, input_text: str) -> Dict[str, Any]:
        input_lower = input_text.lower()
        return {
            "has_data_sample": any(word in input_lower for word in ["data", "sample", "dataset"]),
            "requires_validation": any(word in input_lower for word in ["validate", "check", "verify"]),
            "requires_anomaly_detection": any(word in input_lower for word in ["anomaly", "outlier", "unusual"]),
            "data_type": "numeric" if any(char.isdigit() for char in input_text) else "text"
        }
    
    def _determine_required_capability(self, input_analysis: Dict[str, Any]) -> Optional[str]:
        if input_analysis.get("requires_anomaly_detection"):
            return "anomaly_detection"
        elif input_analysis.get("requires_validation"):
            return "data_validation"
        else:
            return "quality_assessment"
    
    async def _prepare_capability_context(self, capability: str, state) -> Dict[str, Any]:
        return {
            "capability": capability,
            "quality_metrics": ["completeness", "accuracy", "consistency", "validity"],
            "timestamp": datetime.now().isoformat()
        }
    
    async def _execute_capability(self, capability: str, state) -> Any:
        if capability == "quality_assessment":
            return {
                "overall_score": 87,
                "completeness": 95,
                "accuracy": 88,
                "consistency": 82,
                "validity": 85,
                "issues_found": 3
            }
        elif capability == "data_validation":
            return {
                "validation_passed": True,
                "format_valid": True,
                "schema_compliant": True,
                "warnings": ["Some null values detected"]
            }
        elif capability == "anomaly_detection":
            return {
                "anomalies_detected": 2,
                "anomaly_types": ["outlier", "missing_pattern"],
                "confidence": 0.85,
                "affected_records": 15
            }
        return {"error": "Unknown capability"}
    
    async def _compile_response(self, state) -> str:
        if state.errors:
            return f"Data quality assessment encountered issues: {'; '.join(state.errors)}"
        
        response_parts = []
        for capability, result in state.agent_outputs.items():
            if isinstance(result, dict):
                if capability == "quality_assessment":
                    score = result.get("overall_score", 0)
                    issues = result.get("issues_found", 0)
                    response_parts.append(f"**Data Quality Score:** {score}/100")
                    response_parts.append(f"**Issues Found:** {issues}")
                elif capability == "anomaly_detection":
                    anomalies = result.get("anomalies_detected", 0)
                    confidence = result.get("confidence", 0)
                    response_parts.append(f"**Anomalies Detected:** {anomalies}")
                    response_parts.append(f"**Detection Confidence:** {confidence:.2%}")
        
        # Include tool results
        if state.tool_results:
            response_parts.append("\n**Tool Analysis:**")
            for tool_name, result in state.tool_results.items():
                response_parts.append(f"- {tool_name}: {result}")
        
        return "\n".join(response_parts) if response_parts else "Data quality assessment completed."


async def demonstrate_langchain_integration():
    """Demonstrate the complete LangChain/LangGraph integration."""
    
    logger.info("🚀 Starting LangChain/LangGraph Integration Demonstration")
    
    # Step 1: Initialize shared services
    logger.info("Step 1: Initializing shared services...")
    config = {
        "prompt": {"storage_path": ":memory:"},
        "rag": {"storage_path": ":memory:"},
        "memory": {"storage_path": ":memory:"},
        "knowledge": {"storage_path": ":memory:"},
        "context": {"storage_path": ":memory:"}
    }
    
    shared_services = await create_shared_services(config)
    
    # Check available features
    features = get_feature_status()
    logger.info(f"Available features: {features}")
    
    # Step 2: Initialize LangChain integration
    logger.info("Step 2: Initializing LangChain integration...")
    langchain_integration = await create_langchain_integration(shared_services, config)
    
    # Step 3: Create and initialize agents
    logger.info("Step 3: Creating and initializing agents...")
    agents = {
        "analytics": ExampleAnalyticsAgent(shared_services),
        "compliance": ComplianceAgent(shared_services),
        "data_quality": DataQualityAgent(shared_services)
    }
    
    # Initialize all agents
    for agent_id, agent in agents.items():
        await agent.initialize()
        logger.info(f"✅ Agent {agent_id} initialized with capabilities: {agent.get_capabilities()}")
    
    # Step 4: Setup knowledge base with sample data
    logger.info("Step 4: Setting up knowledge base...")
    if shared_services.is_feature_available("knowledge"):
        # Create some sample entities
        analytics_entity = await shared_services.knowledge.create_entity(
            name="Data Analytics",
            entity_type="process",
            properties={"description": "Process of analyzing data for insights"}
        )
        
        compliance_entity = await shared_services.knowledge.create_entity(
            name="Regulatory Compliance",
            entity_type="requirement",
            properties={"description": "Adherence to regulatory standards"}
        )
        
        # Create relationship
        await shared_services.knowledge.create_relationship(
            source_id=analytics_entity,
            target_id=compliance_entity,
            relationship_type="must_comply_with"
        )
    
    # Step 5: Add sample documents to RAG
    logger.info("Step 5: Adding sample documents to RAG...")
    if shared_services.is_feature_available("rag"):
        sample_docs = [
            {
                "content": "Best practices for data analytics include ensuring data quality, using appropriate statistical methods, and validating results through multiple approaches.",
                "metadata": {"type": "best_practices", "domain": "analytics"}
            },
            {
                "content": "Compliance requirements include GDPR for data protection, SOX for financial reporting, and HIPAA for healthcare data privacy.",
                "metadata": {"type": "regulations", "domain": "compliance"}
            },
            {
                "content": "Data quality assessment should cover completeness, accuracy, consistency, timeliness, and validity of data elements.",
                "metadata": {"type": "guidelines", "domain": "data_quality"}
            }
        ]
        
        for doc in sample_docs:
            await shared_services.rag.add_document(**doc)
    
    # Step 6: Create and execute LangGraph workflows
    logger.info("Step 6: Creating LangGraph workflows...")
    
    # Create a comprehensive analysis workflow
    workflow_agents = [
        {"agent_id": "data_quality_agent", "type": "data_quality"},
        {"agent_id": "compliance_agent", "type": "compliance"},
        {"agent_id": "analytics_agent", "type": "analytics"}
    ]
    
    langchain_integration.create_custom_workflow(
        "comprehensive_analysis",
        workflow_agents,
        "sequential"
    )
    
    # Create a parallel workflow for quick assessment
    parallel_agents = [
        {"agent_id": "data_quality_agent", "type": "data_quality"},
        {"agent_id": "compliance_agent", "type": "compliance"}
    ]
    
    langchain_integration.create_custom_workflow(
        "parallel_assessment",
        parallel_agents,
        "parallel"
    )
    
    # Step 7: Execute individual agents
    logger.info("Step 7: Testing individual agents...")
    
    test_scenarios = [
        {
            "agent": "analytics", 
            "message": "Analyze the user engagement data and provide trend insights for the last quarter"
        },
        {
            "agent": "compliance",
            "message": "Perform a regulatory compliance check for GDPR requirements"
        },
        {
            "agent": "data_quality",
            "message": "Validate this dataset: 1.2, 3.4, 5.6, 100.0, 2.1, 3.3 and detect any anomalies"
        }
    ]
    
    for scenario in test_scenarios:
        logger.info(f"\n📋 Testing {scenario['agent']} agent...")
        agent = agents[scenario['agent']]
        
        result = await agent.process_message(
            message=scenario['message'],
            user_id="demo_user",
            session_id="demo_session"
        )
        
        logger.info(f"✅ Agent Response: {result.data.get('response', 'No response')}")
        logger.info(f"   Capabilities used: {result.data.get('capabilities_used', [])}")
        logger.info(f"   Tools used: {result.data.get('tools_used', [])}")
        logger.info(f"   Processing steps: {result.data.get('processing_steps', 0)}")
        
        # Show metrics
        metrics = agent.get_metrics()
        logger.info(f"   Agent metrics: {metrics}")
    
    # Step 8: Execute LangGraph workflows
    logger.info("\nStep 8: Testing LangGraph workflows...")
    
    workflow_tests = [
        {
            "workflow": "comprehensive_analysis",
            "message": "Perform a comprehensive analysis of our customer data including quality checks, compliance validation, and trend analysis"
        },
        {
            "workflow": "parallel_assessment", 
            "message": "Quick assessment: check data quality and compliance status"
        }
    ]
    
    for test in workflow_tests:
        logger.info(f"\n🔄 Executing workflow: {test['workflow']}")
        
        workflow_result = await langchain_integration.execute_workflow(
            workflow_name=test['workflow'],
            message=test['message'],
            user_id="demo_user",
            session_id="workflow_session"
        )
        
        if workflow_result['success']:
            logger.info(f"✅ Workflow completed successfully")
            logger.info(f"   Executed agents: {workflow_result['executed_agents']}")
            logger.info(f"   Total steps: {workflow_result['step_count']}")
            logger.info(f"   Duration: {workflow_result['duration']:.2f} seconds")
            
            # Show agent outputs
            for agent_id, output in workflow_result.get('agent_outputs', {}).items():
                logger.info(f"   {agent_id} output: {str(output)[:100]}...")
        else:
            logger.error(f"❌ Workflow failed: {workflow_result.get('error', 'Unknown error')}")
    
    # Step 9: Demonstrate shared services integration
    logger.info("\nStep 9: Demonstrating shared services integration...")
    
    # Check memory storage
    if shared_services.is_feature_available("memory"):
        memories = await shared_services.memory.retrieve_memories(
            query="analytics compliance data quality",
            limit=5
        )
        logger.info(f"   Retrieved {len(memories)} relevant memories")
    
    # Check knowledge base
    if shared_services.is_feature_available("knowledge"):
        entities = await shared_services.knowledge.search_entities(
            query="analytics compliance",
            limit=3
        )
        logger.info(f"   Found {len(entities)} relevant knowledge entities")
    
    # Check context management
    if shared_services.is_feature_available("context"):
        try:
            # This would show session management
            logger.info("   Context management: Sessions and conversations tracked")
        except Exception as e:
            logger.warning(f"   Context management: {str(e)}")
    
    # Step 10: Show integration benefits
    logger.info("\nStep 10: Integration Benefits Summary")
    logger.info("✅ Standardized Communication: All agents use StandardAgentInput/Output")
    logger.info("✅ Shared Memory: Agents can access and store shared memories")
    logger.info("✅ Knowledge Integration: Agents leverage shared knowledge base")
    logger.info("✅ RAG Integration: Agents can search and retrieve relevant documents")
    logger.info("✅ Context Management: Session and conversation state managed")
    logger.info("✅ Tool Integration: LangChain tools integrated with shared services")
    logger.info("✅ Workflow Orchestration: LangGraph workflows with shared state")
    logger.info("✅ Performance Tracking: Comprehensive metrics and monitoring")
    
    # Cleanup
    logger.info("\nStep 11: Cleanup...")
    for agent in agents.values():
        await agent.shutdown()
    
    await langchain_integration.shutdown()
    await shared_services.shutdown()
    
    logger.info("🎉 LangChain/LangGraph Integration Demonstration Complete!")


async def main():
    """Main function to run the demonstration."""
    try:
        await demonstrate_langchain_integration()
    except Exception as e:
        logger.error(f"Demonstration failed: {str(e)}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
