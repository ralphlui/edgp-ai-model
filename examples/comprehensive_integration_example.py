"""
Comprehensive Integration Example

Demonstrates the complete collaborative AI platform using all shared services:
- Standardized communication
- Prompt engineering
- RAG management
- Memory management
- Knowledge management
- Tool management
- Context management

This example shows how multiple agents collaborate using shared features
to provide comprehensive analytics and insights.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from pathlib import Path

# Import core shared services
from core.shared import (
    # Communication
    StandardAgentInput, StandardAgentOutput, Priority, TracingContext,
    create_standard_input, create_standard_output, create_error_output,
    
    # Service availability checks
    get_feature_status, create_shared_services
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CollaborativeAnalyticsWorkflow:
    """
    Comprehensive workflow demonstrating multi-agent collaboration
    using all shared services for analytics and insights generation.
    """
    
    def __init__(self):
        self.services = None
        self.session_id = None
        self.conversation_id = None
        self.workflow_id = "collaborative_analytics_v1"
        
    async def initialize(self, config: Dict[str, Any] = None):
        """Initialize the workflow with shared services."""
        # Check feature availability
        features = get_feature_status()
        logger.info("Available features: %s", features)
        
        # Initialize shared services
        self.services = await create_shared_services(config)
        
        # Create session and conversation for the workflow
        if self.services.is_feature_available("context"):
            self.session_id = await self.services.context.create_session(
                user_id="analytics_user",
                session_type="collaborative_analytics"
            )
            
            self.conversation_id = await self.services.context.start_conversation(
                session_id=self.session_id,
                title="Comprehensive Analytics Workflow"
            )
        
        logger.info("Workflow initialized with session: %s", self.session_id)
        
    async def setup_knowledge_base(self):
        """Setup the knowledge base with analytics concepts."""
        if not self.services.is_feature_available("knowledge"):
            logger.warning("Knowledge management not available")
            return
        
        logger.info("Setting up knowledge base...")
        
        # Create core analytics entities
        entities = {
            "user_engagement": {
                "name": "User Engagement",
                "entity_type": "metric",
                "properties": {
                    "definition": "Measure of user interaction with the platform",
                    "unit": "percentage",
                    "importance": "high"
                }
            },
            "conversion_rate": {
                "name": "Conversion Rate", 
                "entity_type": "metric",
                "properties": {
                    "definition": "Percentage of users who complete desired actions",
                    "unit": "percentage",
                    "calculation": "conversions / total_visitors * 100"
                }
            },
            "churn_analysis": {
                "name": "Churn Analysis",
                "entity_type": "process",
                "properties": {
                    "definition": "Analysis of user attrition patterns",
                    "importance": "critical"
                }
            }
        }
        
        entity_ids = {}
        for key, entity_data in entities.items():
            entity_id = await self.services.knowledge.create_entity(**entity_data)
            entity_ids[key] = entity_id
            logger.info("Created entity: %s (%s)", entity_data["name"], entity_id)
        
        # Create relationships between entities
        relationships = [
            {
                "source_id": entity_ids["user_engagement"],
                "target_id": entity_ids["conversion_rate"],
                "relationship_type": "influences",
                "properties": {"correlation": "positive", "strength": "strong"}
            },
            {
                "source_id": entity_ids["user_engagement"],
                "target_id": entity_ids["churn_analysis"],
                "relationship_type": "inversely_related",
                "properties": {"correlation": "negative", "strength": "medium"}
            }
        ]
        
        for rel in relationships:
            rel_id = await self.services.knowledge.create_relationship(**rel)
            logger.info("Created relationship: %s", rel_id)
    
    async def setup_prompts(self):
        """Setup agent-specific prompts."""
        if not self.services.is_feature_available("prompt"):
            logger.warning("Prompt management not available")
            return
        
        logger.info("Setting up prompts...")
        
        # System prompts for different agent types
        prompts = {
            "data_analysis": {
                "name": "data_analysis_system_prompt",
                "content": """You are a Data Analysis Agent specializing in comprehensive data analytics.

Your responsibilities:
- Analyze user engagement metrics and patterns
- Identify trends and anomalies in data
- Generate statistical insights and summaries
- Collaborate with other agents for comprehensive analysis

Context: {context}
Data: {data}
Analysis Focus: {focus}

Provide detailed analysis with:
1. Key metrics and trends
2. Statistical insights
3. Anomalies or notable patterns
4. Recommendations for further investigation""",
                "category": "system",
                "agent_ids": ["data_analysis_agent"]
            },
            "insights_generation": {
                "name": "insights_generation_prompt",
                "content": """You are an Insights Generation Agent focused on deriving actionable insights.

Your role:
- Transform data analysis into business insights
- Identify opportunities and risks
- Generate strategic recommendations
- Connect patterns to business outcomes

Analysis Results: {analysis_results}
Business Context: {business_context}
Historical Data: {historical_context}

Generate insights including:
1. Key business implications
2. Actionable recommendations
3. Risk assessment
4. Opportunity identification
5. Success metrics for proposed actions""",
                "category": "agent_specific",
                "agent_ids": ["insights_agent"]
            },
            "report_generation": {
                "name": "report_generation_template",
                "content": """Generate a comprehensive analytics report based on the following:

Analysis: {analysis}
Insights: {insights}
Timeframe: {timeframe}
Stakeholders: {stakeholders}

Report Structure:
# Executive Summary
{executive_summary}

# Key Findings
{key_findings}

# Detailed Analysis
{detailed_analysis}

# Recommendations
{recommendations}

# Next Steps
{next_steps}""",
                "category": "template",
                "agent_ids": ["reporting_agent"]
            }
        }
        
        for prompt_data in prompts.values():
            prompt_id = await self.services.prompt.create_prompt(**prompt_data)
            logger.info("Created prompt: %s (%s)", prompt_data["name"], prompt_id)
    
    async def setup_rag_documents(self):
        """Setup RAG system with relevant documents."""
        if not self.services.is_feature_available("rag"):
            logger.warning("RAG management not available") 
            return
        
        logger.info("Setting up RAG documents...")
        
        # Sample analytics documents
        documents = [
            {
                "content": """User Engagement Best Practices:
                
1. Content Personalization: Tailor content based on user behavior and preferences
2. Interactive Features: Implement polls, quizzes, and interactive content
3. Mobile Optimization: Ensure seamless mobile experience
4. Loading Speed: Optimize page load times for better engagement
5. Social Features: Enable sharing, commenting, and community interaction
6. Gamification: Add points, badges, and achievements
7. Push Notifications: Send relevant, timely notifications
8. A/B Testing: Continuously test and optimize engagement strategies""",
                "metadata": {
                    "type": "best_practices",
                    "category": "user_engagement",
                    "source": "analytics_guidelines",
                    "importance": "high"
                }
            },
            {
                "content": """Conversion Rate Optimization Strategies:
                
1. Landing Page Optimization:
   - Clear value propositions
   - Simplified forms
   - Trust signals and testimonials
   - Fast loading times
   
2. User Experience Improvements:
   - Intuitive navigation
   - Mobile responsiveness
   - Clear call-to-action buttons
   - Reduced friction in user journey
   
3. Personalization:
   - Dynamic content based on user segments
   - Behavioral targeting
   - Personalized recommendations
   
4. Analytics and Testing:
   - A/B testing for all elements
   - Funnel analysis
   - Heatmap analysis
   - User feedback collection""",
                "metadata": {
                    "type": "strategy_guide",
                    "category": "conversion_optimization",
                    "source": "marketing_best_practices"
                }
            },
            {
                "content": """Churn Prediction and Prevention:
                
Early Warning Indicators:
- Decreased login frequency
- Reduced feature usage
- Support ticket patterns
- Engagement score decline
- Payment history changes

Prevention Strategies:
1. Proactive Outreach:
   - Automated email campaigns
   - Personalized offers
   - Customer success check-ins
   
2. Product Improvements:
   - Feature usage analytics
   - User feedback implementation
   - Performance optimizations
   
3. Engagement Programs:
   - Loyalty programs
   - Educational content
   - Community building
   
4. Predictive Analytics:
   - Machine learning models
   - Risk scoring
   - Intervention triggers""",
                "metadata": {
                    "type": "strategy_guide",
                    "category": "churn_prevention",
                    "source": "customer_success"
                }
            }
        ]
        
        for doc_data in documents:
            doc_id = await self.services.rag.add_document(**doc_data)
            logger.info("Added document: %s", doc_id)
    
    async def store_historical_context(self):
        """Store historical context and memories."""
        if not self.services.is_feature_available("memory"):
            logger.warning("Memory management not available")
            return
        
        logger.info("Storing historical context...")
        
        # Store important memories from previous analyses
        memories = [
            {
                "content": "Q3 showed 15% increase in user engagement, primarily driven by mobile app improvements",
                "memory_type": "semantic",
                "metadata": {
                    "timeframe": "Q3_2023",
                    "category": "engagement_insights",
                    "importance": "high"
                }
            },
            {
                "content": "Conversion rate optimization campaign resulted in 12% improvement after landing page redesign",
                "memory_type": "semantic", 
                "metadata": {
                    "campaign": "landing_page_optimization",
                    "timeframe": "Q2_2023",
                    "result_type": "success"
                }
            },
            {
                "content": "Churn analysis revealed that users inactive for 7+ days have 80% probability of churning within 30 days",
                "memory_type": "factual",
                "metadata": {
                    "analysis_type": "churn_prediction",
                    "statistical_confidence": "high",
                    "model": "logistic_regression"
                }
            }
        ]
        
        for memory_data in memories:
            memory_id = await self.services.memory.store_memory(**memory_data)
            logger.info("Stored memory: %s", memory_id)
    
    async def register_analysis_tools(self):
        """Register analytics tools."""
        if not self.services.is_feature_available("tools"):
            logger.warning("Tool management not available")
            return
        
        logger.info("Registering analysis tools...")
        
        # Define analysis tools
        async def calculate_engagement_score(metrics: Dict[str, float]) -> Dict[str, Any]:
            """Calculate composite engagement score."""
            weights = {
                "session_duration": 0.3,
                "page_views": 0.25,
                "interactions": 0.25,
                "return_visits": 0.2
            }
            
            score = sum(metrics.get(metric, 0) * weight for metric, weight in weights.items())
            
            return {
                "engagement_score": min(score, 100),  # Cap at 100
                "components": metrics,
                "weights_used": weights,
                "calculation_time": datetime.now().isoformat()
            }
        
        async def analyze_conversion_funnel(funnel_data: List[Dict[str, Any]]) -> Dict[str, Any]:
            """Analyze conversion funnel performance."""
            if not funnel_data:
                return {"error": "No funnel data provided"}
            
            stages = []
            total_users = funnel_data[0].get("users", 0)
            
            for i, stage in enumerate(funnel_data):
                stage_users = stage.get("users", 0)
                conversion_rate = (stage_users / total_users * 100) if total_users > 0 else 0
                drop_off = 0
                
                if i > 0:
                    prev_users = funnel_data[i-1].get("users", 0)
                    drop_off = prev_users - stage_users
                
                stages.append({
                    "stage": stage.get("stage", f"Stage_{i+1}"),
                    "users": stage_users,
                    "conversion_rate": round(conversion_rate, 2),
                    "drop_off": drop_off
                })
            
            return {
                "funnel_analysis": stages,
                "total_conversion_rate": stages[-1]["conversion_rate"] if stages else 0,
                "biggest_drop_off": max(stages, key=lambda x: x["drop_off"])["stage"] if stages else None
            }
        
        async def predict_churn_risk(user_metrics: Dict[str, Any]) -> Dict[str, Any]:
            """Predict churn risk based on user metrics."""
            # Simplified churn prediction logic
            risk_factors = {
                "days_since_last_login": user_metrics.get("days_since_last_login", 0),
                "session_frequency": user_metrics.get("sessions_per_week", 0),
                "feature_usage": user_metrics.get("features_used", 0),
                "support_tickets": user_metrics.get("support_tickets_last_month", 0)
            }
            
            # Calculate risk score (simplified model)
            risk_score = 0
            risk_score += min(risk_factors["days_since_last_login"] * 5, 50)
            risk_score += max(0, (7 - risk_factors["session_frequency"]) * 10)
            risk_score += max(0, (5 - risk_factors["feature_usage"]) * 8)
            risk_score += risk_factors["support_tickets"] * 15
            
            risk_level = "low"
            if risk_score > 70:
                risk_level = "high"
            elif risk_score > 40:
                risk_level = "medium"
            
            return {
                "churn_risk_score": min(risk_score, 100),
                "risk_level": risk_level,
                "risk_factors": risk_factors,
                "recommendations": self._get_churn_recommendations(risk_level)
            }
        
        # Register tools
        tools = [
            {
                "name": "engagement_calculator",
                "description": "Calculate composite user engagement score",
                "function": calculate_engagement_score,
                "parameters": {
                    "metrics": "Dictionary of engagement metrics"
                },
                "security_level": "standard"
            },
            {
                "name": "funnel_analyzer", 
                "description": "Analyze conversion funnel performance",
                "function": analyze_conversion_funnel,
                "parameters": {
                    "funnel_data": "List of funnel stage data with user counts"
                },
                "security_level": "standard"
            },
            {
                "name": "churn_predictor",
                "description": "Predict user churn risk",
                "function": predict_churn_risk,
                "parameters": {
                    "user_metrics": "Dictionary of user behavior metrics"
                },
                "security_level": "standard"
            }
        ]
        
        for tool_data in tools:
            tool_id = await self.services.tools.register_tool(**tool_data)
            logger.info("Registered tool: %s (%s)", tool_data["name"], tool_id)
    
    def _get_churn_recommendations(self, risk_level: str) -> List[str]:
        """Get churn prevention recommendations based on risk level."""
        recommendations = {
            "low": [
                "Continue current engagement strategies",
                "Monitor for any changes in behavior",
                "Provide valuable content and features"
            ],
            "medium": [
                "Send personalized re-engagement email",
                "Offer product tutorials or tips",
                "Check if user needs help with features",
                "Consider special offers or incentives"
            ],
            "high": [
                "Immediate personal outreach",
                "Offer customer success consultation",
                "Provide significant incentives to stay",
                "Conduct exit interview if churned",
                "Implement retention-focused features"
            ]
        }
        return recommendations.get(risk_level, [])
    
    async def execute_comprehensive_analysis(self, analysis_request: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a comprehensive multi-agent analysis workflow."""
        logger.info("Starting comprehensive analysis workflow...")
        
        try:
            with TracingContext("comprehensive_analysis") as trace:
                # Store shared context for this analysis
                if self.services.is_feature_available("context"):
                    await self.services.context.store_shared_context(
                        name="current_analysis",
                        data=analysis_request,
                        scope="session"
                    )
                
                # Step 1: Data Analysis
                data_analysis_result = await self._perform_data_analysis(analysis_request)
                
                # Step 2: Generate Insights
                insights_result = await self._generate_insights(data_analysis_result)
                
                # Step 3: Create Comprehensive Report
                report_result = await self._create_report(data_analysis_result, insights_result)
                
                # Step 4: Store Results
                await self._store_analysis_results(data_analysis_result, insights_result, report_result)
                
                # Compile final results
                comprehensive_result = {
                    "analysis_id": str(trace.trace_id),
                    "timestamp": datetime.now().isoformat(),
                    "data_analysis": data_analysis_result,
                    "insights": insights_result,
                    "report": report_result,
                    "recommendations": self._compile_recommendations(insights_result),
                    "next_steps": self._suggest_next_steps(analysis_request, insights_result)
                }
                
                logger.info("Comprehensive analysis completed successfully")
                return comprehensive_result
                
        except Exception as e:
            logger.error("Analysis workflow failed: %s", str(e))
            return {
                "error": "Analysis workflow failed",
                "error_details": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def _perform_data_analysis(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Perform detailed data analysis."""
        logger.info("Performing data analysis...")
        
        analysis_results = {}
        
        # Use tools for analysis if available
        if self.services.is_feature_available("tools"):
            # Calculate engagement scores
            if "engagement_metrics" in request:
                engagement_result = await self.services.tools.execute_tool(
                    "engagement_calculator",
                    request["engagement_metrics"]
                )
                analysis_results["engagement_analysis"] = engagement_result
            
            # Analyze conversion funnel
            if "funnel_data" in request:
                funnel_result = await self.services.tools.execute_tool(
                    "funnel_analyzer",
                    request["funnel_data"]
                )
                analysis_results["funnel_analysis"] = funnel_result
            
            # Predict churn risks
            if "user_cohorts" in request:
                churn_results = []
                for cohort in request["user_cohorts"]:
                    churn_result = await self.services.tools.execute_tool(
                        "churn_predictor",
                        cohort
                    )
                    churn_results.append(churn_result)
                analysis_results["churn_analysis"] = churn_results
        
        # Retrieve relevant memories for context
        if self.services.is_feature_available("memory"):
            relevant_memories = await self.services.memory.retrieve_memories(
                query="engagement analysis conversion churn",
                limit=5
            )
            analysis_results["historical_context"] = [
                {"content": mem.content, "metadata": mem.metadata}
                for mem in relevant_memories
            ]
        
        # Search for relevant documents
        if self.services.is_feature_available("rag"):
            doc_results = await self.services.rag.search(
                query="user engagement conversion optimization",
                top_k=3
            )
            analysis_results["relevant_knowledge"] = [
                {"content": doc.content, "metadata": doc.metadata}
                for doc in doc_results
            ]
        
        return analysis_results
    
    async def _generate_insights(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate actionable insights from analysis results."""
        logger.info("Generating insights...")
        
        insights = {
            "key_insights": [],
            "opportunities": [],
            "risks": [],
            "trends": [],
            "recommendations": []
        }
        
        # Analyze engagement insights
        if "engagement_analysis" in analysis_results:
            engagement = analysis_results["engagement_analysis"]
            if engagement.get("engagement_score", 0) < 50:
                insights["risks"].append("Low user engagement score indicates potential retention issues")
                insights["opportunities"].append("Significant room for engagement improvement")
                insights["recommendations"].append("Focus on engagement optimization strategies")
        
        # Analyze funnel insights
        if "funnel_analysis" in analysis_results:
            funnel = analysis_results["funnel_analysis"]
            if funnel.get("total_conversion_rate", 0) < 10:
                insights["risks"].append("Low conversion rate affecting business goals")
                insights["opportunities"].append("Conversion optimization could significantly impact revenue")
        
        # Analyze churn insights
        if "churn_analysis" in analysis_results:
            high_risk_users = sum(1 for result in analysis_results["churn_analysis"] 
                                if result.get("risk_level") == "high")
            if high_risk_users > 0:
                insights["risks"].append(f"{high_risk_users} users at high churn risk")
                insights["recommendations"].append("Implement immediate retention campaigns")
        
        # Generate trend analysis
        if "historical_context" in analysis_results:
            insights["trends"].append("Historical data shows seasonal engagement patterns")
            insights["key_insights"].append("Previous optimization efforts have shown positive results")
        
        return insights
    
    async def _create_report(self, analysis_results: Dict[str, Any], insights: Dict[str, Any]) -> Dict[str, Any]:
        """Create comprehensive analytics report."""
        logger.info("Creating comprehensive report...")
        
        # Get report template if available
        if self.services.is_feature_available("prompt"):
            template = await self.services.prompt.get_prompt_by_name("report_generation_template")
            if template:
                # Render report using template (simplified for demo)
                report_content = template.content.format(
                    analysis=str(analysis_results),
                    insights=str(insights),
                    timeframe="Current Period",
                    stakeholders="Product, Marketing, Customer Success",
                    executive_summary="Comprehensive analysis reveals opportunities for improvement",
                    key_findings="; ".join(insights.get("key_insights", [])),
                    detailed_analysis="See analysis results section",
                    recommendations="; ".join(insights.get("recommendations", [])),
                    next_steps="Implement priority recommendations"
                )
            else:
                report_content = "Template not found - using default format"
        else:
            report_content = "Prompt management not available - simplified report"
        
        report = {
            "report_id": f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "generated_at": datetime.now().isoformat(),
            "content": report_content,
            "executive_summary": {
                "total_metrics_analyzed": len(analysis_results),
                "key_insights_count": len(insights.get("key_insights", [])),
                "risks_identified": len(insights.get("risks", [])),
                "opportunities_found": len(insights.get("opportunities", []))
            },
            "sections": {
                "data_analysis": analysis_results,
                "insights": insights,
                "methodology": "Multi-agent collaborative analysis using shared services"
            }
        }
        
        return report
    
    async def _store_analysis_results(self, analysis: Dict[str, Any], insights: Dict[str, Any], report: Dict[str, Any]):
        """Store analysis results for future reference."""
        logger.info("Storing analysis results...")
        
        # Store in memory for future reference
        if self.services.is_feature_available("memory"):
            # Store key insights as memories
            for insight in insights.get("key_insights", []):
                await self.services.memory.store_memory(
                    content=insight,
                    memory_type="semantic",
                    metadata={
                        "analysis_session": self.session_id,
                        "timestamp": datetime.now().isoformat(),
                        "type": "insight"
                    }
                )
            
            # Store important recommendations
            for rec in insights.get("recommendations", []):
                await self.services.memory.store_memory(
                    content=f"Recommendation: {rec}",
                    memory_type="episodic",
                    metadata={
                        "analysis_session": self.session_id,
                        "type": "recommendation"
                    }
                )
        
        # Update knowledge base with new insights
        if self.services.is_feature_available("knowledge"):
            # Create facts from analysis results
            if "engagement_analysis" in analysis:
                await self.services.knowledge.create_fact(
                    content=f"Current engagement score: {analysis['engagement_analysis'].get('engagement_score', 'N/A')}",
                    fact_type="measurement",
                    confidence=0.9
                )
        
        # Store conversation in context
        if self.services.is_feature_available("context"):
            await self.services.context.add_message(
                conversation_id=self.conversation_id,
                role="assistant",
                content=f"Completed comprehensive analysis with {len(insights.get('key_insights', []))} insights",
                metadata={
                    "analysis_id": report.get("report_id"),
                    "insights_count": len(insights.get("key_insights", [])),
                    "recommendations_count": len(insights.get("recommendations", []))
                }
            )
    
    def _compile_recommendations(self, insights: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Compile prioritized recommendations."""
        recommendations = []
        
        # High priority recommendations from risks
        for risk in insights.get("risks", []):
            recommendations.append({
                "priority": "high",
                "category": "risk_mitigation",
                "recommendation": f"Address: {risk}",
                "impact": "high"
            })
        
        # Medium priority from opportunities
        for opportunity in insights.get("opportunities", []):
            recommendations.append({
                "priority": "medium",
                "category": "opportunity",
                "recommendation": f"Pursue: {opportunity}",
                "impact": "medium"
            })
        
        # Add specific recommendations
        for rec in insights.get("recommendations", []):
            recommendations.append({
                "priority": "medium",
                "category": "improvement",
                "recommendation": rec,
                "impact": "medium"
            })
        
        return recommendations
    
    def _suggest_next_steps(self, request: Dict[str, Any], insights: Dict[str, Any]) -> List[str]:
        """Suggest concrete next steps."""
        next_steps = []
        
        # Based on analysis type
        if "engagement_metrics" in request:
            next_steps.append("Implement engagement optimization strategies")
            next_steps.append("Set up automated engagement monitoring")
        
        if "funnel_data" in request:
            next_steps.append("Optimize conversion funnel bottlenecks")
            next_steps.append("A/B test funnel improvements")
        
        if "user_cohorts" in request:
            next_steps.append("Deploy targeted retention campaigns")
            next_steps.append("Monitor churn risk indicators")
        
        # General next steps
        next_steps.extend([
            "Schedule follow-up analysis in 2 weeks",
            "Share insights with stakeholder teams",
            "Implement top priority recommendations",
            "Monitor key metrics for improvement"
        ])
        
        return next_steps
    
    async def shutdown(self):
        """Shutdown the workflow and cleanup resources."""
        logger.info("Shutting down collaborative analytics workflow...")
        
        if self.services:
            await self.services.shutdown()
        
        logger.info("Workflow shutdown complete")


async def main():
    """Main function demonstrating the comprehensive collaborative AI platform."""
    
    # Configuration for shared services
    config = {
        "prompt": {"storage_path": ":memory:"},
        "rag": {"storage_path": ":memory:"},
        "memory": {"storage_path": ":memory:"},
        "knowledge": {"storage_path": ":memory:"},
        "context": {"storage_path": ":memory:"}
    }
    
    # Initialize the collaborative workflow
    workflow = CollaborativeAnalyticsWorkflow()
    
    try:
        # Initialize workflow
        await workflow.initialize(config)
        
        # Setup all components
        await workflow.setup_knowledge_base()
        await workflow.setup_prompts()
        await workflow.setup_rag_documents()
        await workflow.store_historical_context()
        await workflow.register_analysis_tools()
        
        # Example analysis request
        analysis_request = {
            "engagement_metrics": {
                "session_duration": 45.2,
                "page_views": 8.5,
                "interactions": 12.3,
                "return_visits": 3.2
            },
            "funnel_data": [
                {"stage": "Landing", "users": 1000},
                {"stage": "Signup", "users": 250},
                {"stage": "Activation", "users": 180},
                {"stage": "Purchase", "users": 45}
            ],
            "user_cohorts": [
                {
                    "cohort": "new_users",
                    "days_since_last_login": 3,
                    "sessions_per_week": 5,
                    "features_used": 8,
                    "support_tickets_last_month": 0
                },
                {
                    "cohort": "at_risk_users", 
                    "days_since_last_login": 14,
                    "sessions_per_week": 1,
                    "features_used": 2,
                    "support_tickets_last_month": 3
                }
            ]
        }
        
        # Execute comprehensive analysis
        logger.info("Executing comprehensive analysis...")
        results = await workflow.execute_comprehensive_analysis(analysis_request)
        
        # Display results
        logger.info("Analysis Results:")
        logger.info("================")
        logger.info("Analysis ID: %s", results.get("analysis_id"))
        logger.info("Insights Generated: %d", len(results.get("insights", {}).get("key_insights", [])))
        logger.info("Recommendations: %d", len(results.get("recommendations", [])))
        logger.info("Next Steps: %d", len(results.get("next_steps", [])))
        
        # Show sample insights
        insights = results.get("insights", {})
        if insights.get("key_insights"):
            logger.info("\nKey Insights:")
            for insight in insights["key_insights"][:3]:  # Show first 3
                logger.info("- %s", insight)
        
        if insights.get("recommendations"):
            logger.info("\nTop Recommendations:")
            for rec in insights["recommendations"][:3]:  # Show first 3
                logger.info("- %s", rec)
        
        logger.info("\nComprehensive collaborative AI platform demo completed successfully!")
        
    except Exception as e:
        logger.error("Demo failed: %s", str(e))
        raise
    
    finally:
        # Cleanup
        await workflow.shutdown()


if __name__ == "__main__":
    # Run the comprehensive demo
    asyncio.run(main())
