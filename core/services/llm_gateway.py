"""
Simplified LLM Gateway for development/testing without external dependencies.
"""

from typing import Dict, List, Any, Optional, AsyncGenerator
import asyncio
import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class LLMResponse:
    """Simplified LLM response object."""
    def __init__(self, content: str, metadata: Optional[Dict[str, Any]] = None):
        self.content = content
        self.metadata = metadata or {}


class SimpleLLMProvider:
    """Mock LLM provider for development/testing."""
    
    def __init__(self, provider_name: str = "mock"):
        self.provider_name = provider_name
        self.is_available = True
    
    async def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate a mock response."""
        # Simulate processing time
        await asyncio.sleep(0.1)
        
        # Generate mock response based on prompt content
        if "policy" in prompt.lower():
            content = self._generate_policy_response()
        elif "quality" in prompt.lower():
            content = self._generate_quality_response()
        elif "compliance" in prompt.lower():
            content = self._generate_compliance_response()
        elif "remediation" in prompt.lower():
            content = self._generate_remediation_response()
        elif "analytics" in prompt.lower():
            content = self._generate_analytics_response()
        else:
            content = "This is a mock response from the simplified LLM gateway."
        
        return LLMResponse(
            content=content,
            metadata={
                "provider": self.provider_name,
                "timestamp": datetime.utcnow().isoformat(),
                "mock": True
            }
        )
    
    async def stream_generate(self, prompt: str, **kwargs) -> AsyncGenerator[str, None]:
        """Generate streaming mock response."""
        content = await self.generate(prompt, **kwargs)
        words = content.content.split()
        
        for word in words:
            await asyncio.sleep(0.01)  # Simulate streaming delay
            yield word + " "
    
    def _generate_policy_response(self) -> str:
        return """Based on the analysis, I recommend the following data governance policies:

1. **Data Validation Policy**: Implement email format validation using regex pattern
2. **Access Control Policy**: Ensure role-based access to sensitive data fields
3. **Data Retention Policy**: Set retention periods based on regulatory requirements
4. **Data Quality Policy**: Establish quality thresholds and monitoring procedures

These policies will help ensure data integrity and compliance with regulations."""

    def _generate_quality_response(self) -> str:
        return """Data Quality Assessment Results:

**Overall Quality Score**: 85%

**Key Findings**:
- Completeness: 92% (Good)
- Accuracy: 78% (Needs Improvement)
- Consistency: 89% (Good)
- Validity: 91% (Good)

**Issues Detected**:
- 156 records with invalid email formats
- 23 duplicate customer entries
- 8% missing phone numbers

**Recommendations**:
- Implement email validation rules
- Run deduplication process
- Add phone number collection workflow"""

    def _generate_compliance_response(self) -> str:
        return """Compliance Assessment Results:

**Overall Compliance Score**: 87%

**GDPR Compliance**:
- ✅ Data subject rights implemented
- ❌ Right to erasure needs improvement
- ✅ Privacy by design principles followed
- ⚠️  Data processing records incomplete

**Recommendations**:
- Implement automated data deletion workflow
- Complete data processing activity records
- Add consent management system
- Conduct privacy impact assessment"""

    def _generate_remediation_response(self) -> str:
        return """Data Remediation Recommendations:

**Priority Tasks**:
1. **High Priority**: Fix 156 invalid email records
   - Estimated effort: 4 hours
   - Automation level: High
   
2. **Medium Priority**: Resolve duplicate customers
   - Estimated effort: 8 hours
   - Automation level: Medium
   
3. **Low Priority**: Collect missing phone numbers
   - Estimated effort: 16 hours
   - Automation level: Low

**Implementation Plan**:
- Phase 1: Automated email validation and correction
- Phase 2: Semi-automated duplicate resolution
- Phase 3: Customer outreach for missing data"""

    def _generate_analytics_response(self) -> str:
        return """Analytics Report:

**Data Quality Trends** (Last 30 days):
- Quality scores improved by 12%
- Issue resolution time decreased by 25%
- Customer satisfaction increased by 8%

**Key Metrics**:
- Total records processed: 1,245,678
- Issues resolved: 3,456
- Policies implemented: 12
- Compliance score: 87%

**Insights**:
- Data quality is trending upward
- Most common issues are in email validation
- Customer onboarding process needs improvement
- Compliance gaps in data retention policies"""


class SimpleLLMGateway:
    """Simplified LLM Gateway for development/testing."""
    
    def __init__(self):
        self.providers = {
            "mock": SimpleLLMProvider("mock"),
            "bedrock": SimpleLLMProvider("bedrock-mock"),
            "openai": SimpleLLMProvider("openai-mock")
        }
        self.primary_provider = "mock"
        self.fallback_providers = ["bedrock", "openai"]
    
    async def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate response using primary provider with fallback."""
        try:
            provider = self.providers[self.primary_provider]
            return await provider.generate(prompt, **kwargs)
        except Exception as e:
            logger.warning(f"Primary provider failed: {e}. Trying fallback...")
            for fallback_name in self.fallback_providers:
                try:
                    provider = self.providers[fallback_name]
                    return await provider.generate(prompt, **kwargs)
                except Exception as fe:
                    logger.warning(f"Fallback provider {fallback_name} failed: {fe}")
            
            # If all providers fail, return error response
            return LLMResponse(
                content="I apologize, but I'm currently unable to process your request due to technical difficulties. Please try again later.",
                metadata={"error": "All LLM providers unavailable"}
            )
    
    async def stream_generate(self, prompt: str, **kwargs) -> AsyncGenerator[str, None]:
        """Generate streaming response."""
        try:
            provider = self.providers[self.primary_provider]
            async for chunk in provider.stream_generate(prompt, **kwargs):
                yield chunk
        except Exception as e:
            logger.error(f"Streaming generation failed: {e}")
            yield "Error: Unable to stream response."


# Global instance
llm_gateway = SimpleLLMGateway()
