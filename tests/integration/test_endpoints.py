"""
Integration tests for agent endpoints.
"""

import pytest
from httpx import AsyncClient


class TestAgentEndpoints:
    """Test agent endpoint functionality."""
    
    @pytest.mark.asyncio
    async def test_health_endpoint(self, async_client: AsyncClient):
        """Test health check endpoint."""
        response = await async_client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert "agents_initialized" in data
        assert "llm_gateway_status" in data
    
    @pytest.mark.asyncio
    async def test_list_agents_endpoint(self, async_client: AsyncClient):
        """Test list agents endpoint."""
        response = await async_client.get("/agents")
        assert response.status_code in [200, 503]  # 503 if agents not initialized
        
        if response.status_code == 200:
            data = response.json()
            assert "agents" in data
            assert isinstance(data["agents"], list)
    
    @pytest.mark.asyncio
    async def test_policy_suggestion_endpoint(
        self, 
        async_client: AsyncClient,
        sample_policy_request: dict
    ):
        """Test policy suggestion endpoint."""
        response = await async_client.post(
            "/agents/policy-suggestion/suggest-policies",
            json=sample_policy_request
        )
        
        # Should return 200 or 503 depending on agent initialization
        assert response.status_code in [200, 503]
        
        if response.status_code == 200:
            data = response.json()
            assert "request_id" in data
            assert "agent_type" in data
            assert "success" in data
    
    @pytest.mark.asyncio
    async def test_data_quality_endpoint(
        self,
        async_client: AsyncClient,
        sample_quality_request: dict
    ):
        """Test data quality endpoint."""
        response = await async_client.post(
            "/agents/data-quality/detect-anomalies",
            json=sample_quality_request
        )
        
        assert response.status_code in [200, 503]
        
        if response.status_code == 200:
            data = response.json()
            assert "request_id" in data
            assert "overall_quality_score" in data
            assert "quality_metrics" in data
    
    @pytest.mark.asyncio
    async def test_compliance_endpoint(
        self,
        async_client: AsyncClient,
        sample_compliance_request: dict
    ):
        """Test compliance endpoint."""
        response = await async_client.post(
            "/agents/data-privacy-compliance/scan-risks",
            json=sample_compliance_request
        )
        
        assert response.status_code in [200, 503, 422]  # 422 for validation errors
        
        if response.status_code == 200:
            data = response.json()
            assert "request_id" in data
            assert "compliance_score" in data
            assert "privacy_risks" in data
    
    @pytest.mark.asyncio
    async def test_invalid_request_validation(self, async_client: AsyncClient):
        """Test request validation with invalid data."""
        invalid_request = {
            "invalid_field": "invalid_value"
        }
        
        response = await async_client.post(
            "/agents/policy-suggestion/suggest-policies",
            json=invalid_request
        )
        
        assert response.status_code == 422  # Validation error
        
        data = response.json()
        assert "detail" in data
    
    @pytest.mark.asyncio
    async def test_root_endpoint(self, async_client: AsyncClient):
        """Test root endpoint."""
        response = await async_client.get("/")
        assert response.status_code == 200
        
        data = response.json()
        assert "service" in data
        assert "version" in data
        assert "status" in data
