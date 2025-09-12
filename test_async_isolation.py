import pytest

class TestAsyncIsolation:
    @pytest.mark.asyncio
    async def test_isolated_async(self):
        """Test isolated async function."""
        assert True
