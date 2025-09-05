#!/usr/bin/env python3
"""
Simple test to debug the standardized interface.
"""

import asyncio
from agents.data_quality.agent import DataQualityAgent
from core.types.communication import (
    StandardAgentInput,
    DataQualityInput, 
    DataPayload,
    OperationParameters,
    QualityDimension,
    create_standard_input
)


async def simple_test():
    print("🧪 Simple Standardized Interface Test")
    print("=" * 40)
    
    # Create agent
    print("Creating agent...")
    agent = DataQualityAgent("test_agent")
    print(f"Agent created: {agent.agent_id}")
    
    # Create simple input
    print("Creating input...")
    data_payload = DataPayload(
        data_type="tabular",
        data_format="json", 
        content={"test": "data"},
        source_system="test"
    )
    
    operation_params = OperationParameters(
        quality_threshold=0.8,
        include_recommendations=True
    )
    
    quality_input = DataQualityInput(
        data_payload=data_payload,
        operation_parameters=operation_params,
        quality_dimensions=[QualityDimension.COMPLETENESS]
    )
    
    agent_input = create_standard_input(
        source_agent_id="test_client",
        target_agent_id="test_agent", 
        capability_name="data_quality_assessment",
        data=quality_input
    )
    
    print(f"Input created with message ID: {agent_input.message_id}")
    
    # Process
    print("Processing...")
    try:
        result = await agent.process_standardized_input(agent_input)
        print(f"✅ Success: {result.success}")
        print(f"Status: {result.status}")
        print(f"Result type: {type(result)}")
        print(f"Result fields: {list(result.__dict__.keys())}")
        
        # Check for different field names
        if hasattr(result, 'result') and result.result:
            print(f"Quality Score: {result.result.overall_quality_score}")
        elif hasattr(result, 'data') and result.data:
            print(f"Quality Score: {result.data.overall_quality_score}")
        else:
            print("No data/result field found")
            
        if hasattr(result, 'error_message') and result.error_message:
            print(f"Error message: {result.error_message}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(simple_test())
