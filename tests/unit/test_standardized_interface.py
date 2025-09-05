#!/usr/bin/env python3
"""
Test script for the new standardized agent communication interface.
This script demonstrates the new agentic AI best practices implementation.
"""

import asyncio
import json
from datetime import datetime
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents.data_quality.agent import DataQualityAgent
from core.types.communication import (
    StandardAgentInput,
    DataQualityInput,
    DataPayload,
    OperationParameters,
    QualityDimension,
    AgentContext,
    create_standard_input
)


async def test_standardized_data_quality_assessment():
    """Test the standardized data quality assessment interface."""
    print("🧪 Testing Standardized Data Quality Assessment Interface")
    print("=" * 60)
    
    # Initialize agent
    agent = DataQualityAgent("test_data_quality_agent")
    
    # Create sample data with quality issues
    sample_data = {
        "customers": [
            {"id": 1, "name": "John Doe", "email": "john@example.com", "age": 30},
            {"id": 2, "name": "", "email": "invalid-email", "age": -5},  # Issues: missing name, invalid email, negative age
            {"id": 3, "name": "Jane Smith", "email": "jane@example.com", "age": 25},
            {"id": 4, "name": "Bob Wilson", "email": "bob@example.com", "age": None}  # Issue: missing age
        ]
    }
    
    # Create data payload
    data_payload = DataPayload(
        data_type="tabular",
        data_format="json",
        content=sample_data,
        source_system="test_database",
        schema_version="1.0",
        data_lineage=["raw_input", "cleaned", "validated"]
    )
    
    # Create operation parameters
    operation_params = OperationParameters(
        quality_threshold=0.8,
        include_recommendations=True,
        processing_mode="comprehensive",
        performance_tracking=True
    )
    
    # Create data quality input
    quality_input = DataQualityInput(
        data_payload=data_payload,
        operation_parameters=operation_params,
        quality_dimensions=[
            QualityDimension.COMPLETENESS,
            QualityDimension.ACCURACY,
            QualityDimension.VALIDITY,
            QualityDimension.CONSISTENCY
        ],
        include_anomaly_detection=True,
        include_profiling=True
    )
    
    # Create standardized input
    agent_input = create_standard_input(
        source_agent_id="test_client",
        target_agent_id="test_data_quality_agent",
        capability_name="data_quality_assessment",
        data=quality_input
    )
    
    print(f"📥 Input Details:")
    print(f"   Source Agent: {agent_input.source_agent_id}")
    print(f"   Target Agent: {agent_input.target_agent_id}")
    print(f"   Capability: {agent_input.capability_name}")
    print(f"   Message ID: {agent_input.message_id}")
    print(f"   Execution Mode: {agent_input.execution_mode}")
    print(f"   Quality Dimensions: {[d.value for d in quality_input.quality_dimensions]}")
    print()
    
    # Process using standardized interface
    start_time = datetime.now()
    try:
        result = await agent.process_standardized_input(agent_input)
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        print(f"✅ Processing successful!")
        print(f"   Processing Time: {processing_time:.2f}ms")
        print(f"   Result Status: {result.status}")
        print(f"   Success: {result.success}")
        print()
        
        # Display results
        if result.success and result.data:
            quality_output = result.data
            
            print(f"📊 Quality Assessment Results:")
            print(f"   Overall Quality Score: {quality_output.overall_quality_score:.3f}")
            print()
            
            print(f"📈 Dimension Scores:")
            for dimension, score in quality_output.dimension_scores.items():
                status = "✅" if score >= 0.8 else "⚠️" if score >= 0.5 else "❌"
                print(f"   {status} {dimension.value}: {score:.3f}")
            print()
            
            print(f"🔍 Processing Details:")
            processing_result = quality_output.processing_result
            print(f"   Operation Type: {processing_result.operation_type}")
            print(f"   Key Findings: {len(processing_result.key_findings)} items")
            for finding in processing_result.key_findings:
                print(f"     • {finding}")
            print()
            
            if quality_output.quality_issues:
                print(f"⚠️ Quality Issues Found ({len(quality_output.quality_issues)}):")
                for issue in quality_output.quality_issues:
                    print(f"   • {issue['dimension']}: Score {issue['score']:.2f} (threshold: {issue['threshold']}) - {issue['severity']} severity")
                print()
            
            if quality_output.improvement_recommendations:
                print(f"💡 Improvement Recommendations:")
                for rec in quality_output.improvement_recommendations:
                    print(f"   • {rec}")
                print()
            
            if quality_output.priority_actions:
                print(f"🎯 Priority Actions:")
                for action in quality_output.priority_actions:
                    print(f"   • {action['action']} (Priority: {action['priority']}, Effort: {action['estimated_effort']})")
                print()
            
            # Show tracing information
            print(f"🔗 Tracing Information:")
            print(f"   Session ID: {result.context.session_id}")
            print(f"   Correlation ID: {result.context.correlation_id}")
            print(f"   Trace ID: {result.context.trace_id}")
            print()
            
        else:
            print(f"❌ Processing failed:")
            print(f"   Error Code: {result.error_code}")
            print(f"   Error Message: {result.error_message}")
            
    except Exception as e:
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        print(f"❌ Processing failed with exception:")
        print(f"   Error: {str(e)}")
        print(f"   Time before failure: {processing_time:.2f}ms")


async def test_anomaly_detection():
    """Test the anomaly detection capability."""
    print("\n🔍 Testing Anomaly Detection Interface")
    print("=" * 60)
    
    agent = DataQualityAgent("test_anomaly_agent")
    
    # Create sample data with anomalies
    sample_data = {
        "measurements": [1.2, 1.1, 1.3, 1.0, 15.7, 1.1, 1.4, 1.2, 0.9, 1.3, 89.2, 1.1]  # 15.7 and 89.2 are anomalies
    }
    
    data_payload = DataPayload(
        data_type="timeseries",
        data_format="json",
        content=sample_data,
        source_system="sensor_data"
    )
    
    operation_params = OperationParameters(
        quality_threshold=0.95,
        include_recommendations=True
    )
    
    quality_input = DataQualityInput(
        data_payload=data_payload,
        operation_parameters=operation_params,
        include_anomaly_detection=True,
        include_profiling=False
    )
    
    agent_input = create_standard_input(
        source_agent_id="test_client",
        target_agent_id="test_anomaly_agent",
        capability_name="anomaly_detection",
        data=quality_input
    )
    
    print(f"📥 Processing anomaly detection for {len(sample_data['measurements'])} data points")
    
    try:
        result = await agent.process_standardized_input(agent_input)
        
        if result.success and result.data:
            quality_output = result.data
            print(f"✅ Anomaly detection completed")
            print(f"   Quality Score: {quality_output.overall_quality_score:.3f}")
            print(f"   Anomalies Detected: {len(quality_output.anomalies_detected)}")
            
            if quality_output.anomalies_detected:
                print(f"🚨 Detected Anomalies:")
                for anomaly in quality_output.anomalies_detected:
                    print(f"   • {anomaly}")
            
        else:
            print(f"❌ Anomaly detection failed")
            
    except Exception as e:
        print(f"❌ Exception during anomaly detection: {str(e)}")


async def test_data_profiling():
    """Test the data profiling capability."""
    print("\n📊 Testing Data Profiling Interface")
    print("=" * 60)
    
    agent = DataQualityAgent("test_profiling_agent")
    
    # Create sample data for profiling
    sample_data = {
        "transactions": [
            {"amount": 100.50, "currency": "USD", "date": "2024-01-15", "type": "purchase"},
            {"amount": 75.25, "currency": "USD", "date": "2024-01-16", "type": "refund"},
            {"amount": 200.00, "currency": "EUR", "date": "2024-01-17", "type": "purchase"},
            {"amount": 50.75, "currency": "USD", "date": "2024-01-18", "type": "purchase"}
        ]
    }
    
    data_payload = DataPayload(
        data_type="tabular",
        data_format="json",
        content=sample_data,
        source_system="payment_system"
    )
    
    operation_params = OperationParameters(
        quality_threshold=0.8,
        include_recommendations=True
    )
    
    quality_input = DataQualityInput(
        data_payload=data_payload,
        operation_parameters=operation_params,
        include_anomaly_detection=False,
        include_profiling=True
    )
    
    agent_input = create_standard_input(
        source_agent_id="test_client",
        target_agent_id="test_profiling_agent",
        capability_name="data_profiling",
        data=quality_input
    )
    
    print(f"📥 Profiling dataset with {len(sample_data['transactions'])} records")
    
    try:
        result = await agent.process_standardized_input(agent_input)
        
        if result.success and result.data:
            quality_output = result.data
            print(f"✅ Data profiling completed")
            
            if quality_output.data_profile:
                print(f"📈 Data Profile:")
                for key, value in quality_output.data_profile.items():
                    print(f"   • {key}: {value}")
            
        else:
            print(f"❌ Data profiling failed")
            
    except Exception as e:
        print(f"❌ Exception during data profiling: {str(e)}")


async def main():
    """Run all standardized interface tests."""
    print("🚀 EDGP AI Model - Standardized Interface Testing")
    print("=" * 60)
    print("Testing the new agentic AI best practices implementation")
    print("with comprehensive type safety and traceability.")
    print()
    
    try:
        # Run all tests
        await test_standardized_data_quality_assessment()
        await test_anomaly_detection()
        await test_data_profiling()
        
        print("\n🎉 All tests completed!")
        print("The standardized communication interface is working correctly.")
        
    except Exception as e:
        print(f"\n💥 Test suite failed with exception: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
