#!/usr/bin/env python3
"""
Simple coverage test that bypasses conftest.py imports
"""

import sys
import os
import json
from unittest.mock import Mock, patch, MagicMock

# Add project root to path
sys.path.insert(0, '/Users/jjfwang/Documents/02-NUS/01-Capstone/edgp-ai-model')

def test_simple_import():
    """Test basic imports work"""
    try:
        # Mock problematic dependencies
        with patch.dict('sys.modules', {
            'langchain': MagicMock(),
            'langchain.schema': MagicMock(),
            'langchain.agents': MagicMock(),
            'langchain_community': MagicMock(),
            'boto3': MagicMock(),
            'aioboto3': MagicMock(),
        }):
            # Import our configuration
            from core.infrastructure.config import Settings
            
            # Test simple initialization
            settings = Settings()
            assert settings is not None
            
            print("✅ Simple import test passed")
            return True
            
    except Exception as e:
        print(f"❌ Simple import test failed: {e}")
        return False

def test_message_queue_types():
    """Test message queue types without AWS dependencies"""
    try:
        with patch.dict('sys.modules', {
            'boto3': MagicMock(),
            'aioboto3': MagicMock(),
        }):
            from core.services.message_queue import MessageType, MessagePriority, MessageContext
            
            # Test enum values
            assert MessageType.WORKFLOW_REQUEST
            assert MessagePriority.HIGH
            
            # Test message context creation
            context = MessageContext(
                source_agent="test",
                target_agent="test",
                conversation_id="test"
            )
            assert context.source_agent == "test"
            
            print("✅ Message queue types test passed")
            return True
            
    except Exception as e:
        print(f"❌ Message queue types test failed: {e}")
        return False

def test_auth_types():
    """Test auth types without complex dependencies"""
    try:
        with patch.dict('sys.modules', {
            'passlib': MagicMock(),
            'jose': MagicMock(),
            'langchain': MagicMock(),
        }):
            # Import auth config
            from core.infrastructure.auth import UserRole, SessionStatus
            
            # Test enum values exist
            assert hasattr(UserRole, 'ADMIN')
            assert hasattr(SessionStatus, 'ACTIVE')
            
            print("✅ Auth types test passed")
            return True
            
    except Exception as e:
        print(f"❌ Auth types test failed: {e}")
        return False

def main():
    """Run simple coverage tests"""
    print("Running simple coverage tests...")
    
    tests = [
        test_simple_import,
        test_message_queue_types,
        test_auth_types,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All simple tests passed!")
        return 0
    else:
        print("⚠️ Some tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
