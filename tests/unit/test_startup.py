"""
Unit tests for the startup script.
"""
import pytest
import sys
import os
from unittest.mock import Mock, patch, call
from argparse import Namespace
from io import StringIO

# Add the project root to the path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import start


class TestStartupScript:
    """Test cases for the startup script."""
    
    def test_parse_arguments_default(self):
        """Test parsing arguments with defaults."""
        with patch('sys.argv', ['start.py']):
            args = start.parse_arguments()
            
            # Check default values based on actual implementation
            assert hasattr(args, 'env')
            assert hasattr(args, 'aws_region')
    
    def test_parse_arguments_custom(self):
        """Test parsing arguments with custom values."""
        with patch('sys.argv', [
            'start.py',
            '--env', 'DEV',
            '--aws-region', 'us-west-2',
            '--aws-access-key-id', 'test_key',
            '--aws-secret-access-key', 'test_secret'
        ]):
            args = start.parse_arguments()
            
            assert args.env == 'DEV'
            assert args.aws_region == 'us-west-2'
            assert args.aws_access_key_id == 'test_key'
            assert args.aws_secret_access_key == 'test_secret'
    
    def test_setup_environment_variables(self):
        """Test setting up environment variables from args."""
        args = Mock()
        args.env = 'DEV'
        args.debug = True
        args.log_level = 'DEBUG'
        args.host = '127.0.0.1'
        args.port = 8080
        args.aws_region = 'us-west-2'
        args.aws_access_key_id = 'test_key'
        args.aws_secret_access_key = 'test_secret'
        args.aws_profile = None
        args.openai_api_key = None
        args.anthropic_api_key = None
        args.llm_provider = None
        args.workflow_request_sqs_url = None
        args.workflow_response_sqs_url = None
        
        with patch.dict('os.environ', {}, clear=True):
            start.setup_environment_variables(args)
            
            assert os.environ['APP_ENV'] == 'DEV'
            assert os.environ['DEBUG'] == 'true'
            assert os.environ['LOG_LEVEL'] == 'DEBUG'
            assert os.environ['API_HOST'] == '127.0.0.1'
            assert os.environ['API_PORT'] == '8080'
            assert os.environ['AWS_REGION'] == 'us-west-2'
            assert os.environ['AWS_ACCESS_KEY_ID'] == 'test_key'
            assert os.environ['AWS_SECRET_ACCESS_KEY'] == 'test_secret'
    
    def test_validate_configuration_success(self):
        """Test configuration validation with valid setup."""
        with patch.dict('os.environ', {
            'AWS_ACCESS_KEY_ID': 'test_key',
            'AWS_SECRET_ACCESS_KEY': 'test_secret',
            'AWS_REGION': 'us-east-1',
            'OPENAI_API_KEY': 'test_openai_key',
            'DEFAULT_LLM_PROVIDER': 'openai'
        }):
            errors, warnings = start.validate_configuration()
            
            assert len(errors) == 0
            assert isinstance(warnings, list)
    
    def test_validate_configuration_warnings(self):
        """Test configuration validation with warnings."""
        with patch.dict('os.environ', {
            'AWS_ACCESS_KEY_ID': 'test_key',
            'AWS_SECRET_ACCESS_KEY': 'test_secret',
            'DEFAULT_LLM_PROVIDER': 'openai'
            # Missing AWS_REGION and OPENAI_API_KEY
        }, clear=True):
            errors, warnings = start.validate_configuration()
            
            assert len(errors) == 0
            assert len(warnings) >= 1  # Should have warnings about missing config
    
    @patch('sys.stdout', new_callable=StringIO)
    def test_print_configuration_summary(self, mock_stdout):
        """Test printing configuration summary."""
        with patch.dict('os.environ', {
            'APP_ENV': 'test',
            'AWS_REGION': 'us-east-1'
        }):
            start.print_configuration_summary()
            
            output = mock_stdout.getvalue()
            assert 'EDGP AI Model - Configuration Summary' in output
            assert '=' in output  # Should have separator lines
    
    @patch('subprocess.run')
    def test_main_execution_with_mock(self, mock_subprocess):
        """Test main function execution."""
        with patch('start.parse_arguments') as mock_parse:
            with patch('start.setup_environment_variables') as mock_setup:
                with patch('start.validate_configuration') as mock_validate:
                    with patch('start.print_configuration_summary') as mock_print:
                        with patch('uvicorn.run') as mock_uvicorn:
                            with patch('start.main', wraps=start.main) as mock_main:
                                # Setup mocks
                                mock_args = Mock()
                                mock_parse.return_value = mock_args
                                mock_validate.return_value = ([], [])  # no errors, no warnings
                                mock_subprocess.return_value.returncode = 0
                                
                                # Mock the uvicorn import and execution
                                with patch.dict('os.environ', {
                                    'API_HOST': '0.0.0.0',
                                    'API_PORT': '8000',
                                    'LOG_LEVEL': 'info'
                                }):
                                    try:
                                        start.main()
                                    except SystemExit:
                                        pass  # Expected for successful execution
                                
                                # Verify function calls
                                mock_parse.assert_called_once()
                                mock_setup.assert_called_once_with(mock_args)
                                mock_validate.assert_called_once()
                                mock_print.assert_called_once()
    
    def test_environment_variable_setup_with_none_values(self):
        """Test environment variable setup with None values."""
        args = Mock()
        args.env = None
        args.debug = None
        args.log_level = None
        args.host = None
        args.port = None
        args.aws_region = None
        args.aws_access_key_id = None
        args.aws_secret_access_key = None
        args.aws_profile = None
        args.openai_api_key = None
        args.anthropic_api_key = None
        args.llm_provider = None
        args.workflow_request_sqs_url = None
        args.workflow_response_sqs_url = None
        
        # Should not raise any exceptions
        start.setup_environment_variables(args)
    
    def test_configuration_validation_empty_environment(self):
        """Test configuration validation with empty environment."""
        with patch.dict('os.environ', {}, clear=True):
            errors, warnings = start.validate_configuration()
            
            assert isinstance(errors, list)
            assert isinstance(warnings, list)
