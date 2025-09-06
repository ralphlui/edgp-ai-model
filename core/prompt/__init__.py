"""
Prompt Engineering System

A comprehensive prompt engineering framework providing:
- Standardized prompt templates and management
- Advanced prompt optimization and validation
- Template versioning and registry
- Industry-standard prompt techniques
- Performance tracking and analytics
"""

from .types import (
    PromptType,
    PromptCategory, 
    PromptTechnique,
    PromptVariable,
    PromptMetadata,
    PromptContext,
    PromptResult,
    PromptValidationResult,
    PromptOptimizationResult,
    OptimizationMetric
)

from .templates import (
    PromptTemplate,
    SystemPromptTemplate,
    AgentPromptTemplate,
    TaskPromptTemplate,
    ChainOfThoughtTemplate,
    FewShotTemplate
)

from .manager import PromptManager
from .builder import PromptBuilder, create_system_prompt, create_agent_prompt, create_task_prompt
from .validator import PromptValidator
from .optimizer import PromptOptimizer, OptimizationConfig, PerformanceData
from .registry import PromptRegistry, PromptVersion

__all__ = [
    # Types
    "PromptType",
    "PromptCategory",
    "PromptTechnique", 
    "PromptVariable",
    "PromptMetadata",
    "PromptContext",
    "PromptResult",
    "PromptValidationResult",
    "PromptOptimizationResult",
    "OptimizationMetric",
    
    # Templates
    "PromptTemplate",
    "SystemPromptTemplate",
    "AgentPromptTemplate", 
    "TaskPromptTemplate",
    "ChainOfThoughtTemplate",
    "FewShotTemplate",
    
    # Core components
    "PromptManager",
    "PromptBuilder",
    "PromptValidator",
    "PromptOptimizer",
    "PromptRegistry",
    
    # Additional classes
    "OptimizationConfig",
    "PerformanceData", 
    "PromptVersion",
    
    # Convenience functions
    "create_system_prompt",
    "create_agent_prompt", 
    "create_task_prompt"
]

from .manager import PromptManager
from .templates import (
    SystemPromptTemplate,
    AgentPromptTemplate, 
    TaskPromptTemplate,
    PromptTemplate
)
from .types import (
    PromptType,
    PromptCategory,
    PromptMetadata,
    PromptConfig,
    PromptContext,
    PromptResult
)
from .builder import PromptBuilder
from .optimizer import PromptOptimizer
from .validator import PromptValidator
from .registry import PromptRegistry

# Global prompt manager instance
prompt_manager = PromptManager()

__all__ = [
    # Core classes
    'PromptManager',
    'PromptTemplate',
    'SystemPromptTemplate',
    'AgentPromptTemplate',
    'TaskPromptTemplate',
    
    # Types and enums
    'PromptType',
    'PromptCategory', 
    'PromptMetadata',
    'PromptConfig',
    'PromptContext',
    'PromptResult',
    
    # Utilities
    'PromptBuilder',
    'PromptOptimizer',
    'PromptValidator',
    'PromptRegistry',
    
    # Global instance
    'prompt_manager'
]
