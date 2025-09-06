"""
Prompt Engineering Types and Enums

Defines the core types, enums, and data structures for the prompt management system
following industry standards for prompt engineering.
"""

from enum import Enum, auto
from typing import Dict, List, Optional, Any, Union, Callable
from datetime import datetime
from dataclasses import dataclass, field
from pydantic import BaseModel, Field, validator
import uuid


class PromptType(str, Enum):
    """Types of prompts in the system."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    FUNCTION = "function"
    TOOL = "tool"


class PromptCategory(str, Enum):
    """Categories of prompts for organization."""
    SYSTEM_CORE = "system_core"
    AGENT_SPECIFIC = "agent_specific"
    TASK_SPECIFIC = "task_specific"
    CONTEXT_SPECIFIC = "context_specific"
    CHAIN_OF_THOUGHT = "chain_of_thought"
    FEW_SHOT = "few_shot"
    ZERO_SHOT = "zero_shot"
    INSTRUCTION_FOLLOWING = "instruction_following"
    REASONING = "reasoning"
    ANALYSIS = "analysis"
    GENERATION = "generation"
    CLASSIFICATION = "classification"
    EXTRACTION = "extraction"
    SUMMARIZATION = "summarization"
    EVALUATION = "evaluation"


class PromptPriority(str, Enum):
    """Priority levels for prompt selection."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class PromptStatus(str, Enum):
    """Status of prompt templates."""
    ACTIVE = "active"
    DRAFT = "draft"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"
    TESTING = "testing"


class PromptTechnique(str, Enum):
    """Prompt engineering techniques."""
    ZERO_SHOT = "zero_shot"
    FEW_SHOT = "few_shot"
    CHAIN_OF_THOUGHT = "chain_of_thought"
    TREE_OF_THOUGHT = "tree_of_thought"
    INSTRUCTION_FOLLOWING = "instruction_following"
    ROLE_PLAYING = "role_playing"
    STEP_BY_STEP = "step_by_step"
    TEMPLATE_BASED = "template_based"
    CONTEXT_INJECTION = "context_injection"
    CONSTRAINT_GUIDANCE = "constraint_guidance"


class OptimizationMetric(str, Enum):
    """Metrics for prompt optimization."""
    ACCURACY = "accuracy"
    RELEVANCE = "relevance"
    CLARITY = "clarity"
    COMPLETENESS = "completeness"
    CONSISTENCY = "consistency"
    EFFICIENCY = "efficiency"
    TOKEN_COUNT = "token_count"
    RESPONSE_TIME = "response_time"
    USER_SATISFACTION = "user_satisfaction"


@dataclass
class PromptMetrics:
    """Performance metrics for prompt evaluation."""
    accuracy_score: float = 0.0
    relevance_score: float = 0.0
    clarity_score: float = 0.0
    completeness_score: float = 0.0
    consistency_score: float = 0.0
    efficiency_score: float = 0.0
    average_token_count: int = 0
    average_response_time_ms: float = 0.0
    success_rate: float = 0.0
    error_rate: float = 0.0
    usage_count: int = 0
    last_updated: datetime = field(default_factory=datetime.utcnow)


class PromptMetadata(BaseModel):
    """Metadata for prompt templates."""
    
    # Identification
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: str
    version: str = "1.0.0"
    
    # Classification
    category: PromptCategory
    technique: PromptTechnique
    priority: PromptPriority = PromptPriority.MEDIUM
    status: PromptStatus = PromptStatus.DRAFT
    
    # Authoring
    author: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Versioning
    parent_id: Optional[str] = None
    version_history: List[str] = Field(default_factory=list)
    
    # Tags and keywords
    tags: List[str] = Field(default_factory=list)
    keywords: List[str] = Field(default_factory=list)
    
    # Usage context
    target_models: List[str] = Field(default_factory=list)
    supported_languages: List[str] = Field(default=["en"])
    
    # Performance
    metrics: Optional[PromptMetrics] = None
    
    # Experimental
    a_b_test_group: Optional[str] = None
    experiment_id: Optional[str] = None
    
    @validator('version')
    def validate_version(cls, v):
        """Validate semantic version format."""
        parts = v.split('.')
        if len(parts) != 3:
            raise ValueError("Version must be in format 'major.minor.patch'")
        for part in parts:
            if not part.isdigit():
                raise ValueError("Version parts must be numeric")
        return v


class PromptVariable(BaseModel):
    """Variable definition for prompt templates."""
    
    name: str
    type: str  # string, number, boolean, array, object
    description: str
    required: bool = True
    default_value: Optional[Any] = None
    validation_pattern: Optional[str] = None
    examples: List[Any] = Field(default_factory=list)
    constraints: Dict[str, Any] = Field(default_factory=dict)


class PromptContext(BaseModel):
    """Context information for prompt generation."""
    
    # Agent context
    agent_id: Optional[str] = None
    agent_type: Optional[str] = None
    capability: Optional[str] = None
    
    # Task context
    task_type: Optional[str] = None
    task_priority: Optional[str] = None
    task_complexity: Optional[str] = None
    
    # User context
    user_id: Optional[str] = None
    user_role: Optional[str] = None
    user_preferences: Dict[str, Any] = Field(default_factory=dict)
    
    # Session context
    session_id: Optional[str] = None
    conversation_history: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Data context
    data_type: Optional[str] = None
    data_size: Optional[str] = None
    data_sensitivity: Optional[str] = None
    
    # Environment context
    environment: str = "production"
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    locale: str = "en-US"
    
    # Custom context
    custom_variables: Dict[str, Any] = Field(default_factory=dict)


class PromptConfig(BaseModel):
    """Configuration for prompt generation and processing."""
    
    # Template selection
    auto_select_template: bool = True
    fallback_template_id: Optional[str] = None
    
    # Variable handling
    strict_variable_validation: bool = True
    allow_missing_variables: bool = False
    variable_substitution_format: str = "{{variable}}"
    
    # Optimization
    enable_optimization: bool = True
    optimization_metrics: List[OptimizationMetric] = Field(default_factory=lambda: [
        OptimizationMetric.ACCURACY,
        OptimizationMetric.RELEVANCE
    ])
    
    # Caching
    enable_caching: bool = True
    cache_ttl_seconds: int = 3600
    
    # A/B Testing
    enable_ab_testing: bool = False
    ab_test_split_ratio: float = 0.5
    
    # Logging and monitoring
    enable_logging: bool = True
    log_level: str = "INFO"
    track_metrics: bool = True
    
    # Safety and compliance
    enable_content_filtering: bool = True
    max_prompt_length: int = 8192
    allowed_prompt_types: List[PromptType] = Field(default_factory=lambda: [
        PromptType.SYSTEM,
        PromptType.USER,
        PromptType.ASSISTANT
    ])


class PromptResult(BaseModel):
    """Result of prompt generation or processing."""
    
    # Generated content
    prompt: str
    formatted_prompt: str
    
    # Metadata
    template_id: str
    template_version: str
    generation_timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # Variables used
    variables_used: Dict[str, Any] = Field(default_factory=dict)
    missing_variables: List[str] = Field(default_factory=list)
    
    # Performance metrics
    generation_time_ms: float = 0.0
    token_count: int = 0
    estimated_cost: float = 0.0
    
    # Quality indicators
    confidence_score: float = 1.0
    quality_indicators: Dict[str, float] = Field(default_factory=dict)
    
    # A/B testing
    ab_test_variant: Optional[str] = None
    
    # Warnings and errors
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    
    # Success flag
    success: bool = True


class PromptOptimizationRequest(BaseModel):
    """Request for prompt optimization."""
    
    template_id: str
    optimization_goals: List[OptimizationMetric]
    target_metrics: Dict[OptimizationMetric, float] = Field(default_factory=dict)
    constraints: Dict[str, Any] = Field(default_factory=dict)
    test_dataset: Optional[List[Dict[str, Any]]] = None
    max_iterations: int = 10
    optimization_timeout_seconds: int = 300


class PromptValidationResult(BaseModel):
    """Result of prompt validation."""
    
    is_valid: bool
    validation_score: float = 0.0
    issues: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    suggestions: List[str] = Field(default_factory=list)
    
    # Specific validation checks
    syntax_valid: bool = True
    variable_valid: bool = True
    length_valid: bool = True
    content_safe: bool = True
    technique_appropriate: bool = True
    
    validation_timestamp: datetime = Field(default_factory=datetime.utcnow)


# Type aliases for common use cases
PromptVariables = Dict[str, Any]
PromptFormatter = Callable[[str, PromptVariables], str]
PromptValidator = Callable[[str], PromptValidationResult]
PromptOptimizer = Callable[[str, PromptOptimizationRequest], str]
