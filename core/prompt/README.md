# Prompt Engineering System

A comprehensive, industry-standard prompt engineering framework for managing, optimizing, and scaling prompt templates across AI applications.

## Overview

The EDGP AI Model's prompt engineering system provides enterprise-grade capabilities for:

- **Template Management**: Standardized prompt templates with variable substitution
- **Optimization**: Automated prompt optimization based on performance metrics
- **Validation**: Comprehensive validation for safety, quality, and best practices
- **Registry**: Centralized template registry with versioning and lifecycle management
- **Analytics**: Performance tracking and usage analytics
- **Industry Standards**: Implementation of proven prompt engineering techniques

## Core Components

### 1. Prompt Templates (`templates.py`)

Standardized template classes supporting various prompt engineering techniques:

```python
from core.prompt import SystemPromptTemplate, AgentPromptTemplate, ChainOfThoughtTemplate

# System prompt for AI behavior definition
system_prompt = SystemPromptTemplate(
    template_id="data_analyst_system",
    template="You are an expert data analyst. {{behavior_guidelines}}",
    variables=[
        PromptVariable(
            name="behavior_guidelines",
            type="string", 
            description="Specific behavioral guidelines for the analyst"
        )
    ]
)

# Agent-specific task prompt
agent_prompt = AgentPromptTemplate(
    template_id="data_quality_analysis",
    template="Analyze the data quality of {{dataset_name}} focusing on {{quality_dimensions}}",
    agent_type="DataQualityAgent"
)

# Chain-of-thought reasoning prompt
cot_prompt = ChainOfThoughtTemplate(
    template_id="step_by_step_analysis",
    template="Let's analyze this step by step:\n{{reasoning_steps}}",
    reasoning_framework="analytical"
)
```

### 2. Prompt Manager (`manager.py`)

Central management system with caching, A/B testing, and performance tracking:

```python
from core.prompt import PromptManager

manager = PromptManager()

# Register templates
manager.register_template(system_prompt)

# Render prompts with caching
rendered = manager.render_template(
    "data_analyst_system",
    {"behavior_guidelines": "Be precise and data-driven"}
)

# A/B testing
manager.create_ab_test(
    "prompt_optimization_test",
    template_a="original_prompt",
    template_b="optimized_prompt",
    traffic_split=0.5
)

# Performance tracking
manager.track_performance(
    template_id="data_analyst_system",
    metrics={
        "accuracy": 0.92,
        "response_time": 1.5,
        "user_satisfaction": 0.88
    }
)
```

### 3. Prompt Builder (`builder.py`)

Fluent interface for constructing complex prompts:

```python
from core.prompt import PromptBuilder

# Fluent prompt construction
prompt = (PromptBuilder()
    .system_role("You are a data privacy expert")
    .add_context("compliance_requirements", "GDPR and CCPA regulations")
    .add_instructions([
        "Analyze the data handling practices",
        "Identify potential compliance issues",
        "Suggest remediation steps"
    ])
    .add_examples([
        {"input": "User email collection", "output": "Ensure explicit consent..."},
        {"input": "Data retention policy", "output": "Implement automated deletion..."}
    ])
    .set_output_format("structured_json")
    .build()
)
```

### 4. Prompt Validator (`validator.py`)

Comprehensive validation for safety, quality, and compliance:

```python
from core.prompt import PromptValidator

validator = PromptValidator()

# Validate template
result = validator.validate_template(system_prompt)

if not result.is_valid:
    print(f"Validation issues: {result.issues}")
    print(f"Warnings: {result.warnings}")
    print(f"Suggestions: {result.suggestions}")

# Validate rendered prompt text
text_result = validator.validate_prompt_text(rendered_prompt)
print(f"Content safety: {text_result.content_safe}")
print(f"Validation score: {text_result.validation_score}")
```

### 5. Prompt Optimizer (`optimizer.py`)

Automated optimization based on performance data and best practices:

```python
from core.prompt import PromptOptimizer, PerformanceData

optimizer = PromptOptimizer()

# Performance data
performance = PerformanceData(
    accuracy=0.85,
    response_time=2.1,
    user_satisfaction=0.79
)

# Optimize template
result = optimizer.optimize_template(
    template=system_prompt,
    performance_data=performance
)

print(f"Optimization score: {result.optimization_score}")
print(f"Improvements: {result.improvements}")

# Get optimization suggestions
suggestions = optimizer.suggest_optimizations(system_prompt, performance)
for suggestion in suggestions:
    print(f"{suggestion['type']}: {suggestion['description']}")
```

### 6. Prompt Registry (`registry.py`)

Centralized registry with versioning and lifecycle management:

```python
from core.prompt import PromptRegistry

registry = PromptRegistry()

# Register with versioning
template_id = registry.register_template(
    template=system_prompt,
    version="1.0.0",
    created_by="system",
    changelog="Initial version"
)

# Search templates
results = registry.search_templates(
    query="data analysis",
    category=PromptCategory.SYSTEM_PROMPT,
    technique=PromptTechnique.ROLE_PLAYING
)

# Version management
versions = registry.list_versions(template_id)
for version, prompt_version in versions:
    print(f"Version {version}: {prompt_version.changelog}")

# Export/import
registry.export_templates("prompts_backup.json")
registry.import_templates("shared_prompts.json")
```

## Prompt Engineering Techniques

The system supports industry-standard prompt engineering techniques:

### 1. Few-Shot Learning
```python
few_shot = FewShotTemplate(
    template_id="classification_few_shot",
    template="Classify the following text:\n{{examples}}\n\nText: {{input_text}}",
    examples=[
        {"input": "Great product!", "output": "Positive"},
        {"input": "Poor quality", "output": "Negative"},
        {"input": "Average experience", "output": "Neutral"}
    ]
)
```

### 2. Chain-of-Thought Reasoning
```python
cot_template = ChainOfThoughtTemplate(
    template_id="mathematical_reasoning",
    template="Solve this step by step:\n\n{{problem}}\n\nStep 1: {{step1}}\nStep 2: {{step2}}\nConclusion: {{conclusion}}",
    reasoning_framework="mathematical"
)
```

### 3. Role-Based Prompting
```python
role_prompt = AgentPromptTemplate(
    template_id="expert_consultant",
    template="As a {{expert_role}} with {{years_experience}} years of experience, {{task_description}}",
    agent_type="ConsultantAgent"
)
```

### 4. Structured Output
```python
structured_prompt = TaskPromptTemplate(
    template_id="structured_analysis",
    template="Analyze the following and provide output in JSON format:\n{{input_data}}\n\nRequired format:\n{{output_schema}}",
    expected_output_format="json"
)
```

## Type System

Comprehensive type definitions ensure type safety and clear interfaces:

```python
from core.prompt import (
    PromptType, PromptCategory, PromptTechnique,
    PromptVariable, PromptMetadata, PromptContext
)

# Enums for categorization
category = PromptCategory.AGENT_SPECIFIC
technique = PromptTechnique.CHAIN_OF_THOUGHT

# Variable definition with constraints
variable = PromptVariable(
    name="max_tokens",
    type="number",
    description="Maximum tokens for response",
    constraints={"min": 1, "max": 4096},
    default_value=1000
)

# Rich metadata
metadata = PromptMetadata(
    name="Data Quality Analyzer",
    description="Analyzes data quality metrics and issues",
    category=PromptCategory.AGENT_SPECIFIC,
    technique=PromptTechnique.STRUCTURED_OUTPUT,
    tags=["data", "quality", "analysis"],
    author="EDGP AI Team",
    version="1.2.0"
)
```

## Integration with Agents

### BaseAgent Integration

The prompt system integrates seamlessly with the standardized agent communication:

```python
from core.agents.base import BaseAgent
from core.prompt import PromptManager

class DataQualityAgent(BaseAgent):
    def __init__(self):
        super().__init__(agent_id="data_quality_agent")
        self.prompt_manager = PromptManager()
        
        # Load agent-specific prompts
        self._load_prompts()
    
    def _load_prompts(self):
        """Load and register agent-specific prompts."""
        system_prompt = SystemPromptTemplate(
            template_id="dq_system_prompt",
            template="You are a data quality expert. {{guidelines}}",
            variables=[
                PromptVariable(
                    name="guidelines",
                    type="string",
                    description="Specific guidelines for data quality analysis"
                )
            ]
        )
        
        self.prompt_manager.register_template(system_prompt)
    
    def process_request(self, request: StandardAgentInput) -> StandardAgentOutput:
        """Process request using prompt templates."""
        # Render appropriate prompt
        prompt = self.prompt_manager.render_template(
            "dq_system_prompt",
            {"guidelines": "Focus on completeness, accuracy, and consistency"}
        )
        
        # Process with LLM using rendered prompt
        response = self._call_llm(prompt, request.data)
        
        return StandardAgentOutput(
            success=True,
            data=response,
            metadata={"prompt_template": "dq_system_prompt"}
        )
```

## Configuration and Customization

### Configuration Options

```python
# Prompt Manager Configuration
manager_config = {
    "cache_enabled": True,
    "cache_ttl": 3600,
    "ab_testing_enabled": True,
    "performance_tracking": True,
    "template_validation": True
}

manager = PromptManager(config=manager_config)

# Validator Configuration
validator_config = {
    "max_length": 8192,
    "min_length": 10,
    "content_safety_enabled": True,
    "technique_validation": True
}

validator = PromptValidator(config=validator_config)

# Optimizer Configuration
optimizer_config = OptimizationConfig(
    target_metrics=[OptimizationMetric.ACCURACY, OptimizationMetric.CLARITY],
    max_iterations=10,
    optimization_techniques=["length_optimization", "clarity_enhancement"]
)

optimizer = PromptOptimizer(optimizer_config)
```

## Best Practices

### 1. Template Organization
- Use clear, descriptive template IDs
- Organize by category and agent type
- Include comprehensive metadata
- Version templates semantically

### 2. Variable Design
- Use descriptive variable names
- Include detailed descriptions
- Set appropriate constraints
- Provide sensible defaults

### 3. Performance Optimization
- Monitor template performance
- Use A/B testing for optimization
- Implement caching for frequently used templates
- Regular cleanup of unused templates

### 4. Safety and Compliance
- Validate all templates before deployment
- Monitor for content safety issues
- Implement approval workflows for sensitive prompts
- Regular audits of prompt usage

## Usage Examples

### Complete Data Quality Analysis Workflow

```python
from core.prompt import *

# 1. Create prompt manager
manager = PromptManager()

# 2. Define and register templates
system_template = SystemPromptTemplate(
    template_id="dq_system",
    template="You are a data quality expert specializing in {{domain}}. {{guidelines}}",
    variables=[
        PromptVariable("domain", "string", "Data domain (e.g., customer, financial)"),
        PromptVariable("guidelines", "string", "Specific analysis guidelines")
    ]
)

analysis_template = TaskPromptTemplate(
    template_id="dq_analysis",
    template="""Analyze the data quality of the following dataset:

Dataset: {{dataset_name}}
Schema: {{schema_info}}
Sample Data: {{sample_data}}

Please evaluate:
1. Completeness: {{completeness_criteria}}
2. Accuracy: {{accuracy_criteria}}
3. Consistency: {{consistency_criteria}}

Provide results in JSON format:
{{output_format}}""",
    expected_output_format="json"
)

manager.register_template(system_template)
manager.register_template(analysis_template)

# 3. Create optimization and validation pipeline
validator = PromptValidator()
optimizer = PromptOptimizer()

# Validate templates
validation_result = validator.validate_template(analysis_template)
if not validation_result.is_valid:
    print(f"Template issues: {validation_result.issues}")

# 4. Use in agent processing
context = {
    "domain": "customer data",
    "guidelines": "Focus on PII compliance and data freshness",
    "dataset_name": "customer_profiles",
    "schema_info": "id, name, email, created_at, updated_at",
    "sample_data": "1,John Doe,john@example.com,2024-01-01,2024-01-15",
    "completeness_criteria": "No null values in required fields",
    "accuracy_criteria": "Valid email formats and date ranges", 
    "consistency_criteria": "Consistent naming conventions",
    "output_format": '{"completeness": 0.95, "accuracy": 0.88, "consistency": 0.92}'
}

# Render prompts
system_prompt = manager.render_template("dq_system", context)
analysis_prompt = manager.render_template("dq_analysis", context)

print("System Prompt:", system_prompt)
print("Analysis Prompt:", analysis_prompt)

# 5. Track performance and optimize
performance = PerformanceData(
    accuracy=0.89,
    response_time=1.8,
    user_satisfaction=0.91
)

manager.track_performance("dq_analysis", performance.to_dict())

# Get optimization suggestions
suggestions = optimizer.suggest_optimizations(analysis_template, performance)
for suggestion in suggestions:
    print(f"Optimization: {suggestion['description']}")
```

This prompt engineering system provides a comprehensive, scalable foundation for managing AI prompts across the EDGP platform, ensuring consistency, quality, and optimal performance.
