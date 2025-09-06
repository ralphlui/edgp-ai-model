"""
Prompt Template Classes

Defines the core template classes for different types of prompts,
following industry standards for prompt engineering and template management.
"""

import re
import json
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from jinja2 import Template, Environment, BaseLoader, TemplateError
import yaml

from .types import (
    PromptType,
    PromptCategory,
    PromptTechnique,
    PromptMetadata,
    PromptVariable,
    PromptContext,
    PromptResult,
    PromptValidationResult,
    PromptVariables
)


class PromptTemplate(ABC):
    """
    Abstract base class for all prompt templates.
    
    Provides core functionality for template management, variable substitution,
    and prompt generation following industry best practices.
    """
    
    def __init__(
        self,
        template: str,
        metadata: PromptMetadata,
        variables: List[PromptVariable] = None,
        parent_template: Optional['PromptTemplate'] = None
    ):
        self.template = template
        self.metadata = metadata
        self.variables = variables or []
        self.parent_template = parent_template
        
        # Initialize Jinja2 environment for advanced templating
        self.jinja_env = Environment(loader=BaseLoader())
        self.jinja_template = self.jinja_env.from_string(template)
        
        # Variable lookup for quick access
        self.variable_map = {var.name: var for var in self.variables}
        
        # Performance tracking
        self.usage_count = 0
        self.last_used = None
        self.performance_metrics = {}
    
    @abstractmethod
    def get_prompt_type(self) -> PromptType:
        """Return the type of this prompt template."""
        pass
    
    def render(
        self,
        variables: PromptVariables = None,
        context: PromptContext = None
    ) -> PromptResult:
        """
        Render the template with provided variables and context.
        
        Args:
            variables: Dictionary of variables to substitute
            context: Additional context for prompt generation
            
        Returns:
            PromptResult with generated prompt and metadata
        """
        start_time = datetime.utcnow()
        variables = variables or {}
        context = context or PromptContext()
        
        try:
            # Validate variables
            validation_result = self.validate_variables(variables)
            if not validation_result.is_valid:
                return PromptResult(
                    prompt="",
                    formatted_prompt="",
                    template_id=self.metadata.id,
                    template_version=self.metadata.version,
                    errors=validation_result.issues,
                    success=False
                )
            
            # Merge context variables
            merged_variables = self._merge_context_variables(variables, context)
            
            # Apply template inheritance if needed
            if self.parent_template:
                merged_variables = self._apply_inheritance(merged_variables)
            
            # Render with Jinja2
            formatted_prompt = self.jinja_template.render(**merged_variables)
            
            # Post-process the prompt
            final_prompt = self._post_process(formatted_prompt, context)
            
            # Calculate metrics
            generation_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            token_count = self._estimate_token_count(final_prompt)
            
            # Update usage statistics
            self.usage_count += 1
            self.last_used = datetime.utcnow()
            
            return PromptResult(
                prompt=final_prompt,
                formatted_prompt=formatted_prompt,
                template_id=self.metadata.id,
                template_version=self.metadata.version,
                variables_used=merged_variables,
                generation_time_ms=generation_time,
                token_count=token_count,
                success=True
            )
            
        except Exception as e:
            return PromptResult(
                prompt="",
                formatted_prompt="",
                template_id=self.metadata.id,
                template_version=self.metadata.version,
                errors=[str(e)],
                success=False
            )
    
    def validate_variables(self, variables: PromptVariables) -> PromptValidationResult:
        """Validate provided variables against template requirements."""
        issues = []
        warnings = []
        
        # Check required variables
        for var in self.variables:
            if var.required and var.name not in variables:
                issues.append(f"Required variable '{var.name}' is missing")
            elif var.name in variables:
                # Validate variable type and constraints
                value = variables[var.name]
                if not self._validate_variable_value(var, value):
                    issues.append(f"Variable '{var.name}' validation failed")
        
        # Check for unexpected variables
        expected_vars = {var.name for var in self.variables}
        provided_vars = set(variables.keys())
        unexpected = provided_vars - expected_vars
        
        if unexpected:
            warnings.append(f"Unexpected variables provided: {', '.join(unexpected)}")
        
        return PromptValidationResult(
            is_valid=len(issues) == 0,
            issues=issues,
            warnings=warnings,
            variable_valid=len(issues) == 0
        )
    
    def _validate_variable_value(self, variable: PromptVariable, value: Any) -> bool:
        """Validate a specific variable value."""
        try:
            # Basic type checking
            if variable.type == "string" and not isinstance(value, str):
                return False
            elif variable.type == "number" and not isinstance(value, (int, float)):
                return False
            elif variable.type == "boolean" and not isinstance(value, bool):
                return False
            elif variable.type == "array" and not isinstance(value, list):
                return False
            elif variable.type == "object" and not isinstance(value, dict):
                return False
            
            # Pattern validation for strings
            if variable.validation_pattern and isinstance(value, str):
                if not re.match(variable.validation_pattern, value):
                    return False
            
            # Constraint validation
            if variable.constraints:
                if "min_length" in variable.constraints and len(str(value)) < variable.constraints["min_length"]:
                    return False
                if "max_length" in variable.constraints and len(str(value)) > variable.constraints["max_length"]:
                    return False
                if "min_value" in variable.constraints and isinstance(value, (int, float)) and value < variable.constraints["min_value"]:
                    return False
                if "max_value" in variable.constraints and isinstance(value, (int, float)) and value > variable.constraints["max_value"]:
                    return False
            
            return True
            
        except Exception:
            return False
    
    def _merge_context_variables(self, variables: PromptVariables, context: PromptContext) -> PromptVariables:
        """Merge explicit variables with context-derived variables."""
        merged = variables.copy()
        
        # Add context variables
        if context.agent_id:
            merged["agent_id"] = context.agent_id
        if context.agent_type:
            merged["agent_type"] = context.agent_type
        if context.task_type:
            merged["task_type"] = context.task_type
        if context.user_id:
            merged["user_id"] = context.user_id
        
        # Add timestamp and locale
        merged["current_timestamp"] = context.timestamp.isoformat()
        merged["locale"] = context.locale
        
        # Add custom variables
        merged.update(context.custom_variables)
        
        return merged
    
    def _apply_inheritance(self, variables: PromptVariables) -> PromptVariables:
        """Apply template inheritance by merging parent template variables."""
        if not self.parent_template:
            return variables
        
        # Get parent template's rendered content
        parent_result = self.parent_template.render(variables)
        if parent_result.success:
            variables["parent_content"] = parent_result.prompt
        
        return variables
    
    def _post_process(self, prompt: str, context: PromptContext) -> str:
        """Post-process the rendered prompt."""
        # Remove excessive whitespace
        prompt = re.sub(r'\n\s*\n\s*\n', '\n\n', prompt)
        prompt = prompt.strip()
        
        # Apply context-specific formatting
        if context.locale != "en-US":
            # Could add localization logic here
            pass
        
        return prompt
    
    def _estimate_token_count(self, text: str) -> int:
        """Rough estimation of token count (GPT-style)."""
        # Simple heuristic: ~4 characters per token
        return len(text) // 4
    
    def clone(self, new_metadata: PromptMetadata = None) -> 'PromptTemplate':
        """Create a clone of this template with optional new metadata."""
        metadata = new_metadata or self.metadata.copy()
        return self.__class__(
            template=self.template,
            metadata=metadata,
            variables=self.variables.copy(),
            parent_template=self.parent_template
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert template to dictionary representation."""
        return {
            "template": self.template,
            "metadata": self.metadata.dict(),
            "variables": [var.dict() for var in self.variables],
            "usage_count": self.usage_count,
            "last_used": self.last_used.isoformat() if self.last_used else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PromptTemplate':
        """Create template from dictionary representation."""
        metadata = PromptMetadata(**data["metadata"])
        variables = [PromptVariable(**var_data) for var_data in data.get("variables", [])]
        
        instance = cls(
            template=data["template"],
            metadata=metadata,
            variables=variables
        )
        
        instance.usage_count = data.get("usage_count", 0)
        if data.get("last_used"):
            instance.last_used = datetime.fromisoformat(data["last_used"])
        
        return instance


class SystemPromptTemplate(PromptTemplate):
    """
    Template for system-level prompts that define the AI's role and behavior.
    
    System prompts are typically used to establish the AI's persona, capabilities,
    and general guidelines for interaction.
    """
    
    def get_prompt_type(self) -> PromptType:
        return PromptType.SYSTEM
    
    def __init__(
        self,
        template: str,
        metadata: PromptMetadata,
        variables: List[PromptVariable] = None,
        persona_definition: str = None,
        capability_description: str = None,
        behavioral_guidelines: List[str] = None
    ):
        super().__init__(template, metadata, variables)
        self.persona_definition = persona_definition
        self.capability_description = capability_description
        self.behavioral_guidelines = behavioral_guidelines or []
    
    def render_with_persona(
        self,
        persona_variables: Dict[str, Any],
        variables: PromptVariables = None,
        context: PromptContext = None
    ) -> PromptResult:
        """Render template with specific persona variables."""
        merged_variables = variables.copy() if variables else {}
        merged_variables.update(persona_variables)
        
        # Add persona-specific context
        if self.persona_definition:
            merged_variables["persona"] = self.persona_definition
        if self.capability_description:
            merged_variables["capabilities"] = self.capability_description
        if self.behavioral_guidelines:
            merged_variables["guidelines"] = "\n".join(f"- {guideline}" for guideline in self.behavioral_guidelines)
        
        return self.render(merged_variables, context)


class AgentPromptTemplate(PromptTemplate):
    """
    Template for agent-specific prompts that define specialized behavior.
    
    Agent prompts are tailored to specific agent types and their unique capabilities.
    """
    
    def get_prompt_type(self) -> PromptType:
        return PromptType.USER
    
    def __init__(
        self,
        template: str,
        metadata: PromptMetadata,
        variables: List[PromptVariable] = None,
        agent_type: str = None,
        capabilities: List[str] = None,
        specialized_instructions: Dict[str, str] = None
    ):
        super().__init__(template, metadata, variables)
        self.agent_type = agent_type
        self.capabilities = capabilities or []
        self.specialized_instructions = specialized_instructions or {}
    
    def render_for_capability(
        self,
        capability: str,
        variables: PromptVariables = None,
        context: PromptContext = None
    ) -> PromptResult:
        """Render template for a specific capability."""
        merged_variables = variables.copy() if variables else {}
        
        # Add capability-specific context
        merged_variables["current_capability"] = capability
        if capability in self.specialized_instructions:
            merged_variables["specialized_instruction"] = self.specialized_instructions[capability]
        
        # Update context
        if context:
            context.capability = capability
        else:
            context = PromptContext(capability=capability)
        
        return self.render(merged_variables, context)


class TaskPromptTemplate(PromptTemplate):
    """
    Template for task-specific prompts that guide specific operations.
    
    Task prompts are designed for particular operations like analysis,
    generation, classification, etc.
    """
    
    def get_prompt_type(self) -> PromptType:
        return PromptType.USER
    
    def __init__(
        self,
        template: str,
        metadata: PromptMetadata,
        variables: List[PromptVariable] = None,
        task_type: str = None,
        input_format: str = None,
        output_format: str = None,
        examples: List[Dict[str, Any]] = None
    ):
        super().__init__(template, metadata, variables)
        self.task_type = task_type
        self.input_format = input_format
        self.output_format = output_format
        self.examples = examples or []
    
    def render_with_examples(
        self,
        variables: PromptVariables = None,
        context: PromptContext = None,
        num_examples: int = None
    ) -> PromptResult:
        """Render template including few-shot examples."""
        merged_variables = variables.copy() if variables else {}
        
        # Add examples
        examples_to_use = self.examples
        if num_examples and num_examples < len(self.examples):
            examples_to_use = self.examples[:num_examples]
        
        if examples_to_use:
            example_text = self._format_examples(examples_to_use)
            merged_variables["examples"] = example_text
        
        # Add format specifications
        if self.input_format:
            merged_variables["input_format"] = self.input_format
        if self.output_format:
            merged_variables["output_format"] = self.output_format
        
        return self.render(merged_variables, context)
    
    def _format_examples(self, examples: List[Dict[str, Any]]) -> str:
        """Format examples for inclusion in prompt."""
        formatted_examples = []
        for i, example in enumerate(examples, 1):
            example_text = f"Example {i}:\n"
            if "input" in example:
                example_text += f"Input: {example['input']}\n"
            if "output" in example:
                example_text += f"Output: {example['output']}\n"
            formatted_examples.append(example_text)
        
        return "\n".join(formatted_examples)


class ChainOfThoughtTemplate(TaskPromptTemplate):
    """
    Template for chain-of-thought prompting technique.
    
    Guides the AI through step-by-step reasoning processes.
    """
    
    def __init__(
        self,
        template: str,
        metadata: PromptMetadata,
        variables: List[PromptVariable] = None,
        reasoning_steps: List[str] = None,
        thinking_framework: str = None
    ):
        super().__init__(template, metadata, variables)
        self.reasoning_steps = reasoning_steps or []
        self.thinking_framework = thinking_framework
        
        # Update metadata to reflect CoT technique
        self.metadata.technique = PromptTechnique.CHAIN_OF_THOUGHT
    
    def render_with_reasoning(
        self,
        variables: PromptVariables = None,
        context: PromptContext = None
    ) -> PromptResult:
        """Render template with chain-of-thought reasoning structure."""
        merged_variables = variables.copy() if variables else {}
        
        # Add reasoning structure
        if self.reasoning_steps:
            steps_text = "\n".join(f"{i+1}. {step}" for i, step in enumerate(self.reasoning_steps))
            merged_variables["reasoning_steps"] = steps_text
        
        if self.thinking_framework:
            merged_variables["thinking_framework"] = self.thinking_framework
        
        # Add CoT-specific instructions
        merged_variables["cot_instruction"] = "Let's work through this step-by-step:"
        
        return self.render(merged_variables, context)


class FewShotTemplate(TaskPromptTemplate):
    """
    Template for few-shot prompting with curated examples.
    
    Provides examples to guide the AI's understanding and response format.
    """
    
    def __init__(
        self,
        template: str,
        metadata: PromptMetadata,
        variables: List[PromptVariable] = None,
        examples: List[Dict[str, Any]] = None,
        example_selection_strategy: str = "sequential"
    ):
        super().__init__(template, metadata, variables, examples=examples)
        self.example_selection_strategy = example_selection_strategy
        
        # Update metadata to reflect few-shot technique
        self.metadata.technique = PromptTechnique.FEW_SHOT
    
    def render_with_selected_examples(
        self,
        variables: PromptVariables = None,
        context: PromptContext = None,
        selection_criteria: Dict[str, Any] = None
    ) -> PromptResult:
        """Render with intelligently selected examples."""
        selected_examples = self._select_examples(selection_criteria or {})
        
        # Temporarily update examples for rendering
        original_examples = self.examples
        self.examples = selected_examples
        
        result = self.render_with_examples(variables, context)
        
        # Restore original examples
        self.examples = original_examples
        
        return result
    
    def _select_examples(self, criteria: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Select examples based on criteria."""
        if not self.examples:
            return []
        
        # Simple selection strategies
        if self.example_selection_strategy == "random":
            import random
            num_examples = criteria.get("num_examples", min(3, len(self.examples)))
            return random.sample(self.examples, num_examples)
        elif self.example_selection_strategy == "similarity":
            # Could implement similarity-based selection here
            return self.examples[:criteria.get("num_examples", 3)]
        else:  # sequential
            return self.examples[:criteria.get("num_examples", 3)]
