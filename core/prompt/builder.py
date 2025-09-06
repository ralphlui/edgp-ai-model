"""
Prompt Builder Utility

Provides a fluent interface for building complex prompts with proper
structure, validation, and optimization following industry best practices.
"""

from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import json

from .types import (
    PromptType,
    PromptCategory,
    PromptTechnique,
    PromptMetadata,
    PromptVariable,
    PromptContext,
    PromptResult,
    PromptPriority,
    PromptStatus
)
from .templates import (
    PromptTemplate,
    SystemPromptTemplate,
    AgentPromptTemplate,
    TaskPromptTemplate,
    ChainOfThoughtTemplate,
    FewShotTemplate
)


class PromptBuilder:
    """
    Fluent builder for creating and composing prompts with proper structure.
    
    Supports various prompt engineering techniques and best practices:
    - Chain of thought reasoning
    - Few-shot learning
    - Role-based prompting
    - Structured output formatting
    - Context injection
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self) -> 'PromptBuilder':
        """Reset builder to initial state."""
        self._template_parts = []
        self._metadata = None
        self._variables = []
        self._examples = []
        self._constraints = []
        self._output_format = None
        self._reasoning_steps = []
        self._context_variables = {}
        self._technique = PromptTechnique.INSTRUCTION_FOLLOWING
        self._category = PromptCategory.TASK_SPECIFIC
        return self
    
    def system(self, content: str) -> 'PromptBuilder':
        """Add system-level instructions."""
        self._template_parts.append(("system", content))
        self._category = PromptCategory.SYSTEM_CORE
        return self
    
    def role(self, role_description: str, capabilities: List[str] = None) -> 'PromptBuilder':
        """Define the AI's role and capabilities."""
        role_text = f"You are {role_description}."
        if capabilities:
            capabilities_text = "\n".join(f"- {cap}" for cap in capabilities)
            role_text += f"\n\nYour capabilities include:\n{capabilities_text}"
        
        self._template_parts.append(("role", role_text))
        self._technique = PromptTechnique.ROLE_PLAYING
        return self
    
    def task(self, task_description: str, importance: str = "important") -> 'PromptBuilder':
        """Define the specific task to be performed."""
        task_text = f"Your task is to {task_description}."
        if importance == "critical":
            task_text = f"**CRITICAL TASK:** {task_text}"
        elif importance == "important":
            task_text = f"**Important:** {task_text}"
        
        self._template_parts.append(("task", task_text))
        return self
    
    def context(self, context_info: str, variables: Dict[str, Any] = None) -> 'PromptBuilder':
        """Add contextual information."""
        self._template_parts.append(("context", f"Context: {context_info}"))
        if variables:
            self._context_variables.update(variables)
        return self
    
    def input_format(self, format_description: str, example: str = None) -> 'PromptBuilder':
        """Specify the expected input format."""
        format_text = f"Input format: {format_description}"
        if example:
            format_text += f"\n\nExample input:\n{example}"
        
        self._template_parts.append(("input_format", format_text))
        return self
    
    def output_format(
        self,
        format_description: str,
        structure: Dict[str, Any] = None,
        example: str = None
    ) -> 'PromptBuilder':
        """Specify the required output format."""
        format_text = f"Output format: {format_description}"
        
        if structure:
            structure_text = json.dumps(structure, indent=2)
            format_text += f"\n\nRequired structure:\n```json\n{structure_text}\n```"
        
        if example:
            format_text += f"\n\nExample output:\n{example}"
        
        self._template_parts.append(("output_format", format_text))
        self._output_format = format_description
        return self
    
    def constraint(self, constraint_description: str) -> 'PromptBuilder':
        """Add constraints or requirements."""
        self._constraints.append(constraint_description)
        return self
    
    def constraints(self, constraint_list: List[str]) -> 'PromptBuilder':
        """Add multiple constraints."""
        self._constraints.extend(constraint_list)
        return self
    
    def example(
        self,
        input_text: str = None,
        output_text: str = None,
        explanation: str = None
    ) -> 'PromptBuilder':
        """Add a few-shot example."""
        example_data = {}
        if input_text:
            example_data["input"] = input_text
        if output_text:
            example_data["output"] = output_text
        if explanation:
            example_data["explanation"] = explanation
        
        self._examples.append(example_data)
        if len(self._examples) == 1:
            self._technique = PromptTechnique.FEW_SHOT
        return self
    
    def examples(self, example_list: List[Dict[str, str]]) -> 'PromptBuilder':
        """Add multiple examples for few-shot learning."""
        self._examples.extend(example_list)
        self._technique = PromptTechnique.FEW_SHOT
        return self
    
    def chain_of_thought(self, reasoning_steps: List[str] = None) -> 'PromptBuilder':
        """Enable chain-of-thought reasoning."""
        if reasoning_steps:
            self._reasoning_steps = reasoning_steps
        else:
            self._reasoning_steps = [
                "Analyze the problem carefully",
                "Consider relevant information and context",
                "Think through the solution step by step",
                "Verify your reasoning",
                "Provide your final answer"
            ]
        
        self._technique = PromptTechnique.CHAIN_OF_THOUGHT
        return self
    
    def thinking_framework(self, framework: str) -> 'PromptBuilder':
        """Add a structured thinking framework."""
        framework_text = f"Use this thinking framework:\n{framework}"
        self._template_parts.append(("thinking_framework", framework_text))
        return self
    
    def step_by_step(self, steps: List[str] = None) -> 'PromptBuilder':
        """Add step-by-step instructions."""
        if not steps:
            steps = ["Break down the problem", "Solve each part", "Combine results"]
        
        steps_text = "Follow these steps:\n" + "\n".join(f"{i+1}. {step}" for i, step in enumerate(steps))
        self._template_parts.append(("steps", steps_text))
        self._technique = PromptTechnique.STEP_BY_STEP
        return self
    
    def variable(
        self,
        name: str,
        description: str,
        var_type: str = "string",
        required: bool = True,
        default_value: Any = None,
        validation_pattern: str = None,
        constraints: Dict[str, Any] = None
    ) -> 'PromptBuilder':
        """Define a template variable."""
        variable = PromptVariable(
            name=name,
            type=var_type,
            description=description,
            required=required,
            default_value=default_value,
            validation_pattern=validation_pattern,
            constraints=constraints or {}
        )
        self._variables.append(variable)
        return self
    
    def placeholder(self, name: str, description: str = None) -> 'PromptBuilder':
        """Add a simple placeholder variable."""
        placeholder_text = f"{{{{{name}}}}}"
        if description:
            placeholder_text += f"  # {description}"
        
        self._template_parts.append(("placeholder", placeholder_text))
        
        # Auto-create variable if not exists
        if not any(var.name == name for var in self._variables):
            self.variable(name, description or f"Value for {name}")
        
        return self
    
    def metadata(
        self,
        name: str,
        description: str,
        author: str,
        version: str = "1.0.0",
        priority: PromptPriority = PromptPriority.MEDIUM,
        tags: List[str] = None
    ) -> 'PromptBuilder':
        """Set template metadata."""
        self._metadata = PromptMetadata(
            name=name,
            description=description,
            author=author,
            version=version,
            category=self._category,
            technique=self._technique,
            priority=priority,
            status=PromptStatus.DRAFT,
            tags=tags or []
        )
        return self
    
    def build_template(self) -> PromptTemplate:
        """Build the final prompt template."""
        if not self._metadata:
            raise ValueError("Metadata must be set before building template")
        
        # Construct the full template
        template_text = self._construct_template()
        
        # Choose appropriate template class
        if self._category == PromptCategory.SYSTEM_CORE:
            return SystemPromptTemplate(
                template=template_text,
                metadata=self._metadata,
                variables=self._variables
            )
        elif self._technique == PromptTechnique.CHAIN_OF_THOUGHT:
            return ChainOfThoughtTemplate(
                template=template_text,
                metadata=self._metadata,
                variables=self._variables,
                reasoning_steps=self._reasoning_steps
            )
        elif self._technique == PromptTechnique.FEW_SHOT:
            return FewShotTemplate(
                template=template_text,
                metadata=self._metadata,
                variables=self._variables,
                examples=self._examples
            )
        else:
            return TaskPromptTemplate(
                template=template_text,
                metadata=self._metadata,
                variables=self._variables,
                examples=self._examples,
                output_format=self._output_format
            )
    
    def build_prompt(
        self,
        variables: Dict[str, Any] = None,
        context: PromptContext = None
    ) -> PromptResult:
        """Build and immediately render the prompt."""
        template = self.build_template()
        merged_variables = variables.copy() if variables else {}
        merged_variables.update(self._context_variables)
        
        return template.render(merged_variables, context)
    
    def _construct_template(self) -> str:
        """Construct the final template text from all parts."""
        sections = []
        
        # Group parts by type for better organization
        grouped_parts = {}
        for part_type, content in self._template_parts:
            if part_type not in grouped_parts:
                grouped_parts[part_type] = []
            grouped_parts[part_type].append(content)
        
        # Build template in logical order
        section_order = [
            "system", "role", "context", "task", 
            "input_format", "thinking_framework", "steps",
            "placeholder", "output_format"
        ]
        
        for section_type in section_order:
            if section_type in grouped_parts:
                sections.extend(grouped_parts[section_type])
        
        # Add examples if using few-shot
        if self._examples and self._technique == PromptTechnique.FEW_SHOT:
            examples_text = self._format_examples()
            sections.append(examples_text)
        
        # Add reasoning structure for chain-of-thought
        if self._reasoning_steps and self._technique == PromptTechnique.CHAIN_OF_THOUGHT:
            cot_text = "Let's work through this step-by-step:\n"
            cot_text += "\n".join(f"{i+1}. {step}" for i, step in enumerate(self._reasoning_steps))
            sections.append(cot_text)
        
        # Add constraints
        if self._constraints:
            constraints_text = "Important constraints:\n"
            constraints_text += "\n".join(f"- {constraint}" for constraint in self._constraints)
            sections.append(constraints_text)
        
        # Add final instruction
        if self._output_format:
            sections.append("Please provide your response in the specified format.")
        
        return "\n\n".join(sections)
    
    def _format_examples(self) -> str:
        """Format examples for inclusion in template."""
        if not self._examples:
            return ""
        
        examples_text = "Here are some examples:\n\n"
        for i, example in enumerate(self._examples, 1):
            example_text = f"Example {i}:\n"
            if "input" in example:
                example_text += f"Input: {example['input']}\n"
            if "output" in example:
                example_text += f"Output: {example['output']}\n"
            if "explanation" in example:
                example_text += f"Explanation: {example['explanation']}\n"
            examples_text += example_text + "\n"
        
        return examples_text.strip()


# Convenience functions for common prompt patterns

def create_analysis_prompt(
    task_description: str,
    input_data: str = None,
    output_format: str = "structured analysis",
    constraints: List[str] = None
) -> PromptBuilder:
    """Create a prompt for data/content analysis."""
    builder = PromptBuilder()
    builder.role("an expert analyst", ["data analysis", "pattern recognition", "insight generation"])
    builder.task(task_description, "important")
    
    if input_data:
        builder.context(f"Data to analyze: {input_data}")
    
    builder.output_format(output_format)
    
    if constraints:
        builder.constraints(constraints)
    
    builder.chain_of_thought([
        "Examine the data carefully",
        "Identify key patterns and trends",
        "Draw meaningful insights",
        "Provide actionable recommendations"
    ])
    
    return builder


def create_generation_prompt(
    content_type: str,
    requirements: str,
    style: str = None,
    examples: List[Dict[str, str]] = None
) -> PromptBuilder:
    """Create a prompt for content generation."""
    builder = PromptBuilder()
    builder.role(f"a skilled {content_type} creator", ["creative writing", "content development"])
    builder.task(f"create {content_type} that {requirements}")
    
    if style:
        builder.constraint(f"Maintain a {style} style")
    
    if examples:
        builder.examples(examples)
    
    return builder


def create_classification_prompt(
    categories: List[str],
    input_description: str,
    criteria: str = None
) -> PromptBuilder:
    """Create a prompt for classification tasks."""
    builder = PromptBuilder()
    builder.role("a classification expert", ["categorization", "pattern matching"])
    builder.task(f"classify the {input_description} into one of the predefined categories")
    
    categories_text = ", ".join(categories)
    builder.context(f"Available categories: {categories_text}")
    
    if criteria:
        builder.context(f"Classification criteria: {criteria}")
    
    builder.output_format("category name only", {"category": "selected_category"})
    
    return builder


def create_extraction_prompt(
    data_to_extract: List[str],
    source_description: str,
    format_structured: bool = True
) -> PromptBuilder:
    """Create a prompt for information extraction."""
    builder = PromptBuilder()
    builder.role("an information extraction specialist", ["data extraction", "structured analysis"])
    
    extract_list = ", ".join(data_to_extract)
    builder.task(f"extract the following information from {source_description}: {extract_list}")
    
    if format_structured:
        structure = {field: f"extracted_{field}" for field in data_to_extract}
        builder.output_format("JSON format", structure)
    
    builder.constraint("Only extract information that is explicitly stated")
    builder.constraint("Use 'N/A' for information that cannot be found")
    
    return builder
