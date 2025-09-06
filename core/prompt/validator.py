"""
Prompt Validator

Validates prompt templates and generated prompts for quality, safety,
and compliance with best practices.
"""

import re
import json
from typing import Dict, List, Optional, Any, Set
from datetime import datetime
import logging

from .types import (
    PromptValidationResult,
    PromptMetadata,
    PromptTemplate,
    PromptTechnique,
    PromptCategory,
    OptimizationMetric
)

logger = logging.getLogger(__name__)


class PromptValidator:
    """
    Comprehensive prompt validation system that checks:
    - Syntax and structure
    - Variable usage and consistency  
    - Content safety and appropriateness
    - Technique-specific requirements
    - Performance characteristics
    - Best practice compliance
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Validation thresholds
        self.max_length = self.config.get("max_length", 8192)
        self.min_length = self.config.get("min_length", 10)
        self.max_variables = self.config.get("max_variables", 50)
        
        # Content safety patterns
        self.prohibited_patterns = self._load_prohibited_patterns()
        self.sensitive_topics = self._load_sensitive_topics()
        
        # Best practice checks
        self.technique_requirements = self._load_technique_requirements()
        
        logger.info("PromptValidator initialized with max_length=%d", self.max_length)
    
    def validate_template(self, template: PromptTemplate) -> PromptValidationResult:
        """
        Comprehensive validation of a prompt template.
        
        Args:
            template: PromptTemplate to validate
            
        Returns:
            PromptValidationResult with validation details
        """
        issues = []
        warnings = []
        suggestions = []
        
        # Basic structure validation
        structure_result = self._validate_structure(template)
        issues.extend(structure_result.get("issues", []))
        warnings.extend(structure_result.get("warnings", []))
        
        # Variable validation
        variable_result = self._validate_variables(template)
        issues.extend(variable_result.get("issues", []))
        warnings.extend(variable_result.get("warnings", []))
        
        # Content safety validation
        safety_result = self._validate_content_safety(template.template)
        issues.extend(safety_result.get("issues", []))
        warnings.extend(safety_result.get("warnings", []))
        
        # Technique-specific validation
        technique_result = self._validate_technique_compliance(template)
        issues.extend(technique_result.get("issues", []))
        suggestions.extend(technique_result.get("suggestions", []))
        
        # Performance validation
        performance_result = self._validate_performance_characteristics(template)
        warnings.extend(performance_result.get("warnings", []))
        suggestions.extend(performance_result.get("suggestions", []))
        
        # Best practices validation
        best_practices_result = self._validate_best_practices(template)
        suggestions.extend(best_practices_result.get("suggestions", []))
        
        # Calculate overall validation score
        validation_score = self._calculate_validation_score(
            len(issues), len(warnings), len(suggestions)
        )
        
        return PromptValidationResult(
            is_valid=len(issues) == 0,
            validation_score=validation_score,
            issues=issues,
            warnings=warnings,
            suggestions=suggestions,
            syntax_valid=structure_result.get("syntax_valid", True),
            variable_valid=variable_result.get("variable_valid", True),
            length_valid=structure_result.get("length_valid", True),
            content_safe=safety_result.get("content_safe", True),
            technique_appropriate=technique_result.get("technique_appropriate", True)
        )
    
    def validate_prompt_text(self, prompt_text: str) -> PromptValidationResult:
        """
        Validate a rendered prompt text.
        
        Args:
            prompt_text: The generated prompt text to validate
            
        Returns:
            PromptValidationResult with validation details
        """
        issues = []
        warnings = []
        suggestions = []
        
        # Basic text validation
        if not prompt_text or not prompt_text.strip():
            issues.append("Prompt text is empty")
            return PromptValidationResult(
                is_valid=False,
                validation_score=0.0,
                issues=issues
            )
        
        # Length validation
        length_result = self._validate_text_length(prompt_text)
        if not length_result["valid"]:
            issues.extend(length_result["issues"])
        warnings.extend(length_result.get("warnings", []))
        
        # Content safety
        safety_result = self._validate_content_safety(prompt_text)
        issues.extend(safety_result.get("issues", []))
        warnings.extend(safety_result.get("warnings", []))
        
        # Structure and clarity
        clarity_result = self._validate_text_clarity(prompt_text)
        warnings.extend(clarity_result.get("warnings", []))
        suggestions.extend(clarity_result.get("suggestions", []))
        
        validation_score = self._calculate_validation_score(
            len(issues), len(warnings), len(suggestions)
        )
        
        return PromptValidationResult(
            is_valid=len(issues) == 0,
            validation_score=validation_score,
            issues=issues,
            warnings=warnings,
            suggestions=suggestions,
            syntax_valid=True,
            variable_valid=True,
            length_valid=length_result["valid"],
            content_safe=safety_result.get("content_safe", True)
        )
    
    def _validate_structure(self, template: PromptTemplate) -> Dict[str, Any]:
        """Validate basic template structure."""
        issues = []
        warnings = []
        
        # Check template content
        if not template.template or not template.template.strip():
            issues.append("Template content is empty")
            return {
                "issues": issues,
                "syntax_valid": False,
                "length_valid": False
            }
        
        # Length validation
        template_length = len(template.template)
        length_valid = True
        
        if template_length > self.max_length:
            issues.append(f"Template exceeds maximum length ({template_length} > {self.max_length})")
            length_valid = False
        elif template_length < self.min_length:
            issues.append(f"Template is too short ({template_length} < {self.min_length})")
            length_valid = False
        elif template_length > self.max_length * 0.8:
            warnings.append(f"Template is quite long ({template_length} characters)")
        
        # Variable syntax validation
        syntax_valid = self._validate_variable_syntax(template.template)
        if not syntax_valid:
            issues.append("Invalid variable syntax found in template")
        
        # Check for basic structure elements
        if len(template.template.split('\n')) < 2:
            warnings.append("Template might benefit from better structure (multiple lines)")
        
        return {
            "issues": issues,
            "warnings": warnings,
            "syntax_valid": syntax_valid,
            "length_valid": length_valid
        }
    
    def _validate_variables(self, template: PromptTemplate) -> Dict[str, Any]:
        """Validate variable definitions and usage."""
        issues = []
        warnings = []
        
        # Check variable count
        if len(template.variables) > self.max_variables:
            issues.append(f"Too many variables ({len(template.variables)} > {self.max_variables})")
        
        # Find variables used in template
        used_variables = self._extract_template_variables(template.template)
        defined_variables = {var.name for var in template.variables}
        
        # Check for undefined variables
        undefined = used_variables - defined_variables
        if undefined:
            issues.append(f"Undefined variables used in template: {', '.join(undefined)}")
        
        # Check for unused variable definitions
        unused = defined_variables - used_variables
        if unused:
            warnings.append(f"Defined but unused variables: {', '.join(unused)}")
        
        # Validate individual variable definitions
        for variable in template.variables:
            var_issues = self._validate_single_variable(variable)
            issues.extend(var_issues)
        
        return {
            "issues": issues,
            "warnings": warnings,
            "variable_valid": len(issues) == 0
        }
    
    def _validate_content_safety(self, content: str) -> Dict[str, Any]:
        """Validate content for safety and appropriateness."""
        issues = []
        warnings = []
        content_safe = True
        
        # Check for prohibited patterns
        for pattern, description in self.prohibited_patterns.items():
            if re.search(pattern, content, re.IGNORECASE):
                issues.append(f"Prohibited content detected: {description}")
                content_safe = False
        
        # Check for sensitive topics
        for topic in self.sensitive_topics:
            if topic.lower() in content.lower():
                warnings.append(f"Sensitive topic detected: {topic}")
        
        # Check for potential prompt injection patterns
        injection_patterns = [
            r"ignore\s+previous\s+instructions",
            r"system\s*:\s*you\s+are",
            r"override\s+your\s+programming",
            r"act\s+as\s+if\s+you\s+are"
        ]
        
        for pattern in injection_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                warnings.append("Potential prompt injection pattern detected")
                break
        
        return {
            "issues": issues,
            "warnings": warnings,
            "content_safe": content_safe
        }
    
    def _validate_technique_compliance(self, template: PromptTemplate) -> Dict[str, Any]:
        """Validate compliance with prompt engineering technique requirements."""
        issues = []
        suggestions = []
        technique_appropriate = True
        
        technique = template.metadata.technique
        requirements = self.technique_requirements.get(technique, {})
        
        # Check technique-specific requirements
        if technique == PromptTechnique.FEW_SHOT:
            if hasattr(template, 'examples') and len(template.examples) == 0:
                issues.append("Few-shot technique requires examples")
                technique_appropriate = False
            elif hasattr(template, 'examples') and len(template.examples) < 2:
                suggestions.append("Few-shot prompts typically work better with 2+ examples")
        
        elif technique == PromptTechnique.CHAIN_OF_THOUGHT:
            if "step" not in template.template.lower() and "think" not in template.template.lower():
                suggestions.append("Chain-of-thought prompts should include thinking/reasoning instructions")
        
        elif technique == PromptTechnique.ROLE_PLAYING:
            if "you are" not in template.template.lower():
                suggestions.append("Role-playing prompts should define the AI's role")
        
        # Check for appropriate technique based on content
        content_lower = template.template.lower()
        
        if "example" in content_lower and technique != PromptTechnique.FEW_SHOT:
            suggestions.append("Consider using few-shot technique when examples are provided")
        
        if ("step" in content_lower or "think" in content_lower) and technique != PromptTechnique.CHAIN_OF_THOUGHT:
            suggestions.append("Consider using chain-of-thought technique for reasoning tasks")
        
        return {
            "issues": issues,
            "suggestions": suggestions,
            "technique_appropriate": technique_appropriate
        }
    
    def _validate_performance_characteristics(self, template: PromptTemplate) -> Dict[str, Any]:
        """Validate performance-related characteristics."""
        warnings = []
        suggestions = []
        
        # Estimate token count
        estimated_tokens = len(template.template) // 4  # Rough estimate
        
        if estimated_tokens > 1000:
            warnings.append(f"High estimated token count ({estimated_tokens})")
            suggestions.append("Consider breaking down into smaller, focused prompts")
        
        # Check complexity
        sentence_count = len(re.split(r'[.!?]+', template.template))
        if sentence_count > 20:
            suggestions.append("Consider simplifying: prompt has many sentences")
        
        # Check variable complexity
        if len(template.variables) > 10:
            suggestions.append("Consider reducing number of variables for better usability")
        
        return {
            "warnings": warnings,
            "suggestions": suggestions
        }
    
    def _validate_best_practices(self, template: PromptTemplate) -> Dict[str, Any]:
        """Validate against prompt engineering best practices."""
        suggestions = []
        
        content = template.template.lower()
        
        # Check for clear instructions
        if "please" in content:
            suggestions.append("Consider using direct instructions instead of polite requests")
        
        # Check for specificity
        vague_terms = ["good", "bad", "nice", "some", "many", "several"]
        for term in vague_terms:
            if f" {term} " in content:
                suggestions.append(f"Consider being more specific than '{term}'")
                break
        
        # Check for output format specification
        if template.metadata.category == PromptCategory.TASK_SPECIFIC:
            if "format" not in content and "structure" not in content:
                suggestions.append("Consider specifying expected output format")
        
        # Check for constraints
        if "don't" in content or "do not" in content:
            suggestions.append("Consider stating what TO do rather than what NOT to do")
        
        # Check metadata completeness
        if not template.metadata.description:
            suggestions.append("Add template description for better documentation")
        
        if not template.metadata.tags:
            suggestions.append("Add tags for better template organization")
        
        return {
            "suggestions": suggestions
        }
    
    def _validate_variable_syntax(self, template_text: str) -> bool:
        """Validate variable syntax in template."""
        # Check for balanced braces
        open_braces = template_text.count('{{')
        close_braces = template_text.count('}}')
        
        if open_braces != close_braces:
            return False
        
        # Check for valid variable names
        variable_pattern = r'\{\{([^}]+)\}\}'
        variables = re.findall(variable_pattern, template_text)
        
        for var in variables:
            var = var.strip()
            if not var:
                return False
            # Variable names should be valid identifiers
            if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', var):
                return False
        
        return True
    
    def _extract_template_variables(self, template_text: str) -> Set[str]:
        """Extract variable names from template text."""
        variable_pattern = r'\{\{([^}]+)\}\}'
        variables = re.findall(variable_pattern, template_text)
        return {var.strip() for var in variables}
    
    def _validate_single_variable(self, variable) -> List[str]:
        """Validate a single variable definition."""
        issues = []
        
        # Check required fields
        if not variable.name:
            issues.append("Variable name is required")
        elif not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', variable.name):
            issues.append(f"Invalid variable name: {variable.name}")
        
        if not variable.description:
            issues.append(f"Variable '{variable.name}' missing description")
        
        # Check type validity
        valid_types = ["string", "number", "boolean", "array", "object"]
        if variable.type not in valid_types:
            issues.append(f"Invalid variable type: {variable.type}")
        
        # Check constraints
        if variable.constraints:
            if variable.type == "string":
                if "min_length" in variable.constraints and variable.constraints["min_length"] < 0:
                    issues.append(f"Invalid min_length for {variable.name}")
                if "max_length" in variable.constraints and variable.constraints["max_length"] < 1:
                    issues.append(f"Invalid max_length for {variable.name}")
        
        return issues
    
    def _validate_text_length(self, text: str) -> Dict[str, Any]:
        """Validate text length."""
        text_length = len(text)
        issues = []
        warnings = []
        
        valid = True
        if text_length > self.max_length:
            issues.append(f"Text exceeds maximum length ({text_length} > {self.max_length})")
            valid = False
        elif text_length < self.min_length:
            issues.append(f"Text is too short ({text_length} < {self.min_length})")
            valid = False
        elif text_length > self.max_length * 0.8:
            warnings.append(f"Text is quite long ({text_length} characters)")
        
        return {
            "valid": valid,
            "issues": issues,
            "warnings": warnings
        }
    
    def _validate_text_clarity(self, text: str) -> Dict[str, Any]:
        """Validate text clarity and structure."""
        warnings = []
        suggestions = []
        
        # Check for very long sentences
        sentences = re.split(r'[.!?]+', text)
        long_sentences = [s for s in sentences if len(s.strip()) > 150]
        
        if long_sentences:
            suggestions.append("Consider breaking down long sentences for clarity")
        
        # Check for repetitive words
        words = re.findall(r'\b\w+\b', text.lower())
        word_freq = {}
        for word in words:
            if len(word) > 3:  # Only check meaningful words
                word_freq[word] = word_freq.get(word, 0) + 1
        
        repetitive_words = [word for word, count in word_freq.items() if count > 5]
        if repetitive_words:
            suggestions.append("Consider varying vocabulary to avoid repetition")
        
        return {
            "warnings": warnings,
            "suggestions": suggestions
        }
    
    def _calculate_validation_score(self, issues_count: int, warnings_count: int, suggestions_count: int) -> float:
        """Calculate overall validation score."""
        base_score = 100.0
        
        # Deduct points for issues (critical)
        base_score -= issues_count * 20.0
        
        # Deduct points for warnings (moderate)
        base_score -= warnings_count * 5.0
        
        # Deduct points for suggestions (minor)
        base_score -= suggestions_count * 1.0
        
        return max(0.0, min(100.0, base_score)) / 100.0
    
    def _load_prohibited_patterns(self) -> Dict[str, str]:
        """Load prohibited content patterns."""
        return {
            r'\b(hack|crack|exploit)\b': "Hacking/security exploitation terms",
            r'\b(illegal|unlawful)\b': "Illegal activity references",
            r'\b(violence|violent|harm)\b': "Violence-related content",
            r'\b(discrimination|racist|sexist)\b': "Discriminatory content"
        }
    
    def _load_sensitive_topics(self) -> List[str]:
        """Load list of sensitive topics that should be flagged."""
        return [
            "politics", "religion", "personal information", "financial data",
            "medical advice", "legal advice", "adult content"
        ]
    
    def _load_technique_requirements(self) -> Dict[PromptTechnique, Dict[str, Any]]:
        """Load requirements for different prompt techniques."""
        return {
            PromptTechnique.FEW_SHOT: {
                "min_examples": 1,
                "recommended_examples": 3
            },
            PromptTechnique.CHAIN_OF_THOUGHT: {
                "requires_reasoning": True,
                "step_keywords": ["step", "think", "reason", "analyze"]
            },
            PromptTechnique.ROLE_PLAYING: {
                "requires_role_definition": True,
                "role_keywords": ["you are", "act as", "imagine you"]
            }
        }
