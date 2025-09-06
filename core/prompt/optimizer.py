"""
Prompt Optimizer

Optimizes prompt templates for better performance, clarity, and effectiveness
using various optimization techniques and performance metrics.
"""

import json
import re
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
from collections import defaultdict
import logging

from .types import (
    PromptTemplate,
    PromptMetadata,
    PromptOptimizationResult,
    OptimizationMetric,
    PromptTechnique,
    PromptCategory,
    PromptVariable
)

logger = logging.getLogger(__name__)


@dataclass
class OptimizationConfig:
    """Configuration for prompt optimization."""
    target_metrics: List[OptimizationMetric]
    max_iterations: int = 10
    convergence_threshold: float = 0.01
    preserve_original: bool = True
    optimization_techniques: List[str] = None
    
    def __post_init__(self):
        if self.optimization_techniques is None:
            self.optimization_techniques = [
                "length_optimization",
                "clarity_enhancement", 
                "specificity_improvement",
                "structure_refinement",
                "variable_optimization"
            ]


@dataclass
class PerformanceData:
    """Performance data for a prompt template."""
    accuracy: Optional[float] = None
    response_time: Optional[float] = None
    token_usage: Optional[int] = None
    user_satisfaction: Optional[float] = None
    completion_rate: Optional[float] = None
    cost_per_request: Optional[float] = None
    error_rate: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "accuracy": self.accuracy,
            "response_time": self.response_time,
            "token_usage": self.token_usage,
            "user_satisfaction": self.user_satisfaction,
            "completion_rate": self.completion_rate,
            "cost_per_request": self.cost_per_request,
            "error_rate": self.error_rate
        }


class PromptOptimizer:
    """
    Advanced prompt optimization system that improves prompts based on:
    - Performance metrics
    - User feedback
    - A/B testing results
    - Best practice patterns
    - Automated analysis
    """
    
    def __init__(self, config: OptimizationConfig = None):
        self.config = config or OptimizationConfig(
            target_metrics=[OptimizationMetric.ACCURACY, OptimizationMetric.CLARITY]
        )
        
        # Optimization history
        self.optimization_history: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        
        # Performance database
        self.performance_data: Dict[str, PerformanceData] = {}
        
        # Optimization patterns
        self.optimization_patterns = self._load_optimization_patterns()
        
        logger.info("PromptOptimizer initialized with %d target metrics", 
                   len(self.config.target_metrics))
    
    def optimize_template(
        self, 
        template: PromptTemplate,
        performance_data: Optional[PerformanceData] = None,
        user_feedback: Optional[Dict[str, Any]] = None
    ) -> PromptOptimizationResult:
        """
        Optimize a prompt template based on performance data and feedback.
        
        Args:
            template: PromptTemplate to optimize
            performance_data: Optional performance metrics
            user_feedback: Optional user feedback data
            
        Returns:
            PromptOptimizationResult with optimized template and analysis
        """
        start_time = datetime.now()
        
        # Store performance data
        if performance_data:
            self.performance_data[template.template_id] = performance_data
        
        # Initialize optimization result
        result = PromptOptimizationResult(
            original_template=template,
            optimized_template=template,
            optimization_score=0.0,
            improvements=[]
        )
        
        # Apply optimization techniques
        current_template = template
        total_improvements = []
        
        for technique in self.config.optimization_techniques:
            optimization_func = getattr(self, f"_optimize_{technique}", None)
            if optimization_func:
                try:
                    technique_result = optimization_func(
                        current_template, performance_data, user_feedback
                    )
                    
                    if technique_result and technique_result.get("improved"):
                        current_template = technique_result["template"]
                        total_improvements.extend(technique_result.get("improvements", []))
                        
                        logger.debug("Applied %s optimization to template %s", 
                                   technique, template.template_id)
                        
                except Exception as e:
                    logger.warning("Optimization technique %s failed: %s", technique, e)
        
        # Calculate optimization score
        optimization_score = self._calculate_optimization_score(
            template, current_template, performance_data
        )
        
        # Update result
        result.optimized_template = current_template
        result.optimization_score = optimization_score
        result.improvements = total_improvements
        result.optimization_time = (datetime.now() - start_time).total_seconds()
        
        # Store optimization history
        self._record_optimization(template.template_id, result)
        
        logger.info("Optimized template %s with score %.2f (%d improvements)", 
                   template.template_id, optimization_score, len(total_improvements))
        
        return result
    
    def analyze_performance_trends(
        self, 
        template_id: str,
        time_window: Optional[timedelta] = None
    ) -> Dict[str, Any]:
        """
        Analyze performance trends for a template over time.
        
        Args:
            template_id: Template identifier
            time_window: Optional time window for analysis
            
        Returns:
            Dictionary with trend analysis
        """
        if template_id not in self.optimization_history:
            return {"error": "No optimization history found"}
        
        history = self.optimization_history[template_id]
        
        if time_window:
            cutoff_time = datetime.now() - time_window
            history = [
                h for h in history 
                if datetime.fromisoformat(h["timestamp"]) > cutoff_time
            ]
        
        if not history:
            return {"error": "No data in specified time window"}
        
        # Calculate trends
        scores = [h["optimization_score"] for h in history]
        improvement_counts = [len(h["improvements"]) for h in history]
        
        trends = {
            "optimization_score_trend": {
                "current": scores[-1] if scores else 0,
                "average": sum(scores) / len(scores) if scores else 0,
                "min": min(scores) if scores else 0,
                "max": max(scores) if scores else 0,
                "trend": "improving" if len(scores) > 1 and scores[-1] > scores[0] else "stable"
            },
            "improvement_trend": {
                "total_optimizations": len(history),
                "average_improvements_per_optimization": sum(improvement_counts) / len(improvement_counts) if improvement_counts else 0,
                "most_recent_improvements": improvement_counts[-1] if improvement_counts else 0
            },
            "optimization_frequency": len(history),
            "time_span_days": (
                datetime.fromisoformat(history[-1]["timestamp"]) - 
                datetime.fromisoformat(history[0]["timestamp"])
            ).days if len(history) > 1 else 0
        }
        
        return trends
    
    def suggest_optimizations(
        self, 
        template: PromptTemplate,
        performance_data: Optional[PerformanceData] = None
    ) -> List[Dict[str, Any]]:
        """
        Suggest specific optimizations without applying them.
        
        Args:
            template: PromptTemplate to analyze
            performance_data: Optional performance data
            
        Returns:
            List of optimization suggestions
        """
        suggestions = []
        
        # Analyze current template
        analysis = self._analyze_template_characteristics(template)
        
        # Length optimization suggestions
        if analysis["length"] > 1000:
            suggestions.append({
                "type": "length_optimization",
                "priority": "high",
                "description": "Template is quite long - consider breaking into smaller, focused parts",
                "expected_improvement": "Better focus and reduced token usage",
                "implementation": "Split into multiple templates or remove unnecessary content"
            })
        
        # Clarity suggestions
        if analysis["sentence_complexity"] > 25:
            suggestions.append({
                "type": "clarity_enhancement",
                "priority": "medium",
                "description": "Some sentences are quite complex",
                "expected_improvement": "Improved understanding and follow-through",
                "implementation": "Break down complex sentences into simpler ones"
            })
        
        # Specificity suggestions
        if analysis["vague_terms"] > 3:
            suggestions.append({
                "type": "specificity_improvement", 
                "priority": "medium",
                "description": "Template contains vague terms that could be more specific",
                "expected_improvement": "More precise and actionable instructions",
                "implementation": "Replace vague terms with specific criteria or examples"
            })
        
        # Variable optimization suggestions
        if len(template.variables) > 10:
            suggestions.append({
                "type": "variable_optimization",
                "priority": "low",
                "description": "Template has many variables which might complicate usage",
                "expected_improvement": "Simplified interface and better usability",
                "implementation": "Consolidate related variables or provide sensible defaults"
            })
        
        # Performance-based suggestions
        if performance_data:
            if performance_data.response_time and performance_data.response_time > 5.0:
                suggestions.append({
                    "type": "performance_optimization",
                    "priority": "high",
                    "description": "Template shows slow response times",
                    "expected_improvement": "Faster processing and better user experience",
                    "implementation": "Reduce complexity or split into multiple steps"
                })
            
            if performance_data.accuracy and performance_data.accuracy < 0.8:
                suggestions.append({
                    "type": "accuracy_improvement",
                    "priority": "high",
                    "description": "Template shows low accuracy results",
                    "expected_improvement": "Better output quality and relevance",
                    "implementation": "Add more specific instructions or examples"
                })
        
        return sorted(suggestions, key=lambda x: {"high": 3, "medium": 2, "low": 1}[x["priority"]], reverse=True)
    
    def _optimize_length_optimization(
        self, 
        template: PromptTemplate,
        performance_data: Optional[PerformanceData],
        user_feedback: Optional[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Optimize template length while preserving meaning."""
        original_length = len(template.template)
        
        # Remove redundant phrases
        optimized_text = self._remove_redundancy(template.template)
        
        # Simplify verbose constructions
        optimized_text = self._simplify_constructions(optimized_text)
        
        # Remove unnecessary words
        optimized_text = self._remove_unnecessary_words(optimized_text)
        
        new_length = len(optimized_text)
        
        if new_length < original_length * 0.95:  # At least 5% reduction
            optimized_template = template.copy()
            optimized_template.template = optimized_text
            
            return {
                "improved": True,
                "template": optimized_template,
                "improvements": [
                    f"Reduced template length from {original_length} to {new_length} characters"
                ]
            }
        
        return {"improved": False}
    
    def _optimize_clarity_enhancement(
        self,
        template: PromptTemplate,
        performance_data: Optional[PerformanceData],
        user_feedback: Optional[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Enhance template clarity and readability."""
        improvements = []
        optimized_text = template.template
        
        # Break down long sentences
        optimized_text, sentence_improvements = self._break_long_sentences(optimized_text)
        improvements.extend(sentence_improvements)
        
        # Improve structure with better formatting
        optimized_text, structure_improvements = self._improve_structure(optimized_text)
        improvements.extend(structure_improvements)
        
        # Add clear section headers where appropriate
        optimized_text, header_improvements = self._add_section_headers(optimized_text)
        improvements.extend(header_improvements)
        
        if improvements:
            optimized_template = template.copy()
            optimized_template.template = optimized_text
            
            return {
                "improved": True,
                "template": optimized_template,
                "improvements": improvements
            }
        
        return {"improved": False}
    
    def _optimize_specificity_improvement(
        self,
        template: PromptTemplate, 
        performance_data: Optional[PerformanceData],
        user_feedback: Optional[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Improve template specificity by replacing vague terms."""
        improvements = []
        optimized_text = template.template
        
        # Replace vague terms with more specific ones
        vague_replacements = {
            r'\bsome\b': 'specific',
            r'\bmany\b': 'several',
            r'\bgood\b': 'high-quality',
            r'\bbad\b': 'problematic',
            r'\bnice\b': 'well-structured',
            r'\bokay\b': 'acceptable',
            r'\bstuff\b': 'content',
            r'\bthings\b': 'elements'
        }
        
        for pattern, replacement in vague_replacements.items():
            if re.search(pattern, optimized_text, re.IGNORECASE):
                optimized_text = re.sub(pattern, replacement, optimized_text, flags=re.IGNORECASE)
                improvements.append(f"Replaced vague term with more specific language")
        
        # Add specific examples where helpful
        if "for example" not in optimized_text.lower() and template.metadata.technique != PromptTechnique.FEW_SHOT:
            # Look for places where examples would be helpful
            if re.search(r'(such as|like|including)', optimized_text, re.IGNORECASE):
                improvements.append("Consider adding specific examples to illustrate concepts")
        
        if improvements:
            optimized_template = template.copy()
            optimized_template.template = optimized_text
            
            return {
                "improved": True,
                "template": optimized_template,
                "improvements": improvements
            }
        
        return {"improved": False}
    
    def _optimize_structure_refinement(
        self,
        template: PromptTemplate,
        performance_data: Optional[PerformanceData], 
        user_feedback: Optional[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Refine template structure for better organization."""
        improvements = []
        optimized_text = template.template
        
        # Ensure proper paragraph structure
        if '\n\n' not in optimized_text and len(optimized_text.split('\n')) > 3:
            # Add paragraph breaks
            lines = optimized_text.split('\n')
            paragraphs = []
            current_paragraph = []
            
            for line in lines:
                line = line.strip()
                if line:
                    current_paragraph.append(line)
                elif current_paragraph:
                    paragraphs.append(' '.join(current_paragraph))
                    current_paragraph = []
            
            if current_paragraph:
                paragraphs.append(' '.join(current_paragraph))
            
            optimized_text = '\n\n'.join(paragraphs)
            improvements.append("Improved paragraph structure for better readability")
        
        # Add logical flow indicators
        if not re.search(r'\b(first|second|third|finally|next|then)\b', optimized_text, re.IGNORECASE):
            # Look for sequences that could benefit from numbering
            sentences = re.split(r'[.!?]+', optimized_text)
            if len(sentences) > 3:
                improvements.append("Consider adding logical flow indicators (first, then, finally)")
        
        if improvements:
            optimized_template = template.copy()
            optimized_template.template = optimized_text
            
            return {
                "improved": True,
                "template": optimized_template,
                "improvements": improvements
            }
        
        return {"improved": False}
    
    def _optimize_variable_optimization(
        self,
        template: PromptTemplate,
        performance_data: Optional[PerformanceData],
        user_feedback: Optional[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Optimize template variables for better usability."""
        improvements = []
        optimized_template = template.copy()
        
        # Consolidate similar variables
        variable_groups = self._group_similar_variables(template.variables)
        
        if len(variable_groups) < len(template.variables):
            improvements.append("Consolidated similar variables for simplicity")
        
        # Add default values where appropriate
        for variable in optimized_template.variables:
            if not variable.default_value and variable.type == "string":
                if "name" in variable.name.lower():
                    variable.default_value = "User"
                    improvements.append(f"Added default value for {variable.name}")
                elif "format" in variable.name.lower():
                    variable.default_value = "standard"
                    improvements.append(f"Added default value for {variable.name}")
        
        # Improve variable descriptions
        for variable in optimized_template.variables:
            if len(variable.description) < 10:
                original_desc = variable.description
                variable.description = self._enhance_variable_description(variable)
                if variable.description != original_desc:
                    improvements.append(f"Enhanced description for variable {variable.name}")
        
        if improvements:
            return {
                "improved": True,
                "template": optimized_template,
                "improvements": improvements
            }
        
        return {"improved": False}
    
    def _analyze_template_characteristics(self, template: PromptTemplate) -> Dict[str, Any]:
        """Analyze template characteristics for optimization insights."""
        content = template.template
        
        return {
            "length": len(content),
            "word_count": len(content.split()),
            "sentence_count": len(re.split(r'[.!?]+', content)),
            "sentence_complexity": self._calculate_sentence_complexity(content),
            "vague_terms": self._count_vague_terms(content),
            "variable_count": len(template.variables),
            "paragraph_count": len(content.split('\n\n')),
            "question_count": content.count('?'),
            "instruction_count": self._count_instructions(content)
        }
    
    def _calculate_optimization_score(
        self,
        original: PromptTemplate,
        optimized: PromptTemplate,
        performance_data: Optional[PerformanceData]
    ) -> float:
        """Calculate optimization score based on improvements."""
        score = 0.0
        
        # Length improvement
        original_length = len(original.template)
        optimized_length = len(optimized.template)
        
        if optimized_length < original_length:
            length_improvement = (original_length - optimized_length) / original_length
            score += min(length_improvement * 30, 15)  # Max 15 points for length
        
        # Clarity improvement (estimated)
        original_complexity = self._calculate_sentence_complexity(original.template)
        optimized_complexity = self._calculate_sentence_complexity(optimized.template)
        
        if optimized_complexity < original_complexity:
            clarity_improvement = (original_complexity - optimized_complexity) / original_complexity
            score += min(clarity_improvement * 40, 20)  # Max 20 points for clarity
        
        # Structure improvement
        original_structure = self._analyze_structure_quality(original.template)
        optimized_structure = self._analyze_structure_quality(optimized.template)
        
        if optimized_structure > original_structure:
            structure_improvement = (optimized_structure - original_structure) / max(original_structure, 1)
            score += min(structure_improvement * 30, 15)  # Max 15 points for structure
        
        # Variable optimization
        if len(optimized.variables) <= len(original.variables):
            variable_improvement = max(0, len(original.variables) - len(optimized.variables))
            score += min(variable_improvement * 2, 10)  # Max 10 points for variables
        
        # Performance-based scoring
        if performance_data:
            if performance_data.accuracy and performance_data.accuracy > 0.8:
                score += 20
            if performance_data.response_time and performance_data.response_time < 3.0:
                score += 15
            if performance_data.user_satisfaction and performance_data.user_satisfaction > 0.8:
                score += 15
        
        return min(score / 100.0, 1.0)  # Normalize to 0-1
    
    def _calculate_sentence_complexity(self, content: str) -> float:
        """Calculate average sentence complexity."""
        sentences = [s.strip() for s in re.split(r'[.!?]+', content) if s.strip()]
        
        if not sentences:
            return 0.0
        
        total_complexity = 0
        for sentence in sentences:
            words = sentence.split()
            complexity = len(words) + sentence.count(',') * 2 + sentence.count(';') * 3
            total_complexity += complexity
        
        return total_complexity / len(sentences)
    
    def _count_vague_terms(self, content: str) -> int:
        """Count vague terms in content."""
        vague_terms = ['some', 'many', 'several', 'good', 'bad', 'nice', 'okay', 'stuff', 'things']
        count = 0
        content_lower = content.lower()
        
        for term in vague_terms:
            count += len(re.findall(rf'\b{term}\b', content_lower))
        
        return count
    
    def _count_instructions(self, content: str) -> int:
        """Count instruction indicators in content."""
        instruction_patterns = [
            r'\bplease\b', r'\byou should\b', r'\bmust\b', r'\bneed to\b',
            r'\brequired\b', r'\bensure\b', r'\bmake sure\b'
        ]
        
        count = 0
        for pattern in instruction_patterns:
            count += len(re.findall(pattern, content, re.IGNORECASE))
        
        return count
    
    def _analyze_structure_quality(self, content: str) -> float:
        """Analyze structural quality of content."""
        score = 0.0
        
        # Check for paragraph breaks
        if '\n\n' in content:
            score += 25
        
        # Check for logical flow indicators
        if re.search(r'\b(first|second|third|finally|next|then)\b', content, re.IGNORECASE):
            score += 25
        
        # Check for section headers or organization
        if re.search(r'^[A-Z][^.!?]*:$', content, re.MULTILINE):
            score += 25
        
        # Check for balanced sentence lengths
        sentences = [s.strip() for s in re.split(r'[.!?]+', content) if s.strip()]
        if sentences:
            lengths = [len(s.split()) for s in sentences]
            avg_length = sum(lengths) / len(lengths)
            variance = sum((l - avg_length) ** 2 for l in lengths) / len(lengths)
            if variance < 100:  # Low variance indicates balanced sentences
                score += 25
        
        return score
    
    def _remove_redundancy(self, content: str) -> str:
        """Remove redundant phrases and repetitive content."""
        # Remove common redundant phrases
        redundant_patterns = [
            (r'\bplease note that\b', ''),
            (r'\bit is important to note that\b', ''),
            (r'\bkeep in mind that\b', ''),
            (r'\bas mentioned before\b', ''),
            (r'\bin order to\b', 'to'),
            (r'\bdue to the fact that\b', 'because'),
            (r'\bin the event that\b', 'if')
        ]
        
        for pattern, replacement in redundant_patterns:
            content = re.sub(pattern, replacement, content, flags=re.IGNORECASE)
        
        return content.strip()
    
    def _simplify_constructions(self, content: str) -> str:
        """Simplify verbose constructions."""
        simplifications = [
            (r'\bmake use of\b', 'use'),
            (r'\bcarry out\b', 'perform'),
            (r'\btake into consideration\b', 'consider'),
            (r'\bcome to the conclusion\b', 'conclude'),
            (r'\bgive consideration to\b', 'consider'),
            (r'\bmake an attempt\b', 'try'),
            (r'\bput emphasis on\b', 'emphasize')
        ]
        
        for pattern, replacement in simplifications:
            content = re.sub(pattern, replacement, content, flags=re.IGNORECASE)
        
        return content
    
    def _remove_unnecessary_words(self, content: str) -> str:
        """Remove unnecessary filler words."""
        unnecessary_patterns = [
            r'\bvery\s+',
            r'\breally\s+', 
            r'\bquite\s+',
            r'\brather\s+',
            r'\bsomewhat\s+',
            r'\bactually\s+',
            r'\bobviously\s+',
            r'\bclearly\s+'
        ]
        
        for pattern in unnecessary_patterns:
            content = re.sub(pattern, '', content, flags=re.IGNORECASE)
        
        # Clean up extra spaces
        content = re.sub(r'\s+', ' ', content)
        
        return content.strip()
    
    def _break_long_sentences(self, content: str) -> Tuple[str, List[str]]:
        """Break down overly long sentences."""
        improvements = []
        sentences = re.split(r'([.!?]+)', content)
        
        result_parts = []
        for i in range(0, len(sentences), 2):
            if i + 1 < len(sentences):
                sentence = sentences[i].strip()
                punctuation = sentences[i + 1]
                
                if len(sentence.split()) > 25:  # Long sentence
                    # Try to break at logical points
                    if ', and ' in sentence:
                        parts = sentence.split(', and ')
                        result_parts.append(parts[0] + '.')
                        result_parts.append(' And ' + ', and '.join(parts[1:]) + punctuation)
                        improvements.append("Broke down long sentence for clarity")
                    elif ', but ' in sentence:
                        parts = sentence.split(', but ')
                        result_parts.append(parts[0] + '.')
                        result_parts.append(' But ' + ', but '.join(parts[1:]) + punctuation)
                        improvements.append("Broke down long sentence for clarity")
                    else:
                        result_parts.append(sentence + punctuation)
                else:
                    result_parts.append(sentence + punctuation)
            else:
                result_parts.append(sentences[i])
        
        return ''.join(result_parts), improvements
    
    def _improve_structure(self, content: str) -> Tuple[str, List[str]]:
        """Improve content structure with formatting."""
        improvements = []
        
        # Add bullet points for lists
        if re.search(r'(first|second|third|fourth|fifth)', content, re.IGNORECASE):
            # Convert numbered lists to bullet points
            content = re.sub(r'\b(first|second|third|fourth|fifth)\b,?\s*', r'• ', content, flags=re.IGNORECASE)
            improvements.append("Converted numbered list to bullet points for better readability")
        
        return content, improvements
    
    def _add_section_headers(self, content: str) -> Tuple[str, List[str]]:
        """Add section headers where appropriate."""
        improvements = []
        
        # Look for natural breaking points
        if 'instructions:' not in content.lower() and re.search(r'\b(follow these|do the following|steps)', content, re.IGNORECASE):
            content = re.sub(r'(\b(?:follow these|do the following|steps)[^.]*\.)', r'Instructions:\n\1', content, flags=re.IGNORECASE)
            improvements.append("Added section header for instructions")
        
        if 'examples:' not in content.lower() and re.search(r'\b(for example|such as)', content, re.IGNORECASE):
            content = re.sub(r'(\bfor example[^.]*\.)', r'Examples:\n\1', content, flags=re.IGNORECASE)
            improvements.append("Added section header for examples")
        
        return content, improvements
    
    def _group_similar_variables(self, variables: List[PromptVariable]) -> List[List[PromptVariable]]:
        """Group similar variables together."""
        groups = []
        used_variables = set()
        
        for variable in variables:
            if variable.name in used_variables:
                continue
                
            group = [variable]
            used_variables.add(variable.name)
            
            # Find similar variables
            for other_variable in variables:
                if (other_variable.name not in used_variables and 
                    self._variables_similar(variable, other_variable)):
                    group.append(other_variable)
                    used_variables.add(other_variable.name)
            
            groups.append(group)
        
        return groups
    
    def _variables_similar(self, var1: PromptVariable, var2: PromptVariable) -> bool:
        """Check if two variables are similar enough to be consolidated."""
        # Similar types
        if var1.type != var2.type:
            return False
        
        # Similar names (common prefixes/suffixes)
        name1_parts = var1.name.lower().split('_')
        name2_parts = var2.name.lower().split('_')
        
        common_parts = set(name1_parts) & set(name2_parts)
        
        return len(common_parts) > 0
    
    def _enhance_variable_description(self, variable: PromptVariable) -> str:
        """Enhance variable description with more detail."""
        if len(variable.description) >= 20:
            return variable.description
        
        enhanced = variable.description
        
        # Add type information
        if variable.type not in enhanced:
            enhanced += f" ({variable.type})"
        
        # Add constraints information
        if variable.constraints:
            constraint_info = []
            if "min_length" in variable.constraints:
                constraint_info.append(f"min {variable.constraints['min_length']} chars")
            if "max_length" in variable.constraints:
                constraint_info.append(f"max {variable.constraints['max_length']} chars")
            
            if constraint_info:
                enhanced += f" - {', '.join(constraint_info)}"
        
        # Add examples for common variable types
        if "name" in variable.name.lower():
            enhanced += " - e.g., 'John Smith'"
        elif "email" in variable.name.lower():
            enhanced += " - e.g., 'user@example.com'"
        elif "date" in variable.name.lower():
            enhanced += " - e.g., '2024-01-15'"
        
        return enhanced
    
    def _record_optimization(self, template_id: str, result: PromptOptimizationResult):
        """Record optimization in history."""
        history_entry = {
            "timestamp": datetime.now().isoformat(),
            "optimization_score": result.optimization_score,
            "improvements": result.improvements,
            "optimization_time": result.optimization_time
        }
        
        self.optimization_history[template_id].append(history_entry)
    
    def _load_optimization_patterns(self) -> Dict[str, Any]:
        """Load optimization patterns and best practices."""
        return {
            "length_patterns": {
                "target_ratio": 0.8,  # Target 80% of original length
                "min_reduction": 0.05  # At least 5% reduction
            },
            "clarity_patterns": {
                "max_sentence_length": 25,  # Words per sentence
                "max_complexity_score": 30
            },
            "structure_patterns": {
                "paragraph_break_threshold": 3,  # Lines before paragraph break
                "section_header_threshold": 5   # Sentences before section header
            }
        }
