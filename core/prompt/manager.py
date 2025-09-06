"""
Prompt Manager

Central management system for all prompt templates, providing functionality for
template registration, retrieval, caching, and optimization.
"""

import os
import json
import yaml
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict
import hashlib

from .types import (
    PromptType,
    PromptCategory,
    PromptTechnique,
    PromptMetadata,
    PromptVariable,
    PromptContext,
    PromptResult,
    PromptConfig,
    PromptStatus,
    PromptPriority
)
from .templates import (
    PromptTemplate,
    SystemPromptTemplate,
    AgentPromptTemplate,
    TaskPromptTemplate,
    ChainOfThoughtTemplate,
    FewShotTemplate
)

logger = logging.getLogger(__name__)


class PromptCache:
    """Simple in-memory cache for rendered prompts."""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache = {}
        self.access_times = {}
    
    def get(self, key: str) -> Optional[PromptResult]:
        """Get cached prompt result."""
        if key not in self.cache:
            return None
        
        # Check TTL
        if datetime.utcnow() - self.access_times[key] > timedelta(seconds=self.ttl_seconds):
            self.remove(key)
            return None
        
        self.access_times[key] = datetime.utcnow()
        return self.cache[key]
    
    def put(self, key: str, result: PromptResult):
        """Cache prompt result."""
        # Evict oldest if at capacity
        if len(self.cache) >= self.max_size:
            oldest_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
            self.remove(oldest_key)
        
        self.cache[key] = result
        self.access_times[key] = datetime.utcnow()
    
    def remove(self, key: str):
        """Remove cached result."""
        self.cache.pop(key, None)
        self.access_times.pop(key, None)
    
    def clear(self):
        """Clear all cached results."""
        self.cache.clear()
        self.access_times.clear()
    
    def get_cache_key(self, template_id: str, variables: Dict[str, Any], context: PromptContext) -> str:
        """Generate cache key for template + variables + context."""
        # Create deterministic hash of inputs
        data = {
            "template_id": template_id,
            "variables": variables,
            "context": context.dict() if context else {}
        }
        data_str = json.dumps(data, sort_keys=True)
        return hashlib.md5(data_str.encode()).hexdigest()


class PromptManager:
    """
    Central manager for prompt templates with support for:
    - Template registration and retrieval
    - Template inheritance and composition
    - Caching and performance optimization
    - A/B testing and experimentation
    - Version management
    """
    
    def __init__(self, config: PromptConfig = None):
        self.config = config or PromptConfig()
        
        # Template storage
        self.templates: Dict[str, PromptTemplate] = {}
        self.templates_by_category: Dict[PromptCategory, List[str]] = defaultdict(list)
        self.templates_by_agent: Dict[str, List[str]] = defaultdict(list)
        
        # Caching
        self.cache = PromptCache() if self.config.enable_caching else None
        
        # Performance tracking
        self.usage_stats = defaultdict(int)
        self.performance_metrics = defaultdict(list)
        
        # A/B testing
        self.ab_test_groups = {}
        
        logger.info("PromptManager initialized with caching=%s", self.config.enable_caching)
    
    def register_template(self, template: PromptTemplate) -> str:
        """
        Register a new prompt template.
        
        Args:
            template: PromptTemplate instance to register
            
        Returns:
            str: Template ID
        """
        template_id = template.metadata.id
        
        # Check for existing template
        if template_id in self.templates:
            logger.warning("Template %s already exists, updating", template_id)
        
        # Store template
        self.templates[template_id] = template
        
        # Update indexes
        self.templates_by_category[template.metadata.category].append(template_id)
        
        # Index by agent type if applicable
        if isinstance(template, AgentPromptTemplate) and template.agent_type:
            self.templates_by_agent[template.agent_type].append(template_id)
        
        logger.info("Registered template: %s (%s)", template.metadata.name, template_id)
        return template_id
    
    def get_template(self, template_id: str) -> Optional[PromptTemplate]:
        """Get template by ID."""
        return self.templates.get(template_id)
    
    def list_templates(
        self,
        category: PromptCategory = None,
        agent_type: str = None,
        status: PromptStatus = None,
        technique: PromptTechnique = None
    ) -> List[PromptTemplate]:
        """List templates with optional filtering."""
        templates = list(self.templates.values())
        
        if category:
            templates = [t for t in templates if t.metadata.category == category]
        
        if agent_type:
            agent_template_ids = self.templates_by_agent.get(agent_type, [])
            templates = [t for t in templates if t.metadata.id in agent_template_ids]
        
        if status:
            templates = [t for t in templates if t.metadata.status == status]
        
        if technique:
            templates = [t for t in templates if t.metadata.technique == technique]
        
        return templates
    
    def find_template(
        self,
        name: str = None,
        category: PromptCategory = None,
        agent_type: str = None,
        capability: str = None,
        context: PromptContext = None
    ) -> Optional[PromptTemplate]:
        """
        Find the best matching template for given criteria.
        
        Uses intelligent matching to select the most appropriate template.
        """
        candidates = self.list_templates(category=category, agent_type=agent_type)
        
        if not candidates:
            return None
        
        if name:
            # Exact name match first
            for template in candidates:
                if template.metadata.name == name:
                    return template
        
        # Score candidates based on context
        scored_candidates = []
        for template in candidates:
            score = self._score_template_match(template, context, capability)
            scored_candidates.append((score, template))
        
        # Return highest scoring template
        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        return scored_candidates[0][1] if scored_candidates else None
    
    def render_prompt(
        self,
        template_id: str = None,
        template_name: str = None,
        variables: Dict[str, Any] = None,
        context: PromptContext = None,
        use_cache: bool = None
    ) -> PromptResult:
        """
        Render a prompt using specified template.
        
        Args:
            template_id: Specific template ID to use
            template_name: Template name to search for
            variables: Variables to substitute in template
            context: Context for prompt generation
            use_cache: Whether to use caching (overrides config)
            
        Returns:
            PromptResult with generated prompt
        """
        variables = variables or {}
        context = context or PromptContext()
        use_cache = use_cache if use_cache is not None else self.config.enable_caching
        
        # Get template
        template = None
        if template_id:
            template = self.get_template(template_id)
        elif template_name:
            template = self.find_template(name=template_name, context=context)
        
        if not template:
            return PromptResult(
                prompt="",
                formatted_prompt="",
                template_id=template_id or "unknown",
                template_version="unknown",
                errors=["Template not found"],
                success=False
            )
        
        # Check cache
        if use_cache and self.cache:
            cache_key = self.cache.get_cache_key(template.metadata.id, variables, context)
            cached_result = self.cache.get(cache_key)
            if cached_result:
                logger.debug("Cache hit for template %s", template.metadata.id)
                return cached_result
        
        # Handle A/B testing
        if self.config.enable_ab_testing:
            template = self._get_ab_test_template(template, context)
        
        # Render template
        start_time = datetime.utcnow()
        try:
            result = template.render(variables, context)
            
            # Track performance
            render_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            self._track_performance(template.metadata.id, render_time, result.success)
            
            # Cache result
            if use_cache and self.cache and result.success:
                cache_key = self.cache.get_cache_key(template.metadata.id, variables, context)
                self.cache.put(cache_key, result)
            
            return result
            
        except Exception as e:
            logger.error("Error rendering template %s: %s", template.metadata.id, str(e))
            return PromptResult(
                prompt="",
                formatted_prompt="",
                template_id=template.metadata.id,
                template_version=template.metadata.version,
                errors=[str(e)],
                success=False
            )
    
    def create_system_prompt(
        self,
        name: str,
        template: str,
        description: str,
        author: str,
        variables: List[PromptVariable] = None,
        persona_definition: str = None,
        capabilities: str = None,
        guidelines: List[str] = None
    ) -> str:
        """Create and register a new system prompt template."""
        metadata = PromptMetadata(
            name=name,
            description=description,
            category=PromptCategory.SYSTEM_CORE,
            technique=PromptTechnique.INSTRUCTION_FOLLOWING,
            author=author,
            status=PromptStatus.DRAFT
        )
        
        system_template = SystemPromptTemplate(
            template=template,
            metadata=metadata,
            variables=variables or [],
            persona_definition=persona_definition,
            capability_description=capabilities,
            behavioral_guidelines=guidelines or []
        )
        
        return self.register_template(system_template)
    
    def create_agent_prompt(
        self,
        name: str,
        template: str,
        description: str,
        author: str,
        agent_type: str,
        capabilities: List[str] = None,
        variables: List[PromptVariable] = None,
        specialized_instructions: Dict[str, str] = None
    ) -> str:
        """Create and register a new agent-specific prompt template."""
        metadata = PromptMetadata(
            name=name,
            description=description,
            category=PromptCategory.AGENT_SPECIFIC,
            technique=PromptTechnique.ROLE_PLAYING,
            author=author,
            status=PromptStatus.DRAFT
        )
        
        agent_template = AgentPromptTemplate(
            template=template,
            metadata=metadata,
            variables=variables or [],
            agent_type=agent_type,
            capabilities=capabilities or [],
            specialized_instructions=specialized_instructions or {}
        )
        
        return self.register_template(agent_template)
    
    def create_task_prompt(
        self,
        name: str,
        template: str,
        description: str,
        author: str,
        task_type: str,
        technique: PromptTechnique = PromptTechnique.INSTRUCTION_FOLLOWING,
        variables: List[PromptVariable] = None,
        examples: List[Dict[str, Any]] = None,
        input_format: str = None,
        output_format: str = None
    ) -> str:
        """Create and register a new task-specific prompt template."""
        metadata = PromptMetadata(
            name=name,
            description=description,
            category=PromptCategory.TASK_SPECIFIC,
            technique=technique,
            author=author,
            status=PromptStatus.DRAFT
        )
        
        # Choose appropriate template class based on technique
        if technique == PromptTechnique.CHAIN_OF_THOUGHT:
            task_template = ChainOfThoughtTemplate(
                template=template,
                metadata=metadata,
                variables=variables or []
            )
        elif technique == PromptTechnique.FEW_SHOT:
            task_template = FewShotTemplate(
                template=template,
                metadata=metadata,
                variables=variables or [],
                examples=examples or []
            )
        else:
            task_template = TaskPromptTemplate(
                template=template,
                metadata=metadata,
                variables=variables or [],
                task_type=task_type,
                input_format=input_format,
                output_format=output_format,
                examples=examples or []
            )
        
        return self.register_template(task_template)
    
    def load_templates_from_directory(self, directory_path: str) -> int:
        """Load templates from YAML/JSON files in a directory."""
        directory = Path(directory_path)
        if not directory.exists():
            logger.error("Directory %s does not exist", directory_path)
            return 0
        
        loaded_count = 0
        for file_path in directory.glob("*.yaml"):
            try:
                with open(file_path, 'r') as f:
                    data = yaml.safe_load(f)
                    self._load_template_from_data(data)
                    loaded_count += 1
            except Exception as e:
                logger.error("Error loading template from %s: %s", file_path, str(e))
        
        for file_path in directory.glob("*.json"):
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    self._load_template_from_data(data)
                    loaded_count += 1
            except Exception as e:
                logger.error("Error loading template from %s: %s", file_path, str(e))
        
        logger.info("Loaded %d templates from %s", loaded_count, directory_path)
        return loaded_count
    
    def export_template(self, template_id: str, file_path: str) -> bool:
        """Export template to YAML file."""
        template = self.get_template(template_id)
        if not template:
            return False
        
        try:
            data = template.to_dict()
            with open(file_path, 'w') as f:
                yaml.dump(data, f, default_flow_style=False)
            return True
        except Exception as e:
            logger.error("Error exporting template %s: %s", template_id, str(e))
            return False
    
    def get_usage_statistics(self) -> Dict[str, Any]:
        """Get usage statistics for all templates."""
        stats = {
            "total_templates": len(self.templates),
            "templates_by_category": dict(self.templates_by_category),
            "usage_counts": dict(self.usage_stats),
            "performance_metrics": dict(self.performance_metrics)
        }
        
        # Calculate aggregate metrics
        if self.performance_metrics:
            all_times = []
            for times in self.performance_metrics.values():
                all_times.extend(times)
            
            if all_times:
                stats["avg_render_time_ms"] = sum(all_times) / len(all_times)
                stats["max_render_time_ms"] = max(all_times)
                stats["min_render_time_ms"] = min(all_times)
        
        return stats
    
    def _score_template_match(
        self,
        template: PromptTemplate,
        context: PromptContext,
        capability: str = None
    ) -> float:
        """Score how well a template matches the given context."""
        score = 0.0
        
        # Base score for active templates
        if template.metadata.status == PromptStatus.ACTIVE:
            score += 10.0
        elif template.metadata.status == PromptStatus.TESTING:
            score += 5.0
        
        # Priority scoring
        if template.metadata.priority == PromptPriority.CRITICAL:
            score += 8.0
        elif template.metadata.priority == PromptPriority.HIGH:
            score += 5.0
        
        # Context matching
        if context and context.agent_type:
            if isinstance(template, AgentPromptTemplate) and template.agent_type == context.agent_type:
                score += 15.0
        
        if capability:
            if isinstance(template, AgentPromptTemplate) and capability in template.capabilities:
                score += 10.0
        
        # Usage-based scoring (popular templates get slight boost)
        usage_count = self.usage_stats.get(template.metadata.id, 0)
        score += min(usage_count * 0.1, 5.0)  # Cap at 5 points
        
        return score
    
    def _get_ab_test_template(self, template: PromptTemplate, context: PromptContext) -> PromptTemplate:
        """Get A/B test variant if applicable."""
        # Simple A/B testing implementation
        if template.metadata.a_b_test_group:
            # Use context session_id to determine variant
            if context.session_id:
                variant_hash = hash(context.session_id) % 2
                if variant_hash == 1:
                    # Return variant if available
                    variant_id = f"{template.metadata.id}_variant"
                    variant = self.get_template(variant_id)
                    if variant:
                        return variant
        
        return template
    
    def _track_performance(self, template_id: str, render_time_ms: float, success: bool):
        """Track performance metrics for template."""
        self.usage_stats[template_id] += 1
        if success:
            self.performance_metrics[template_id].append(render_time_ms)
        
        # Keep only recent metrics (last 100 renders)
        if len(self.performance_metrics[template_id]) > 100:
            self.performance_metrics[template_id] = self.performance_metrics[template_id][-100:]
    
    def _load_template_from_data(self, data: Dict[str, Any]):
        """Load template from dictionary data."""
        template_type = data.get("type", "task")
        
        if template_type == "system":
            template = SystemPromptTemplate.from_dict(data)
        elif template_type == "agent":
            template = AgentPromptTemplate.from_dict(data)
        else:
            template = TaskPromptTemplate.from_dict(data)
        
        self.register_template(template)
