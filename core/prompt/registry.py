"""
Prompt Registry

Centralized registry for managing prompt templates with versioning,
categorization, search capabilities, and lifecycle management.
"""

import json
import os
from typing import Dict, List, Optional, Any, Set, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import asdict
from pathlib import Path
import logging
import shutil
from collections import defaultdict

from .types import (
    PromptTemplate,
    PromptMetadata, 
    PromptCategory,
    PromptTechnique,
    PromptVariable
)

logger = logging.getLogger(__name__)


class PromptVersion:
    """Represents a versioned prompt template."""
    
    def __init__(
        self,
        template: PromptTemplate,
        version: str,
        created_at: datetime,
        created_by: str = "system",
        changelog: Optional[str] = None,
        parent_version: Optional[str] = None
    ):
        self.template = template
        self.version = version
        self.created_at = created_at
        self.created_by = created_by
        self.changelog = changelog or ""
        self.parent_version = parent_version
        self.is_active = True
        self.usage_count = 0
        self.last_used = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "template": self._template_to_dict(self.template),
            "version": self.version,
            "created_at": self.created_at.isoformat(),
            "created_by": self.created_by,
            "changelog": self.changelog,
            "parent_version": self.parent_version,
            "is_active": self.is_active,
            "usage_count": self.usage_count,
            "last_used": self.last_used.isoformat() if self.last_used else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PromptVersion':
        """Create from dictionary."""
        template = cls._template_from_dict(data["template"])
        version = cls(
            template=template,
            version=data["version"],
            created_at=datetime.fromisoformat(data["created_at"]),
            created_by=data.get("created_by", "system"),
            changelog=data.get("changelog", ""),
            parent_version=data.get("parent_version")
        )
        version.is_active = data.get("is_active", True)
        version.usage_count = data.get("usage_count", 0)
        version.last_used = datetime.fromisoformat(data["last_used"]) if data.get("last_used") else None
        return version
    
    @staticmethod
    def _template_to_dict(template: PromptTemplate) -> Dict[str, Any]:
        """Convert template to dictionary."""
        return {
            "template_id": template.template_id,
            "template": template.template,
            "variables": [asdict(var) for var in template.variables],
            "metadata": asdict(template.metadata),
            "examples": getattr(template, 'examples', []),
            "created_at": template.created_at.isoformat(),
            "updated_at": template.updated_at.isoformat() if template.updated_at else None
        }
    
    @staticmethod
    def _template_from_dict(data: Dict[str, Any]) -> PromptTemplate:
        """Create template from dictionary."""
        variables = [PromptVariable(**var_data) for var_data in data.get("variables", [])]
        metadata = PromptMetadata(**data["metadata"])
        
        template = PromptTemplate(
            template_id=data["template_id"],
            template=data["template"],
            variables=variables,
            metadata=metadata
        )
        
        template.created_at = datetime.fromisoformat(data["created_at"])
        if data.get("updated_at"):
            template.updated_at = datetime.fromisoformat(data["updated_at"])
        
        # Set examples if available
        if hasattr(template, 'examples'):
            template.examples = data.get("examples", [])
        
        return template


class PromptRegistry:
    """
    Centralized registry for managing prompt templates with:
    - Version control and history
    - Template organization and categorization
    - Search and discovery capabilities  
    - Usage tracking and analytics
    - Import/export functionality
    - Template lifecycle management
    """
    
    def __init__(self, registry_path: str = None, auto_backup: bool = True):
        self.registry_path = Path(registry_path or "data/prompt_registry")
        self.auto_backup = auto_backup
        
        # Ensure registry directory exists
        self.registry_path.mkdir(parents=True, exist_ok=True)
        
        # Internal storage
        self.templates: Dict[str, Dict[str, PromptVersion]] = defaultdict(dict)  # template_id -> version -> PromptVersion
        self.categories: Dict[PromptCategory, Set[str]] = defaultdict(set)
        self.techniques: Dict[PromptTechnique, Set[str]] = defaultdict(set)
        self.tags: Dict[str, Set[str]] = defaultdict(set)
        
        # Index for fast search
        self.search_index: Dict[str, Set[str]] = defaultdict(set)  # word -> set of template_ids
        
        # Load existing registry
        self._load_registry()
        
        logger.info("PromptRegistry initialized at %s", self.registry_path)
    
    def register_template(
        self,
        template: PromptTemplate,
        version: str = "1.0.0",
        created_by: str = "system",
        changelog: Optional[str] = None,
        parent_version: Optional[str] = None
    ) -> str:
        """
        Register a new template or version.
        
        Args:
            template: PromptTemplate to register
            version: Version string (semantic versioning recommended)
            created_by: User who created this version
            changelog: Description of changes
            parent_version: Previous version this is based on
            
        Returns:
            Template ID of registered template
        """
        # Validate template
        if not template.template_id:
            raise ValueError("Template must have an ID")
        
        if not template.template.strip():
            raise ValueError("Template content cannot be empty")
        
        # Create version object
        prompt_version = PromptVersion(
            template=template,
            version=version,
            created_at=datetime.now(),
            created_by=created_by,
            changelog=changelog,
            parent_version=parent_version
        )
        
        # Register in memory
        template_id = template.template_id
        
        # Deactivate previous versions if this is a new major version
        if template_id in self.templates:
            if self._is_major_version_change(version, list(self.templates[template_id].keys())):
                for old_version in self.templates[template_id].values():
                    old_version.is_active = False
        
        self.templates[template_id][version] = prompt_version
        
        # Update indexes
        self._update_indexes(template)
        
        # Persist to disk
        self._save_template(template_id, version, prompt_version)
        
        # Auto-backup if enabled
        if self.auto_backup:
            self._create_backup()
        
        logger.info("Registered template %s version %s", template_id, version)
        return template_id
    
    def get_template(
        self,
        template_id: str,
        version: Optional[str] = None
    ) -> Optional[PromptTemplate]:
        """
        Get a template by ID and optionally version.
        
        Args:
            template_id: Template identifier
            version: Specific version (if None, returns latest active version)
            
        Returns:
            PromptTemplate if found, None otherwise
        """
        if template_id not in self.templates:
            return None
        
        template_versions = self.templates[template_id]
        
        if version:
            if version in template_versions:
                prompt_version = template_versions[version]
                # Update usage statistics
                prompt_version.usage_count += 1
                prompt_version.last_used = datetime.now()
                return prompt_version.template
            return None
        else:
            # Return latest active version
            active_versions = [
                (v, pv) for v, pv in template_versions.items() 
                if pv.is_active
            ]
            
            if not active_versions:
                return None
            
            # Sort by version (semantic versioning)
            latest_version = max(active_versions, key=lambda x: self._parse_version(x[0]))
            prompt_version = latest_version[1]
            
            # Update usage statistics
            prompt_version.usage_count += 1
            prompt_version.last_used = datetime.now()
            
            return prompt_version.template
    
    def search_templates(
        self,
        query: Optional[str] = None,
        category: Optional[PromptCategory] = None,
        technique: Optional[PromptTechnique] = None,
        tags: Optional[List[str]] = None,
        created_by: Optional[str] = None,
        date_range: Optional[Tuple[datetime, datetime]] = None
    ) -> List[PromptTemplate]:
        """
        Search templates with various filters.
        
        Args:
            query: Text search query
            category: Filter by category
            technique: Filter by technique
            tags: Filter by tags (all must match)
            created_by: Filter by creator
            date_range: Filter by creation date range
            
        Returns:
            List of matching templates (latest active versions)
        """
        matching_template_ids = set()
        
        # Text search
        if query:
            query_words = query.lower().split()
            for word in query_words:
                if word in self.search_index:
                    if not matching_template_ids:
                        matching_template_ids = self.search_index[word].copy()
                    else:
                        matching_template_ids &= self.search_index[word]
        else:
            # Start with all templates if no query
            matching_template_ids = set(self.templates.keys())
        
        # Filter by category
        if category and category in self.categories:
            matching_template_ids &= self.categories[category]
        
        # Filter by technique
        if technique and technique in self.techniques:
            matching_template_ids &= self.techniques[technique]
        
        # Filter by tags
        if tags:
            for tag in tags:
                if tag in self.tags:
                    matching_template_ids &= self.tags[tag]
                else:
                    matching_template_ids = set()  # Tag not found, no matches
                    break
        
        # Get templates and apply additional filters
        results = []
        for template_id in matching_template_ids:
            template = self.get_template(template_id)  # Gets latest active version
            if not template:
                continue
            
            # Filter by created_by
            if created_by:
                # Get the version info for created_by check
                latest_version = self._get_latest_active_version(template_id)
                if not latest_version or latest_version.created_by != created_by:
                    continue
            
            # Filter by date range
            if date_range:
                start_date, end_date = date_range
                if not (start_date <= template.created_at <= end_date):
                    continue
            
            results.append(template)
        
        # Sort by relevance (usage count + recency)
        return sorted(results, key=self._calculate_relevance_score, reverse=True)
    
    def list_versions(self, template_id: str) -> List[Tuple[str, PromptVersion]]:
        """
        List all versions of a template.
        
        Args:
            template_id: Template identifier
            
        Returns:
            List of (version, PromptVersion) tuples sorted by version
        """
        if template_id not in self.templates:
            return []
        
        versions = list(self.templates[template_id].items())
        return sorted(versions, key=lambda x: self._parse_version(x[0]), reverse=True)
    
    def deactivate_template(self, template_id: str, version: Optional[str] = None):
        """
        Deactivate a template version (soft delete).
        
        Args:
            template_id: Template identifier
            version: Specific version (if None, deactivates all versions)
        """
        if template_id not in self.templates:
            raise ValueError(f"Template {template_id} not found")
        
        if version:
            if version in self.templates[template_id]:
                self.templates[template_id][version].is_active = False
                logger.info("Deactivated template %s version %s", template_id, version)
            else:
                raise ValueError(f"Version {version} not found for template {template_id}")
        else:
            for prompt_version in self.templates[template_id].values():
                prompt_version.is_active = False
            logger.info("Deactivated all versions of template %s", template_id)
        
        # Update indexes
        self._rebuild_indexes()
        
        # Persist changes
        self._save_registry()
    
    def delete_template(self, template_id: str, version: Optional[str] = None):
        """
        Permanently delete a template version.
        
        Args:
            template_id: Template identifier
            version: Specific version (if None, deletes entire template)
        """
        if template_id not in self.templates:
            raise ValueError(f"Template {template_id} not found")
        
        if version:
            if version in self.templates[template_id]:
                del self.templates[template_id][version]
                logger.info("Deleted template %s version %s", template_id, version)
                
                # If no versions left, remove template entirely
                if not self.templates[template_id]:
                    del self.templates[template_id]
            else:
                raise ValueError(f"Version {version} not found for template {template_id}")
        else:
            del self.templates[template_id]
            logger.info("Deleted entire template %s", template_id)
        
        # Update indexes
        self._rebuild_indexes()
        
        # Persist changes
        self._save_registry()
        
        # Remove from disk
        self._delete_template_files(template_id, version)
    
    def export_templates(
        self,
        output_path: str,
        template_ids: Optional[List[str]] = None,
        include_inactive: bool = False
    ) -> str:
        """
        Export templates to a file.
        
        Args:
            output_path: Path for export file
            template_ids: Specific templates to export (None for all)
            include_inactive: Whether to include inactive versions
            
        Returns:
            Path to exported file
        """
        export_data = {
            "export_timestamp": datetime.now().isoformat(),
            "templates": {}
        }
        
        templates_to_export = template_ids or list(self.templates.keys())
        
        for template_id in templates_to_export:
            if template_id in self.templates:
                template_versions = {}
                for version, prompt_version in self.templates[template_id].items():
                    if include_inactive or prompt_version.is_active:
                        template_versions[version] = prompt_version.to_dict()
                
                if template_versions:
                    export_data["templates"][template_id] = template_versions
        
        # Write to file
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        logger.info("Exported %d templates to %s", len(export_data["templates"]), output_path)
        return str(output_file)
    
    def import_templates(
        self,
        import_path: str,
        overwrite_existing: bool = False,
        import_inactive: bool = False
    ) -> List[str]:
        """
        Import templates from a file.
        
        Args:
            import_path: Path to import file
            overwrite_existing: Whether to overwrite existing templates
            import_inactive: Whether to import inactive versions
            
        Returns:
            List of imported template IDs
        """
        with open(import_path, 'r', encoding='utf-8') as f:
            import_data = json.load(f)
        
        imported_templates = []
        
        for template_id, versions_data in import_data.get("templates", {}).items():
            for version, version_data in versions_data.items():
                try:
                    prompt_version = PromptVersion.from_dict(version_data)
                    
                    # Skip inactive versions if not importing them
                    if not import_inactive and not prompt_version.is_active:
                        continue
                    
                    # Check if already exists
                    if (template_id in self.templates and 
                        version in self.templates[template_id] and 
                        not overwrite_existing):
                        logger.warning("Skipping existing template %s version %s", template_id, version)
                        continue
                    
                    # Import the template
                    self.templates[template_id][version] = prompt_version
                    
                    # Update indexes
                    self._update_indexes(prompt_version.template)
                    
                    if template_id not in imported_templates:
                        imported_templates.append(template_id)
                    
                    logger.debug("Imported template %s version %s", template_id, version)
                    
                except Exception as e:
                    logger.error("Failed to import template %s version %s: %s", template_id, version, e)
        
        # Persist changes
        self._save_registry()
        
        logger.info("Imported %d templates from %s", len(imported_templates), import_path)
        return imported_templates
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get registry statistics."""
        total_templates = len(self.templates)
        total_versions = sum(len(versions) for versions in self.templates.values())
        active_templates = len([
            tid for tid, versions in self.templates.items()
            if any(v.is_active for v in versions.values())
        ])
        
        # Category distribution
        category_counts = {cat.value: len(templates) for cat, templates in self.categories.items()}
        
        # Technique distribution
        technique_counts = {tech.value: len(templates) for tech, templates in self.techniques.items()}
        
        # Usage statistics
        total_usage = sum(
            sum(v.usage_count for v in versions.values())
            for versions in self.templates.values()
        )
        
        # Most used templates
        template_usage = []
        for template_id, versions in self.templates.items():
            total_template_usage = sum(v.usage_count for v in versions.values())
            if total_template_usage > 0:
                template_usage.append((template_id, total_template_usage))
        
        most_used = sorted(template_usage, key=lambda x: x[1], reverse=True)[:10]
        
        return {
            "total_templates": total_templates,
            "total_versions": total_versions,
            "active_templates": active_templates,
            "total_usage": total_usage,
            "category_distribution": category_counts,
            "technique_distribution": technique_counts,
            "most_used_templates": most_used,
            "registry_size_mb": self._calculate_registry_size()
        }
    
    def cleanup_unused_templates(
        self,
        max_age_days: int = 90,
        min_usage_count: int = 0
    ) -> List[str]:
        """
        Clean up unused or old templates.
        
        Args:
            max_age_days: Maximum age in days for unused templates
            min_usage_count: Minimum usage count to keep template
            
        Returns:
            List of cleaned up template IDs
        """
        cutoff_date = datetime.now() - timedelta(days=max_age_days)
        cleaned_templates = []
        
        for template_id, versions in list(self.templates.items()):
            versions_to_remove = []
            
            for version, prompt_version in versions.items():
                # Check age and usage
                if (prompt_version.created_at < cutoff_date and 
                    prompt_version.usage_count <= min_usage_count):
                    versions_to_remove.append(version)
            
            # Remove old versions
            for version in versions_to_remove:
                del versions[version]
                logger.debug("Cleaned up template %s version %s", template_id, version)
            
            # If no versions left, remove entire template
            if not versions:
                del self.templates[template_id]
                cleaned_templates.append(template_id)
                logger.info("Cleaned up entire template %s", template_id)
        
        # Rebuild indexes
        self._rebuild_indexes()
        
        # Persist changes
        self._save_registry()
        
        logger.info("Cleaned up %d templates", len(cleaned_templates))
        return cleaned_templates
    
    def _update_indexes(self, template: PromptTemplate):
        """Update search indexes for a template."""
        template_id = template.template_id
        
        # Category index
        self.categories[template.metadata.category].add(template_id)
        
        # Technique index
        self.techniques[template.metadata.technique].add(template_id)
        
        # Tags index
        for tag in template.metadata.tags:
            self.tags[tag].add(template_id)
        
        # Search index (words in template content, description, tags)
        words = set()
        
        # Extract words from template content
        words.update(template.template.lower().split())
        
        # Extract words from metadata
        if template.metadata.description:
            words.update(template.metadata.description.lower().split())
        
        words.update(tag.lower() for tag in template.metadata.tags)
        
        # Add variable names and descriptions
        for variable in template.variables:
            words.add(variable.name.lower())
            if variable.description:
                words.update(variable.description.lower().split())
        
        # Update search index
        for word in words:
            # Clean word (remove punctuation)
            clean_word = ''.join(c for c in word if c.isalnum())
            if clean_word and len(clean_word) > 2:  # Skip very short words
                self.search_index[clean_word].add(template_id)
    
    def _rebuild_indexes(self):
        """Rebuild all indexes from scratch."""
        # Clear existing indexes
        self.categories.clear()
        self.techniques.clear()
        self.tags.clear()
        self.search_index.clear()
        
        # Rebuild from active templates
        for template_id, versions in self.templates.items():
            # Use latest active version for indexing
            active_versions = [v for v in versions.values() if v.is_active]
            if active_versions:
                latest_version = max(active_versions, key=lambda v: self._parse_version(v.version))
                self._update_indexes(latest_version.template)
    
    def _get_latest_active_version(self, template_id: str) -> Optional[PromptVersion]:
        """Get latest active version of a template."""
        if template_id not in self.templates:
            return None
        
        active_versions = [
            (v, pv) for v, pv in self.templates[template_id].items()
            if pv.is_active
        ]
        
        if not active_versions:
            return None
        
        return max(active_versions, key=lambda x: self._parse_version(x[0]))[1]
    
    def _calculate_relevance_score(self, template: PromptTemplate) -> float:
        """Calculate relevance score for search results."""
        # Get usage statistics
        template_id = template.template_id
        latest_version = self._get_latest_active_version(template_id)
        
        if not latest_version:
            return 0.0
        
        # Base score from usage count
        usage_score = min(latest_version.usage_count / 100.0, 1.0)  # Normalize to 0-1
        
        # Recency score (more recent = higher score)
        if latest_version.last_used:
            days_since_last_use = (datetime.now() - latest_version.last_used).days
            recency_score = max(0, 1.0 - (days_since_last_use / 365.0))  # Decay over a year
        else:
            recency_score = 0.0
        
        # Combine scores
        return usage_score * 0.7 + recency_score * 0.3
    
    def _parse_version(self, version: str) -> Tuple[int, int, int]:
        """Parse semantic version string into tuple for comparison."""
        try:
            parts = version.split('.')
            major = int(parts[0]) if len(parts) > 0 else 0
            minor = int(parts[1]) if len(parts) > 1 else 0
            patch = int(parts[2]) if len(parts) > 2 else 0
            return (major, minor, patch)
        except (ValueError, IndexError):
            # Fallback for non-semantic versions
            return (0, 0, 0)
    
    def _is_major_version_change(self, new_version: str, existing_versions: List[str]) -> bool:
        """Check if new version is a major version change."""
        new_major = self._parse_version(new_version)[0]
        existing_majors = [self._parse_version(v)[0] for v in existing_versions]
        
        return new_major > max(existing_majors) if existing_majors else False
    
    def _load_registry(self):
        """Load registry from disk."""
        registry_file = self.registry_path / "registry.json"
        
        if not registry_file.exists():
            logger.info("No existing registry found, starting fresh")
            return
        
        try:
            with open(registry_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Load templates
            for template_id, versions_data in data.get("templates", {}).items():
                for version, version_data in versions_data.items():
                    prompt_version = PromptVersion.from_dict(version_data)
                    self.templates[template_id][version] = prompt_version
            
            # Rebuild indexes
            self._rebuild_indexes()
            
            logger.info("Loaded %d templates from registry", len(self.templates))
            
        except Exception as e:
            logger.error("Failed to load registry: %s", e)
            # Create backup of corrupted file
            shutil.copy2(registry_file, registry_file.with_suffix('.corrupted'))
    
    def _save_registry(self):
        """Save entire registry to disk."""
        registry_file = self.registry_path / "registry.json"
        
        data = {
            "saved_at": datetime.now().isoformat(),
            "templates": {}
        }
        
        for template_id, versions in self.templates.items():
            version_data = {}
            for version, prompt_version in versions.items():
                version_data[version] = prompt_version.to_dict()
            data["templates"][template_id] = version_data
        
        # Write to temporary file first, then rename (atomic operation)
        temp_file = registry_file.with_suffix('.tmp')
        
        try:
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            # Rename to final file
            temp_file.replace(registry_file)
            
        except Exception as e:
            logger.error("Failed to save registry: %s", e)
            if temp_file.exists():
                temp_file.unlink()
            raise
    
    def _save_template(self, template_id: str, version: str, prompt_version: PromptVersion):
        """Save individual template version to disk."""
        template_dir = self.registry_path / "templates" / template_id
        template_dir.mkdir(parents=True, exist_ok=True)
        
        version_file = template_dir / f"{version}.json"
        
        with open(version_file, 'w', encoding='utf-8') as f:
            json.dump(prompt_version.to_dict(), f, indent=2, ensure_ascii=False)
    
    def _delete_template_files(self, template_id: str, version: Optional[str] = None):
        """Delete template files from disk."""
        template_dir = self.registry_path / "templates" / template_id
        
        if version:
            version_file = template_dir / f"{version}.json"
            if version_file.exists():
                version_file.unlink()
        else:
            if template_dir.exists():
                shutil.rmtree(template_dir)
    
    def _create_backup(self):
        """Create backup of registry."""
        backup_dir = self.registry_path / "backups"
        backup_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = backup_dir / f"registry_backup_{timestamp}.json"
        
        registry_file = self.registry_path / "registry.json"
        if registry_file.exists():
            shutil.copy2(registry_file, backup_file)
            
            # Clean old backups (keep last 10)
            backup_files = sorted(backup_dir.glob("registry_backup_*.json"))
            for old_backup in backup_files[:-10]:
                old_backup.unlink()
    
    def _calculate_registry_size(self) -> float:
        """Calculate registry size in MB."""
        total_size = 0
        
        for root, dirs, files in os.walk(self.registry_path):
            for file in files:
                file_path = Path(root) / file
                total_size += file_path.stat().st_size
        
        return total_size / (1024 * 1024)  # Convert to MB
