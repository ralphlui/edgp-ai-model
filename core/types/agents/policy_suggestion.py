"""
Specific types for Policy Suggestion Agent operations.
"""

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from ..base import Priority, ConfidenceLevel, ComplianceRegulation
from ..data import DataSchema, PolicyRecommendation


class PolicyType(str, Enum):
    """Types of policies that can be suggested."""
    ACCESS_CONTROL = "access_control"
    DATA_RETENTION = "data_retention"
    DATA_VALIDATION = "data_validation"
    PRIVACY_PROTECTION = "privacy_protection"
    SHARING_GOVERNANCE = "sharing_governance"
    QUALITY_STANDARDS = "quality_standards"
    SECURITY_CONTROLS = "security_controls"


class ValidationRule(BaseModel):
    """Data validation rule specification."""
    rule_id: str
    rule_name: str
    rule_type: str  # format, range, pattern, business_logic
    target_fields: List[str]
    validation_logic: str
    error_message: str
    severity: str
    is_mandatory: bool = True
    metadata: Dict[str, Any] = Field(default_factory=dict)


class PolicyContext(BaseModel):
    """Context information for policy suggestions."""
    business_domain: str
    industry_sector: str
    company_size: str
    geographic_regions: List[str] = Field(default_factory=list)
    regulatory_environment: List[ComplianceRegulation] = Field(default_factory=list)
    risk_tolerance: str = "medium"  # low, medium, high
    existing_frameworks: List[str] = Field(default_factory=list)


class PolicyGap(BaseModel):
    """Identified gap in existing policies."""
    gap_id: str
    gap_type: str
    description: str
    severity: str
    affected_areas: List[str]
    recommended_action: str
    compliance_impact: List[ComplianceRegulation] = Field(default_factory=list)
    business_risk: str
    implementation_complexity: str  # low, medium, high


class PolicyImplementationPlan(BaseModel):
    """Implementation plan for a suggested policy."""
    plan_id: str
    policy_id: str
    implementation_steps: List[Dict[str, Any]]
    estimated_timeline: str
    required_resources: List[str]
    success_criteria: List[str]
    risk_factors: List[str] = Field(default_factory=list)
    rollback_plan: Optional[str] = None
