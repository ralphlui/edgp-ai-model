"""
Specific types for Data Privacy & Compliance Agent operations.
"""

from enum import Enum
from typing import Any, Dict, List, Optional
from datetime import datetime
from pydantic import BaseModel, Field

from ..base import Severity, ConfidenceLevel, ComplianceRegulation, SensitivityLevel


class PrivacyRiskType(str, Enum):
    """Types of privacy risks."""
    PII_EXPOSURE = "pii_exposure"
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    DATA_MINIMIZATION_VIOLATION = "data_minimization_violation"
    CONSENT_VIOLATION = "consent_violation"
    CROSS_BORDER_TRANSFER = "cross_border_transfer"
    RETENTION_VIOLATION = "retention_violation"
    PURPOSE_LIMITATION_VIOLATION = "purpose_limitation_violation"


class ComplianceViolationType(str, Enum):
    """Types of compliance violations."""
    DATA_BREACH = "data_breach"
    MISSING_CONSENT = "missing_consent"
    IMPROPER_RETENTION = "improper_retention"
    UNAUTHORIZED_PROCESSING = "unauthorized_processing"
    MISSING_SAFEGUARDS = "missing_safeguards"
    INADEQUATE_DOCUMENTATION = "inadequate_documentation"
    VENDOR_COMPLIANCE_GAP = "vendor_compliance_gap"


class PIICategory(str, Enum):
    """Categories of Personally Identifiable Information."""
    DIRECT_IDENTIFIER = "direct_identifier"  # SSN, passport, etc.
    QUASI_IDENTIFIER = "quasi_identifier"    # age, zip code, etc.
    SENSITIVE_ATTRIBUTE = "sensitive_attribute"  # health, religion, etc.
    BEHAVIORAL_DATA = "behavioral_data"      # browsing, purchase history
    BIOMETRIC_DATA = "biometric_data"        # fingerprints, facial recognition
    LOCATION_DATA = "location_data"          # GPS, IP addresses


class PrivacyRisk(BaseModel):
    """Privacy risk assessment details."""
    risk_id: str
    risk_type: PrivacyRiskType
    description: str
    severity: Severity
    confidence: ConfidenceLevel
    
    # Affected data
    affected_fields: List[str]
    affected_records_count: int
    pii_categories: List[PIICategory] = Field(default_factory=list)
    sensitivity_level: SensitivityLevel
    
    # Legal context
    applicable_regulations: List[ComplianceRegulation] = Field(default_factory=list)
    legal_basis_required: bool = True
    consent_required: bool = False
    
    # Mitigation
    recommended_controls: List[str] = Field(default_factory=list)
    mitigation_options: List[str] = Field(default_factory=list)
    
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ComplianceCheck(BaseModel):
    """Compliance check configuration."""
    check_id: str
    check_name: str
    regulation: ComplianceRegulation
    check_type: str
    description: str
    target_data: List[str]
    check_logic: str
    pass_criteria: str
    fail_criteria: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ComplianceAssessment(BaseModel):
    """Result of compliance assessment."""
    assessment_id: str
    regulation: ComplianceRegulation
    overall_compliance_score: float = Field(..., ge=0.0, le=1.0)
    
    # Check results
    checks_passed: int
    checks_failed: int
    checks_total: int
    
    # Violations
    violations: List[str] = Field(default_factory=list)  # References to ComplianceViolation IDs
    
    # Risk assessment
    risk_level: Severity
    risk_factors: List[str] = Field(default_factory=list)
    
    # Recommendations
    immediate_actions: List[str] = Field(default_factory=list)
    long_term_recommendations: List[str] = Field(default_factory=list)
    
    # Next steps
    next_assessment_date: Optional[datetime] = None
    monitoring_requirements: List[str] = Field(default_factory=list)
    
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DataUsagePattern(BaseModel):
    """Data usage pattern for monitoring."""
    pattern_id: str
    user_id: str
    access_timestamp: datetime
    data_accessed: List[str]
    access_method: str
    purpose: str
    duration_minutes: Optional[int] = None
    data_volume: Optional[int] = None
    location: Optional[str] = None
    is_authorized: bool
    risk_indicators: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
