"""
Unit tests for core.types.data module.
"""
import pytest
from unittest.mock import Mock
from datetime import datetime
from typing import Dict, Any, List
from core.types.data import (
    DataSource, DataField, DataSchema, DataRecord, DataIssue, 
    QualityMetrics, ComplianceViolation, RemediationTask,
    PolicyRecommendation, ProcessingRequest, AnalyticsVisualization,
    DataQualityIssue, DataQualityMetrics, ColumnMetadata, DatasetMetadata,
    DataProfile, DataValidationRule, DataTransformation, DataLineage, PolicyRule
)
from core.types.base import (
    Severity, Priority, SensitivityLevel, DataType, QualityDimension,
    DataSourceType, DataFormat, PolicyType, ComplianceStatus, DataClassification
)


class TestDataEnums:
    """Test data-related enum classes."""
    
    def test_data_source_enum(self):
        """Test DataSource enum values."""
        assert DataSource.DATABASE.value == "database"
        assert DataSource.FILE.value == "file"
        assert DataSource.API.value == "api"
        assert DataSource.STREAM.value == "stream"
        assert DataSource.CLOUD_STORAGE.value == "cloud_storage"
    
    def test_data_format_enum(self):
        """Test DataFormat enum values."""
        assert DataFormat.CSV.value == "csv"
        assert DataFormat.JSON.value == "json"
        assert DataFormat.PARQUET.value == "parquet"
        assert DataFormat.XML.value == "xml"
        assert DataFormat.AVRO.value == "avro"
    
    def test_quality_dimension_enum(self):
        """Test QualityDimension enum values."""
        assert QualityDimension.COMPLETENESS.value == "completeness"
        assert QualityDimension.ACCURACY.value == "accuracy"
        assert QualityDimension.CONSISTENCY.value == "consistency"
        assert QualityDimension.VALIDITY.value == "validity"
        assert QualityDimension.UNIQUENESS.value == "uniqueness"
    
    def test_severity_enum(self):
        """Test Severity enum values."""
        assert Severity.LOW.value == "low"
        assert Severity.MEDIUM.value == "medium"
        assert Severity.HIGH.value == "high"
        assert Severity.CRITICAL.value == "critical"
    
    def test_policy_type_enum(self):
        """Test PolicyType enum values."""
        assert PolicyType.DATA_GOVERNANCE.value == "data_governance"
        assert PolicyType.PRIVACY.value == "privacy"
        assert PolicyType.SECURITY.value == "security"
        assert PolicyType.RETENTION.value == "retention"
    
    def test_compliance_status_enum(self):
        """Test ComplianceStatus enum values."""
        assert ComplianceStatus.COMPLIANT.value == "compliant"
        assert ComplianceStatus.NON_COMPLIANT.value == "non_compliant"
        assert ComplianceStatus.PARTIALLY_COMPLIANT.value == "partially_compliant"
        assert ComplianceStatus.UNKNOWN.value == "unknown"
    
    def test_data_classification_enum(self):
        """Test DataClassification enum values."""
        assert DataClassification.PUBLIC.value == "public"
        assert DataClassification.INTERNAL.value == "internal"
        assert DataClassification.CONFIDENTIAL.value == "confidential"
        assert DataClassification.RESTRICTED.value == "restricted"


class TestDataQualityIssue:
    """Test DataQualityIssue model."""
    
    def test_data_quality_issue_creation(self):
        """Test creating DataQualityIssue."""
        issue = DataQualityIssue(
            dimension=QualityDimension.COMPLETENESS,
            severity=Severity.HIGH,
            description="Missing values in required field",
            column_name="email",
            affected_rows=150
        )
        
        assert issue.dimension == QualityDimension.COMPLETENESS
        assert issue.severity == Severity.HIGH
        assert issue.description == "Missing values in required field"
        assert issue.column_name == "email"
        assert issue.affected_rows == 150
        assert issue.issue_id is not None
        assert issue.detected_at is not None
    
    def test_data_quality_issue_with_recommendation(self):
        """Test DataQualityIssue with recommendation."""
        issue = DataQualityIssue(
            dimension=QualityDimension.VALIDITY,
            severity=Severity.MEDIUM,
            description="Invalid email format",
            column_name="email",
            affected_rows=25,
            recommendation="Validate email format using regex pattern"
        )
        
        assert issue.recommendation == "Validate email format using regex pattern"
    
    def test_data_quality_issue_with_context(self):
        """Test DataQualityIssue with additional context."""
        context = {"pattern": "^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$"}
        issue = DataQualityIssue(
            dimension=QualityDimension.VALIDITY,
            severity=Severity.MEDIUM,
            description="Invalid email format",
            column_name="email",
            affected_rows=25,
            additional_context=context
        )
        
        assert issue.additional_context == context


class TestDataQualityMetrics:
    """Test DataQualityMetrics model."""
    
    def test_data_quality_metrics_creation(self):
        """Test creating DataQualityMetrics."""
        metrics = DataQualityMetrics(
            completeness_score=0.95,
            accuracy_score=0.88,
            consistency_score=0.92,
            validity_score=0.85,
            uniqueness_score=0.98
        )
        
        assert metrics.completeness_score == 0.95
        assert metrics.accuracy_score == 0.88
        assert metrics.consistency_score == 0.92
        assert metrics.validity_score == 0.85
        assert metrics.uniqueness_score == 0.98
        assert metrics.overall_score is not None
    
    def test_data_quality_metrics_overall_score_calculation(self):
        """Test overall score calculation."""
        metrics = DataQualityMetrics(
            completeness_score=1.0,
            accuracy_score=0.8,
            consistency_score=0.9,
            validity_score=0.7,
            uniqueness_score=1.0
        )
        
        # Should calculate average of all scores
        expected_overall = (1.0 + 0.8 + 0.9 + 0.7 + 1.0) / 5
        assert abs(metrics.overall_score - expected_overall) < 0.001


class TestColumnMetadata:
    """Test ColumnMetadata model."""
    
    def test_column_metadata_creation(self):
        """Test creating ColumnMetadata."""
        metadata = ColumnMetadata(
            name="customer_id",
            data_type="integer",
            is_nullable=False,
            is_primary_key=True
        )
        
        assert metadata.name == "customer_id"
        assert metadata.data_type == "integer"
        assert metadata.is_nullable is False
        assert metadata.is_primary_key is True
        assert metadata.is_foreign_key is False
        assert metadata.unique_values is None
        assert metadata.min_value is None
        assert metadata.max_value is None
    
    def test_column_metadata_with_statistics(self):
        """Test ColumnMetadata with statistical information."""
        metadata = ColumnMetadata(
            name="age",
            data_type="integer",
            is_nullable=True,
            unique_values=45,
            min_value=18,
            max_value=85,
            mean_value=42.5,
            null_count=5
        )
        
        assert metadata.unique_values == 45
        assert metadata.min_value == 18
        assert metadata.max_value == 85
        assert metadata.mean_value == 42.5
        assert metadata.null_count == 5
    
    def test_column_metadata_with_constraints(self):
        """Test ColumnMetadata with constraints."""
        constraints = ["NOT NULL", "CHECK (age >= 0)"]
        metadata = ColumnMetadata(
            name="age",
            data_type="integer",
            is_nullable=False,
            constraints=constraints
        )
        
        assert metadata.constraints == constraints


class TestDatasetMetadata:
    """Test DatasetMetadata model."""
    
    def test_dataset_metadata_creation(self):
        """Test creating DatasetMetadata."""
        columns = [
            ColumnMetadata(name="id", data_type="integer", is_primary_key=True),
            ColumnMetadata(name="name", data_type="varchar", is_nullable=False),
            ColumnMetadata(name="email", data_type="varchar", is_nullable=True)
        ]
        
        metadata = DatasetMetadata(
            name="customers",
            source=DataSource.DATABASE,
            format=DataFormat.CSV,
            columns=columns,
            row_count=1000
        )
        
        assert metadata.name == "customers"
        assert metadata.source == DataSource.DATABASE
        assert metadata.format == DataFormat.CSV
        assert len(metadata.columns) == 3
        assert metadata.row_count == 1000
        assert metadata.dataset_id is not None
        assert metadata.created_at is not None
    
    def test_dataset_metadata_with_schema(self):
        """Test DatasetMetadata with schema information."""
        columns = [
            ColumnMetadata(name="id", data_type="integer", is_primary_key=True)
        ]
        
        metadata = DatasetMetadata(
            name="products",
            source=DataSource.DATABASE,
            format=DataFormat.JSON,
            columns=columns,
            row_count=500,
            schema_name="retail",
            table_name="products"
        )
        
        assert metadata.schema_name == "retail"
        assert metadata.table_name == "products"
    
    def test_dataset_metadata_with_file_info(self):
        """Test DatasetMetadata with file information."""
        columns = [
            ColumnMetadata(name="timestamp", data_type="datetime"),
            ColumnMetadata(name="value", data_type="float")
        ]
        
        metadata = DatasetMetadata(
            name="sensor_data",
            source=DataSource.FILE,
            format=DataFormat.CSV,
            columns=columns,
            row_count=10000,
            file_path="/data/sensors/sensor_data.csv",
            file_size_bytes=2048000
        )
        
        assert metadata.file_path == "/data/sensors/sensor_data.csv"
        assert metadata.file_size_bytes == 2048000


class TestDataProfile:
    """Test DataProfile model."""
    
    def test_data_profile_creation(self):
        """Test creating DataProfile."""
        metadata = DatasetMetadata(
            name="test_dataset",
            source=DataSource.FILE,
            format=DataFormat.CSV,
            columns=[],
            row_count=100
        )
        
        metrics = DataQualityMetrics(
            completeness_score=0.95,
            accuracy_score=0.88,
            consistency_score=0.92,
            validity_score=0.85,
            uniqueness_score=0.98
        )
        
        profile = DataProfile(
            dataset_metadata=metadata,
            quality_metrics=metrics
        )
        
        assert profile.dataset_metadata == metadata
        assert profile.quality_metrics == metrics
        assert profile.profile_id is not None
        assert profile.generated_at is not None
        assert profile.issues == []
    
    def test_data_profile_with_issues(self):
        """Test DataProfile with quality issues."""
        metadata = DatasetMetadata(
            name="test_dataset",
            source=DataSource.FILE,
            format=DataFormat.CSV,
            columns=[],
            row_count=100
        )
        
        metrics = DataQualityMetrics(
            completeness_score=0.80,
            accuracy_score=0.75,
            consistency_score=0.85,
            validity_score=0.70,
            uniqueness_score=0.90
        )
        
        issues = [
            DataQualityIssue(
                dimension=QualityDimension.COMPLETENESS,
                severity=Severity.MEDIUM,
                description="Missing values",
                column_name="address",
                affected_rows=20
            )
        ]
        
        profile = DataProfile(
            dataset_metadata=metadata,
            quality_metrics=metrics,
            issues=issues
        )
        
        assert len(profile.issues) == 1
        assert profile.issues[0].dimension == QualityDimension.COMPLETENESS


class TestDataValidationRule:
    """Test DataValidationRule model."""
    
    def test_data_validation_rule_creation(self):
        """Test creating DataValidationRule."""
        rule = DataValidationRule(
            name="email_format_check",
            description="Validate email format",
            column_name="email",
            rule_type="regex",
            parameters={"pattern": "^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$"}
        )
        
        assert rule.name == "email_format_check"
        assert rule.description == "Validate email format"
        assert rule.column_name == "email"
        assert rule.rule_type == "regex"
        assert rule.parameters["pattern"] is not None
        assert rule.rule_id is not None
        assert rule.is_active is True
        assert rule.created_at is not None
    
    def test_data_validation_rule_inactive(self):
        """Test creating inactive DataValidationRule."""
        rule = DataValidationRule(
            name="deprecated_rule",
            description="Old validation rule",
            column_name="old_field",
            rule_type="range",
            parameters={"min": 0, "max": 100},
            is_active=False
        )
        
        assert rule.is_active is False
    
    def test_data_validation_rule_with_severity(self):
        """Test DataValidationRule with severity."""
        rule = DataValidationRule(
            name="critical_check",
            description="Critical validation",
            column_name="important_field",
            rule_type="not_null",
            parameters={},
            severity=Severity.CRITICAL
        )
        
        assert rule.severity == Severity.CRITICAL


class TestDataTransformation:
    """Test DataTransformation model."""
    
    def test_data_transformation_creation(self):
        """Test creating DataTransformation."""
        transformation = DataTransformation(
            name="normalize_names",
            description="Normalize customer names",
            transformation_type="standardization",
            source_columns=["first_name", "last_name"],
            target_columns=["full_name"]
        )
        
        assert transformation.name == "normalize_names"
        assert transformation.description == "Normalize customer names"
        assert transformation.transformation_type == "standardization"
        assert transformation.source_columns == ["first_name", "last_name"]
        assert transformation.target_columns == ["full_name"]
        assert transformation.transformation_id is not None
        assert transformation.created_at is not None
    
    def test_data_transformation_with_logic(self):
        """Test DataTransformation with transformation logic."""
        logic = "CONCAT(TRIM(first_name), ' ', TRIM(last_name))"
        parameters = {"case": "title", "remove_extra_spaces": True}
        
        transformation = DataTransformation(
            name="name_concat",
            description="Concatenate names",
            transformation_type="concatenation",
            source_columns=["first_name", "last_name"],
            target_columns=["full_name"],
            transformation_logic=logic,
            parameters=parameters
        )
        
        assert transformation.transformation_logic == logic
        assert transformation.parameters == parameters


class TestDataLineage:
    """Test DataLineage model."""
    
    def test_data_lineage_creation(self):
        """Test creating DataLineage."""
        lineage = DataLineage(
            source_dataset="raw_customers",
            target_dataset="clean_customers",
            transformation_applied="data_cleaning_pipeline"
        )
        
        assert lineage.source_dataset == "raw_customers"
        assert lineage.target_dataset == "clean_customers"
        assert lineage.transformation_applied == "data_cleaning_pipeline"
        assert lineage.lineage_id is not None
        assert lineage.created_at is not None
        assert lineage.upstream_lineage == []
        assert lineage.downstream_lineage == []
    
    def test_data_lineage_with_relationships(self):
        """Test DataLineage with upstream/downstream relationships."""
        upstream = ["source_system_1", "source_system_2"]
        downstream = ["reporting_dashboard", "analytics_model"]
        
        lineage = DataLineage(
            source_dataset="intermediate_data",
            target_dataset="aggregated_data",
            transformation_applied="aggregation_pipeline",
            upstream_lineage=upstream,
            downstream_lineage=downstream
        )
        
        assert lineage.upstream_lineage == upstream
        assert lineage.downstream_lineage == downstream
    
    def test_data_lineage_with_metadata(self):
        """Test DataLineage with additional metadata."""
        metadata = {
            "pipeline_version": "1.2.0",
            "execution_time": "2024-01-15T10:30:00Z",
            "record_count": 10000
        }
        
        lineage = DataLineage(
            source_dataset="input_data",
            target_dataset="output_data",
            transformation_applied="etl_pipeline",
            metadata=metadata
        )
        
        assert lineage.metadata == metadata


class TestPolicyRule:
    """Test PolicyRule model."""
    
    def test_policy_rule_creation(self):
        """Test creating PolicyRule."""
        rule = PolicyRule(
            name="gdpr_privacy_rule",
            policy_type=PolicyType.PRIVACY,
            description="GDPR compliance rule for personal data",
            rule_expression="contains_pii = true",
            severity=Severity.HIGH
        )
        
        assert rule.name == "gdpr_privacy_rule"
        assert rule.policy_type == PolicyType.PRIVACY
        assert rule.description == "GDPR compliance rule for personal data"
        assert rule.rule_expression == "contains_pii = true"
        assert rule.severity == Severity.HIGH
        assert rule.rule_id is not None
        assert rule.is_active is True
        assert rule.created_at is not None
    
    def test_policy_rule_with_parameters(self):
        """Test PolicyRule with parameters."""
        parameters = {
            "data_types": ["email", "phone", "ssn"],
            "retention_days": 365,
            "encryption_required": True
        }
        
        rule = PolicyRule(
            name="data_retention_rule",
            policy_type=PolicyType.RETENTION,
            description="Data retention policy",
            rule_expression="age_days > retention_days",
            severity=Severity.MEDIUM,
            parameters=parameters
        )
        
        assert rule.parameters == parameters
    
    def test_policy_rule_inactive(self):
        """Test creating inactive PolicyRule."""
        rule = PolicyRule(
            name="deprecated_policy",
            policy_type=PolicyType.SECURITY,
            description="Old security policy",
            rule_expression="deprecated_check = true",
            severity=Severity.LOW,
            is_active=False
        )
        
        assert rule.is_active is False


class TestIntegrationScenarios:
    """Test integration scenarios with multiple data types."""
    
    def test_complete_data_quality_assessment(self):
        """Test complete data quality assessment scenario."""
        # Create dataset metadata
        columns = [
            ColumnMetadata(
                name="customer_id",
                data_type="integer",
                is_primary_key=True,
                is_nullable=False
            ),
            ColumnMetadata(
                name="email",
                data_type="varchar",
                is_nullable=True,
                unique_values=950,
                null_count=50
            ),
            ColumnMetadata(
                name="age",
                data_type="integer",
                is_nullable=True,
                min_value=18,
                max_value=85,
                mean_value=42.5
            )
        ]
        
        metadata = DatasetMetadata(
            name="customer_data",
            source=DataSource.DATABASE,
            format=DataFormat.CSV,
            columns=columns,
            row_count=1000,
            schema_name="crm",
            table_name="customers"
        )
        
        # Create quality issues
        issues = [
            DataQualityIssue(
                dimension=QualityDimension.COMPLETENESS,
                severity=Severity.MEDIUM,
                description="Missing email addresses",
                column_name="email",
                affected_rows=50,
                recommendation="Implement email collection during registration"
            ),
            DataQualityIssue(
                dimension=QualityDimension.VALIDITY,
                severity=Severity.HIGH,
                description="Invalid email formats",
                column_name="email",
                affected_rows=25,
                recommendation="Validate email format using regex"
            )
        ]
        
        # Create quality metrics
        metrics = DataQualityMetrics(
            completeness_score=0.95,  # 50/1000 missing emails
            accuracy_score=0.88,
            consistency_score=0.92,
            validity_score=0.97,  # 25/1000 invalid emails
            uniqueness_score=0.95   # 950/1000 unique emails
        )
        
        # Create data profile
        profile = DataProfile(
            dataset_metadata=metadata,
            quality_metrics=metrics,
            issues=issues
        )
        
        # Verify complete assessment
        assert profile.dataset_metadata.name == "customer_data"
        assert len(profile.issues) == 2
        assert profile.quality_metrics.overall_score > 0.90
        assert any(issue.dimension == QualityDimension.COMPLETENESS for issue in profile.issues)
        assert any(issue.dimension == QualityDimension.VALIDITY for issue in profile.issues)
    
    def test_data_transformation_pipeline(self):
        """Test data transformation pipeline scenario."""
        # Create transformations
        transformations = [
            DataTransformation(
                name="clean_names",
                description="Clean and standardize names",
                transformation_type="cleaning",
                source_columns=["first_name", "last_name"],
                target_columns=["clean_first_name", "clean_last_name"],
                transformation_logic="TRIM(UPPER(column))"
            ),
            DataTransformation(
                name="create_full_name",
                description="Concatenate names",
                transformation_type="concatenation",
                source_columns=["clean_first_name", "clean_last_name"],
                target_columns=["full_name"],
                transformation_logic="CONCAT(clean_first_name, ' ', clean_last_name)"
            ),
            DataTransformation(
                name="anonymize_email",
                description="Hash email addresses for privacy",
                transformation_type="anonymization",
                source_columns=["email"],
                target_columns=["email_hash"],
                transformation_logic="SHA256(email)"
            )
        ]
        
        # Create lineage
        lineage = DataLineage(
            source_dataset="raw_customers",
            target_dataset="processed_customers",
            transformation_applied="customer_data_pipeline",
            metadata={
                "transformations_count": len(transformations),
                "pipeline_version": "2.1.0",
                "processing_time_seconds": 45.2
            }
        )
        
        # Verify pipeline
        assert len(transformations) == 3
        assert transformations[0].transformation_type == "cleaning"
        assert transformations[1].transformation_type == "concatenation"
        assert transformations[2].transformation_type == "anonymization"
        assert lineage.metadata["transformations_count"] == 3
    
    def test_compliance_policy_validation(self):
        """Test compliance policy validation scenario."""
        # Create policy rules
        privacy_rules = [
            PolicyRule(
                name="gdpr_consent_required",
                policy_type=PolicyType.PRIVACY,
                description="GDPR consent required for personal data",
                rule_expression="data_category = 'personal' AND consent_given = false",
                severity=Severity.CRITICAL,
                parameters={"applicable_regions": ["EU", "EEA"]}
            ),
            PolicyRule(
                name="pii_encryption_required",
                policy_type=PolicyType.SECURITY,
                description="PII data must be encrypted",
                rule_expression="contains_pii = true AND encrypted = false",
                severity=Severity.HIGH,
                parameters={"encryption_algorithm": "AES-256"}
            ),
            PolicyRule(
                name="data_retention_limit",
                policy_type=PolicyType.RETENTION,
                description="Data must be deleted after retention period",
                rule_expression="age_days > retention_period",
                severity=Severity.MEDIUM,
                parameters={"retention_period_days": 2555}  # 7 years
            )
        ]
        
        # Verify policy rules
        assert len(privacy_rules) == 3
        assert privacy_rules[0].policy_type == PolicyType.PRIVACY
        assert privacy_rules[1].policy_type == PolicyType.SECURITY
        assert privacy_rules[2].policy_type == PolicyType.RETENTION
        assert privacy_rules[0].severity == Severity.CRITICAL
        assert all(rule.is_active for rule in privacy_rules)
