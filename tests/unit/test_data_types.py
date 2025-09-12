"""
Unit tests for core.types.data module.
"""
import pytest
from unittest.mock import Mock
from datetime import datetime
from typing import Dict, Any, List
from pydantic import ValidationError

from core.types.data import (
    DataSource, DataField, DataSchema, DataRecord, DataIssue, 
    QualityMetrics, ComplianceViolation, RemediationTask,
    PolicyRecommendation, ProcessingRequest, AnalyticsVisualization
)
from core.types.base import (
    Severity, Priority, SensitivityLevel, DataType, ConfidenceLevel,
    QualityDimension, ComplianceRegulation, RemediationType, VisualizationType
)


class TestDataEnums:
    """Test data-related enum classes."""
    
    def test_data_type_enum(self):
        """Test DataType enum values."""
        assert DataType.STRING.value == "string"
        assert DataType.INTEGER.value == "integer"
        assert DataType.FLOAT.value == "float"
        assert DataType.BOOLEAN.value == "boolean"
        assert DataType.DATE.value == "date"
        assert DataType.DATETIME.value == "datetime"
        assert DataType.EMAIL.value == "email"
        assert DataType.PHONE.value == "phone"
        assert DataType.URL.value == "url"
        assert DataType.JSON.value == "json"
        assert DataType.BINARY.value == "binary"
    
    def test_sensitivity_level_enum(self):
        """Test SensitivityLevel enum values."""
        assert SensitivityLevel.PUBLIC.value == "public"
        assert SensitivityLevel.INTERNAL.value == "internal"
        assert SensitivityLevel.CONFIDENTIAL.value == "confidential"
        assert SensitivityLevel.RESTRICTED.value == "restricted"
        assert SensitivityLevel.TOP_SECRET.value == "top_secret"
    
    def test_quality_dimension_enum(self):
        """Test QualityDimension enum values."""
        assert QualityDimension.COMPLETENESS.value == "completeness"
        assert QualityDimension.ACCURACY.value == "accuracy"
        assert QualityDimension.CONSISTENCY.value == "consistency"
        assert QualityDimension.TIMELINESS.value == "timeliness"
        assert QualityDimension.VALIDITY.value == "validity"
        assert QualityDimension.UNIQUENESS.value == "uniqueness"
    
    def test_compliance_regulation_enum(self):
        """Test ComplianceRegulation enum values."""
        assert ComplianceRegulation.GDPR.value == "gdpr"
        assert ComplianceRegulation.CCPA.value == "ccpa"
        assert ComplianceRegulation.HIPAA.value == "hipaa"
        assert ComplianceRegulation.SOX.value == "sox"
        assert ComplianceRegulation.PCI_DSS.value == "pci_dss"
        assert ComplianceRegulation.FERPA.value == "ferpa"
        assert ComplianceRegulation.GLBA.value == "glba"

    def test_remediation_type_enum(self):
        """Test RemediationType enum values."""
        assert RemediationType.DEDUPLICATION.value == "deduplication"
        assert RemediationType.FORMAT_CORRECTION.value == "format_correction"
        assert RemediationType.VALUE_CORRECTION.value == "value_correction"
        assert RemediationType.ENRICHMENT.value == "enrichment"
        assert RemediationType.DELETION.value == "deletion"
        assert RemediationType.MASKING.value == "masking"
        assert RemediationType.ENCRYPTION.value == "encryption"

    def test_severity_enum(self):
        """Test Severity enum values."""
        assert Severity.CRITICAL.value == "critical"
        assert Severity.HIGH.value == "high"
        assert Severity.MEDIUM.value == "medium"
        assert Severity.LOW.value == "low"
        assert Severity.INFO.value == "info"

    def test_priority_enum(self):
        """Test Priority enum values."""
        assert Priority.CRITICAL.value == "critical"
        assert Priority.HIGH.value == "high"
        assert Priority.MEDIUM.value == "medium"
        assert Priority.LOW.value == "low"

    def test_confidence_level_enum(self):
        """Test ConfidenceLevel enum values."""
        assert ConfidenceLevel.VERY_HIGH.value == "very_high"
        assert ConfidenceLevel.HIGH.value == "high"
        assert ConfidenceLevel.MEDIUM.value == "medium"
        assert ConfidenceLevel.LOW.value == "low"
        assert ConfidenceLevel.VERY_LOW.value == "very_low"


class TestDataField:
    """Test DataField model."""
    
    def test_data_field_creation(self):
        """Test basic DataField creation."""
        field = DataField(
            name="user_id",
            data_type=DataType.INTEGER,
            is_required=True,
            is_primary_key=True,
            description="Unique user identifier"
        )
        
        assert field.name == "user_id"
        assert field.data_type == DataType.INTEGER
        assert field.is_required is True
        assert field.is_primary_key is True
        assert field.description == "Unique user identifier"
        assert field.sensitivity_level == SensitivityLevel.PUBLIC  # default
        assert field.is_foreign_key is False  # default
        assert field.tags == []  # default

    def test_data_field_with_validation(self):
        """Test DataField with validation constraints."""
        field = DataField(
            name="email",
            data_type=DataType.EMAIL,
            is_required=True,
            max_length=255,
            pattern=r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$",
            sensitivity_level=SensitivityLevel.CONFIDENTIAL,
            tags=["contact_info", "personally_identifiable"]
        )
        
        assert field.name == "email"
        assert field.data_type == DataType.EMAIL
        assert field.max_length == 255
        assert field.pattern is not None
        assert field.sensitivity_level == SensitivityLevel.CONFIDENTIAL
        assert "contact_info" in field.tags

    def test_data_field_with_allowed_values(self):
        """Test DataField with allowed values."""
        field = DataField(
            name="status",
            data_type=DataType.STRING,
            allowed_values=["active", "inactive", "pending"],
            description="User status"
        )
        
        assert field.allowed_values == ["active", "inactive", "pending"]
        assert field.description == "User status"


class TestDataSchema:
    """Test DataSchema model."""
    
    def test_data_schema_creation(self):
        """Test basic DataSchema creation."""
        fields = [
            DataField(name="id", data_type=DataType.INTEGER, is_primary_key=True),
            DataField(name="name", data_type=DataType.STRING, is_required=True),
            DataField(name="email", data_type=DataType.EMAIL)
        ]
        
        schema = DataSchema(
            name="users",
            version="1.0",
            description="User table schema",
            fields=fields,
            primary_keys=["id"],
            indexes=["email"]
        )
        
        assert schema.name == "users"
        assert schema.version == "1.0"
        assert schema.description == "User table schema"
        assert len(schema.fields) == 3
        assert schema.primary_keys == ["id"]
        assert schema.indexes == ["email"]

    def test_data_schema_with_foreign_keys(self):
        """Test DataSchema with foreign key relationships."""
        fields = [
            DataField(name="id", data_type=DataType.INTEGER, is_primary_key=True),
            DataField(name="user_id", data_type=DataType.INTEGER, is_foreign_key=True),
            DataField(name="order_date", data_type=DataType.DATETIME)
        ]
        
        schema = DataSchema(
            name="orders",
            fields=fields,
            foreign_keys={"user_id": "users.id"}
        )
        
        assert schema.foreign_keys["user_id"] == "users.id"

    def test_data_schema_with_constraints(self):
        """Test DataSchema with constraints."""
        fields = [
            DataField(name="id", data_type=DataType.INTEGER, is_primary_key=True),
            DataField(name="amount", data_type=DataType.FLOAT)
        ]
        
        constraints = [
            {"type": "check", "name": "positive_amount", "condition": "amount > 0"}
        ]
        
        schema = DataSchema(
            name="transactions",
            fields=fields,
            constraints=constraints
        )
        
        assert len(schema.constraints) == 1
        assert schema.constraints[0]["type"] == "check"


class TestDataSource:
    """Test DataSource model."""
    
    def test_data_source_creation(self):
        """Test basic DataSource creation."""
        source = DataSource(
            id="db_001",
            name="Main Database",
            type="database",
            connection_string="postgresql://user:pass@localhost:5432/db",
            is_active=True,
            record_count=10000
        )
        
        assert source.id == "db_001"
        assert source.name == "Main Database"
        assert source.type == "database"
        assert source.is_active is True
        assert source.record_count == 10000
        assert source.created_at is not None

    def test_data_source_with_schema(self):
        """Test DataSource with schema."""
        fields = [
            DataField(name="id", data_type=DataType.INTEGER, is_primary_key=True)
        ]
        schema = DataSchema(name="test_table", fields=fields)
        
        # Use the alias "schema" instead of "data_schema"
        source = DataSource(
            id="api_001",
            name="External API",
            type="api",
            schema=schema
        )
        
        assert source.data_schema is not None
        assert source.data_schema.name == "test_table"

    def test_data_source_with_metadata(self):
        """Test DataSource with metadata."""
        metadata = {
            "connection_timeout": 30,
            "retry_count": 3,
            "last_backup": "2024-01-15T10:00:00Z"
        }
        
        source = DataSource(
            id="file_001",
            name="CSV File",
            type="file",
            metadata=metadata,
            tags=["import", "quarterly"]
        )
        
        assert source.metadata["connection_timeout"] == 30
        assert "import" in source.tags


class TestDataRecord:
    """Test DataRecord model."""
    
    def test_data_record_creation(self):
        """Test basic DataRecord creation."""
        data = {
            "id": 1,
            "name": "John Doe",
            "email": "john@example.com"
        }
        
        record = DataRecord(
            id="rec_001",
            source_id="db_001",
            data=data
        )
        
        assert record.id == "rec_001"
        assert record.source_id == "db_001"
        assert record.data == data
        assert record.version == 1  # default
        assert record.is_deleted is False  # default
        assert record.created_at is not None

    def test_data_record_with_versioning(self):
        """Test DataRecord with versioning."""
        record = DataRecord(
            id="rec_002",
            source_id="db_001",
            data={"field": "value"},
            version=3,
            updated_at=datetime.utcnow()
        )
        
        assert record.version == 3
        assert record.updated_at is not None

    def test_data_record_with_metadata(self):
        """Test DataRecord with metadata."""
        metadata = {
            "import_batch": "batch_001",
            "validation_status": "passed"
        }
        
        record = DataRecord(
            id="rec_003",
            source_id="api_001",
            data={"test": "data"},
            metadata=metadata
        )
        
        assert record.metadata["import_batch"] == "batch_001"


class TestDataIssue:
    """Test DataIssue model."""
    
    def test_data_issue_creation(self):
        """Test basic DataIssue creation."""
        issue = DataIssue(
            id="issue_001",
            issue_type="data_quality",
            title="Missing Values",
            description="Found missing values in required fields",
            severity=Severity.HIGH,
            priority=Priority.HIGH,
            confidence=ConfidenceLevel.HIGH,
            source_id="db_001",
            detection_method="automated_scan"
        )
        
        assert issue.id == "issue_001"
        assert issue.issue_type == "data_quality"
        assert issue.title == "Missing Values"
        assert issue.severity == Severity.HIGH
        assert issue.priority == Priority.HIGH
        assert issue.confidence == ConfidenceLevel.HIGH
        assert issue.status == "open"  # default
        assert issue.created_at is not None

    def test_data_issue_with_affected_data(self):
        """Test DataIssue with affected records and fields."""
        issue = DataIssue(
            id="issue_002",
            issue_type="validation",
            title="Invalid Email Format",
            description="Email field contains invalid formats",
            severity=Severity.MEDIUM,
            priority=Priority.MEDIUM,
            confidence=ConfidenceLevel.HIGH,
            affected_records=["rec_001", "rec_002"],
            affected_fields=["email"],
            source_id="db_001",
            detection_method="regex_validation",
            suggested_actions=["Validate email format", "Clean data"]
        )
        
        assert len(issue.affected_records) == 2
        assert "email" in issue.affected_fields
        assert len(issue.suggested_actions) == 2

    def test_data_issue_with_assignment(self):
        """Test DataIssue with assignment."""
        issue = DataIssue(
            id="issue_003",
            issue_type="compliance",
            title="PII Data Exposure",
            description="Personal data found in unsecured field",
            severity=Severity.CRITICAL,
            priority=Priority.CRITICAL,
            confidence=ConfidenceLevel.VERY_HIGH,
            source_id="db_001",
            detection_method="compliance_scan",
            assigned_to="admin@company.com",
            status="in_progress"
        )
        
        assert issue.assigned_to == "admin@company.com"
        assert issue.status == "in_progress"


class TestQualityMetrics:
    """Test QualityMetrics model."""
    
    def test_quality_metrics_creation(self):
        """Test basic QualityMetrics creation."""
        metrics = QualityMetrics(
            dataset_id="dataset_001",
            overall_score=0.85,
            completeness=0.95,
            accuracy=0.90,
            consistency=0.85,
            timeliness=0.80,
            validity=0.88,
            uniqueness=0.92,
            total_records=1000,
            complete_records=950,
            duplicate_records=20,
            invalid_records=50,
            outdated_records=100
        )
        
        assert metrics.dataset_id == "dataset_001"
        assert metrics.overall_score == 0.85
        assert metrics.completeness == 0.95
        assert metrics.total_records == 1000
        assert metrics.complete_records == 950
        assert metrics.calculated_at is not None

    def test_quality_metrics_with_field_metrics(self):
        """Test QualityMetrics with field-level metrics."""
        field_metrics = {
            "email": {"completeness": 0.95, "validity": 0.80},
            "phone": {"completeness": 0.70, "validity": 0.85}
        }
        
        metrics = QualityMetrics(
            dataset_id="dataset_002",
            overall_score=0.82,
            completeness=0.83,
            accuracy=0.85,
            consistency=0.80,
            timeliness=0.85,
            validity=0.82,
            uniqueness=0.78,
            total_records=500,
            complete_records=415,
            duplicate_records=10,
            invalid_records=25,
            outdated_records=50,
            field_metrics=field_metrics
        )
        
        assert metrics.field_metrics["email"]["completeness"] == 0.95
        assert metrics.field_metrics["phone"]["validity"] == 0.85

    def test_quality_metrics_with_trends(self):
        """Test QualityMetrics with trend information."""
        metrics = QualityMetrics(
            dataset_id="dataset_003",
            overall_score=0.78,
            completeness=0.80,
            accuracy=0.75,
            consistency=0.85,
            timeliness=0.70,
            validity=0.80,
            uniqueness=0.85,
            total_records=2000,
            complete_records=1600,
            duplicate_records=50,
            invalid_records=100,
            outdated_records=250,
            trend_direction="improving",
            trend_percentage=5.2
        )
        
        assert metrics.trend_direction == "improving"
        assert metrics.trend_percentage == 5.2

    def test_quality_metrics_score_validation(self):
        """Test QualityMetrics score validation (0.0 to 1.0)."""
        with pytest.raises(ValidationError):
            QualityMetrics(
                dataset_id="dataset_004",
                overall_score=1.5,  # Invalid: > 1.0
                completeness=0.95,
                accuracy=0.90,
                consistency=0.85,
                timeliness=0.80,
                validity=0.88,
                uniqueness=0.92,
                total_records=1000,
                complete_records=950,
                duplicate_records=20,
                invalid_records=50,
                outdated_records=100
            )


class TestComplianceViolation:
    """Test ComplianceViolation model."""
    
    def test_compliance_violation_creation(self):
        """Test basic ComplianceViolation creation."""
        violation = ComplianceViolation(
            id="violation_001",
            regulation="GDPR",
            violation_type="data_retention",
            description="Data retained beyond legal requirement",
            severity=Severity.HIGH,
            confidence=ConfidenceLevel.HIGH,
            affected_data=["personal_data", "contact_info"],
            recommended_actions=["Delete old records", "Update retention policy"]
        )
        
        assert violation.id == "violation_001"
        assert violation.regulation == "GDPR"
        assert violation.violation_type == "data_retention"
        assert violation.severity == Severity.HIGH
        assert violation.confidence == ConfidenceLevel.HIGH
        assert len(violation.affected_data) == 2
        assert len(violation.recommended_actions) == 2
        assert violation.status == "open"  # default

    def test_compliance_violation_with_deadline(self):
        """Test ComplianceViolation with deadline."""
        deadline = datetime(2024, 12, 31, 23, 59, 59)
        
        violation = ComplianceViolation(
            id="violation_002",
            regulation="CCPA",
            violation_type="data_access",
            description="User unable to access personal data",
            severity=Severity.MEDIUM,
            confidence=ConfidenceLevel.HIGH,
            deadline=deadline,
            legal_basis="Right to data portability",
            potential_penalty="$7,500 per violation"
        )
        
        assert violation.deadline == deadline
        assert violation.legal_basis == "Right to data portability"
        assert violation.potential_penalty == "$7,500 per violation"

    def test_compliance_violation_with_metadata(self):
        """Test ComplianceViolation with metadata."""
        metadata = {
            "detection_source": "compliance_audit",
            "auditor": "external_firm",
            "audit_date": "2024-01-15"
        }
        
        violation = ComplianceViolation(
            id="violation_003",
            regulation="HIPAA",
            violation_type="data_breach",
            description="Unauthorized access to patient records",
            severity=Severity.CRITICAL,
            confidence=ConfidenceLevel.VERY_HIGH,
            metadata=metadata
        )
        
        assert violation.metadata["detection_source"] == "compliance_audit"


class TestRemediationTask:
    """Test RemediationTask model."""
    
    def test_remediation_task_creation(self):
        """Test basic RemediationTask creation."""
        task = RemediationTask(
            id="task_001",
            title="Clean Invalid Email Addresses",
            description="Remove or correct invalid email formats",
            task_type="data_cleaning",
            priority=Priority.HIGH,
            estimated_effort_hours=8.0,
            affected_records=["rec_001", "rec_002"]
        )
        
        assert task.id == "task_001"
        assert task.title == "Clean Invalid Email Addresses"
        assert task.task_type == "data_cleaning"
        assert task.priority == Priority.HIGH
        assert task.estimated_effort_hours == 8.0
        assert task.status == "pending"  # default
        assert len(task.affected_records) == 2

    def test_remediation_task_with_assignment(self):
        """Test RemediationTask with assignment."""
        task = RemediationTask(
            id="task_002",
            title="Implement Data Masking",
            description="Mask sensitive fields in development database",
            task_type="data_security",
            priority=Priority.CRITICAL,
            assigned_to="security_team@company.com",
            assigned_team="Security Engineering",
            status="in_progress"
        )
        
        assert task.assigned_to == "security_team@company.com"
        assert task.assigned_team == "Security Engineering"
        assert task.status == "in_progress"

    def test_remediation_task_with_progress(self):
        """Test RemediationTask with progress tracking."""
        task = RemediationTask(
            id="task_003",
            title="Deduplicate Records",
            description="Remove duplicate customer records",
            task_type="deduplication",
            priority=Priority.MEDIUM,
            progress_percentage=75.0,
            records_processed=750,
            records_fixed=700,
            records_failed=50,
            actual_effort_hours=12.5
        )
        
        assert task.progress_percentage == 75.0
        assert task.records_processed == 750
        assert task.records_fixed == 700
        assert task.records_failed == 50
        assert task.actual_effort_hours == 12.5

    def test_remediation_task_with_results(self):
        """Test RemediationTask with completion results."""
        quality_improvement = {
            "completeness": 0.15,
            "validity": 0.25,
            "uniqueness": 0.30
        }
        
        task = RemediationTask(
            id="task_004",
            title="Data Quality Enhancement",
            description="Comprehensive data quality improvement",
            task_type="quality_enhancement",
            priority=Priority.HIGH,
            status="completed",
            progress_percentage=100.0,
            quality_improvement=quality_improvement,
            completion_time=datetime.utcnow(),
            success_rate=0.95
        )
        
        assert task.status == "completed"
        assert task.quality_improvement["completeness"] == 0.15
        assert task.success_rate == 0.95
        assert task.completion_time is not None


class TestPolicyRecommendation:
    """Test PolicyRecommendation model."""
    
    def test_policy_recommendation_creation(self):
        """Test basic PolicyRecommendation creation."""
        recommendation = PolicyRecommendation(
            id="policy_001",
            title="Implement Data Retention Policy",
            description="Establish clear data retention periods",
            policy_type="retention",
            confidence=ConfidenceLevel.HIGH,
            priority=Priority.HIGH,
            business_justification="Comply with GDPR requirements",
            scope=["customer_data", "transaction_logs"],
            compliance_requirements=["GDPR", "CCPA"]
        )
        
        assert recommendation.id == "policy_001"
        assert recommendation.title == "Implement Data Retention Policy"
        assert recommendation.policy_type == "retention"
        assert recommendation.confidence == ConfidenceLevel.HIGH
        assert recommendation.priority == Priority.HIGH
        assert len(recommendation.scope) == 2
        assert len(recommendation.compliance_requirements) == 2
        assert recommendation.status == "pending_review"  # default

    def test_policy_recommendation_with_implementation(self):
        """Test PolicyRecommendation with implementation details."""
        implementation_steps = [
            "Define retention periods by data type",
            "Implement automated archival process",
            "Create deletion schedule",
            "Update privacy policy"
        ]
        
        rules = [
            {"field": "personal_data", "retention_days": 2555},  # 7 years
            {"field": "transaction_data", "retention_days": 365}  # 1 year
        ]
        
        recommendation = PolicyRecommendation(
            id="policy_002",
            title="Data Access Control Policy",
            description="Implement role-based access controls",
            policy_type="access",
            confidence=ConfidenceLevel.VERY_HIGH,
            priority=Priority.CRITICAL,
            business_justification="Prevent unauthorized data access",
            rules=rules,
            enforcement_level="mandatory",
            implementation_steps=implementation_steps,
            estimated_implementation_time="4-6 weeks",
            required_resources=["Security Team", "Database Admin", "Legal Review"]
        )
        
        assert recommendation.enforcement_level == "mandatory"
        assert len(recommendation.implementation_steps) == 4
        assert recommendation.estimated_implementation_time == "4-6 weeks"
        assert len(recommendation.required_resources) == 3
        assert len(recommendation.rules) == 2

    def test_policy_recommendation_with_approval(self):
        """Test PolicyRecommendation with approval workflow."""
        approval_time = datetime.utcnow()
        
        recommendation = PolicyRecommendation(
            id="policy_003",
            title="Data Encryption Policy",
            description="Encrypt all sensitive data at rest",
            policy_type="security",
            confidence=ConfidenceLevel.HIGH,
            priority=Priority.HIGH,
            business_justification="Protect sensitive customer information",
            status="approved",
            reviewed_by="security_officer@company.com",
            approved_by="cto@company.com",
            approved_at=approval_time
        )
        
        assert recommendation.status == "approved"
        assert recommendation.reviewed_by == "security_officer@company.com"
        assert recommendation.approved_by == "cto@company.com"
        assert recommendation.approved_at == approval_time


class TestProcessingRequest:
    """Test ProcessingRequest model."""
    
    def test_processing_request_creation(self):
        """Test basic ProcessingRequest creation."""
        request = ProcessingRequest(
            data_source="api_endpoint",
            data={"user_id": 123, "action": "update"},
            priority="high",
            callback_url="https://api.example.com/callback"
        )
        
        assert request.data_source == "api_endpoint"
        assert request.data["user_id"] == 123
        assert request.priority == "high"
        assert request.callback_url == "https://api.example.com/callback"

    def test_processing_request_with_parameters(self):
        """Test ProcessingRequest with parameters."""
        parameters = {
            "batch_size": 100,
            "timeout": 30,
            "retry_count": 3
        }
        
        metadata = {
            "request_id": "req_001",
            "client_id": "client_123"
        }
        
        request = ProcessingRequest(
            parameters=parameters,
            metadata=metadata
        )
        
        assert request.parameters["batch_size"] == 100
        assert request.metadata["request_id"] == "req_001"
        assert request.priority == "normal"  # default

    def test_processing_request_minimal(self):
        """Test ProcessingRequest with minimal data."""
        request = ProcessingRequest()
        
        assert request.data_source is None
        assert request.data is None
        assert request.priority == "normal"
        assert request.callback_url is None
        assert request.parameters == {}
        assert request.metadata == {}


class TestAnalyticsVisualization:
    """Test AnalyticsVisualization model."""
    
    def test_analytics_visualization_creation(self):
        """Test basic AnalyticsVisualization creation."""
        visualization = AnalyticsVisualization(
            id="viz_001",
            title="User Registration Trends",
            visualization_type="line_chart",
            data_source="user_database",
            x_axis="registration_date",
            y_axis="user_count"
        )
        
        assert visualization.id == "viz_001"
        assert visualization.title == "User Registration Trends"
        assert visualization.visualization_type == "line_chart"
        assert visualization.data_source == "user_database"
        assert visualization.x_axis == "registration_date"
        assert visualization.y_axis == "user_count"

    def test_analytics_visualization_with_data(self):
        """Test AnalyticsVisualization with data."""
        data = [
            {"date": "2024-01-01", "count": 10},
            {"date": "2024-01-02", "count": 15},
            {"date": "2024-01-03", "count": 8}
        ]
        
        summary_stats = {
            "total": 33,
            "average": 11.0,
            "max": 15,
            "min": 8
        }
        
        visualization = AnalyticsVisualization(
            id="viz_002",
            title="Daily User Activity",
            visualization_type="bar_chart",
            data_source="activity_logs",
            data=data,
            summary_stats=summary_stats
        )
        
        assert len(visualization.data) == 3
        assert visualization.summary_stats["total"] == 33
        assert visualization.summary_stats["average"] == 11.0

    def test_analytics_visualization_with_styling(self):
        """Test AnalyticsVisualization with styling options."""
        colors = ["#FF6B6B", "#4ECDC4", "#45B7D1"]
        filters = {"date_range": "last_30_days", "status": "active"}
        
        visualization = AnalyticsVisualization(
            id="viz_003",
            title="Revenue by Product Category",
            visualization_type="pie_chart",
            data_source="sales_data",
            grouping="product_category",
            filters=filters,
            colors=colors,
            theme="dark"
        )
        
        assert visualization.grouping == "product_category"
        assert visualization.filters["date_range"] == "last_30_days"
        assert len(visualization.colors) == 3
        assert visualization.theme == "dark"

    def test_analytics_visualization_with_metadata(self):
        """Test AnalyticsVisualization with metadata."""
        metadata = {
            "created_by": "analyst@company.com",
            "department": "marketing",
            "last_updated": "2024-01-15T10:00:00Z",
            "refresh_interval": "1h"
        }
        
        visualization = AnalyticsVisualization(
            id="viz_004",
            title="Customer Satisfaction Scores",
            visualization_type="gauge",
            data_source="survey_data",
            metadata=metadata
        )
        
        assert visualization.metadata["created_by"] == "analyst@company.com"
        assert visualization.metadata["department"] == "marketing"
        assert visualization.metadata["refresh_interval"] == "1h"


class TestIntegrationScenarios:
    """Test integration scenarios combining multiple data types."""
    
    def test_complete_data_quality_assessment(self):
        """Test complete data quality assessment workflow."""
        # Create data source
        source = DataSource(
            id="db_001",
            name="Customer Database",
            type="database",
            record_count=10000
        )
        
        # Create quality metrics
        metrics = QualityMetrics(
            dataset_id="db_001",
            overall_score=0.75,
            completeness=0.85,
            accuracy=0.70,
            consistency=0.80,
            timeliness=0.65,
            validity=0.75,
            uniqueness=0.90,
            total_records=10000,
            complete_records=8500,
            duplicate_records=500,
            invalid_records=1000,
            outdated_records=2000
        )
        
        # Create data issues
        issue = DataIssue(
            id="issue_001",
            issue_type="data_quality",
            title="Low Accuracy Score",
            description="Accuracy below threshold in customer database",
            severity=Severity.HIGH,
            priority=Priority.HIGH,
            confidence=ConfidenceLevel.HIGH,
            source_id="db_001",
            detection_method="quality_assessment"
        )
        
        # Verify relationships
        assert source.id == metrics.dataset_id
        assert issue.source_id == source.id
        assert metrics.overall_score == 0.75
        assert issue.severity == Severity.HIGH

    def test_data_remediation_pipeline(self):
        """Test data remediation pipeline workflow."""
        # Create remediation task
        task = RemediationTask(
            id="task_001",
            title="Improve Data Quality",
            description="Address quality issues in customer database",
            task_type="quality_improvement",
            priority=Priority.HIGH,
            affected_records=["rec_001", "rec_002"],
            progress_percentage=50.0
        )
        
        # Create processing request for remediation
        request = ProcessingRequest(
            data_source="db_001",
            parameters={
                "task_id": "task_001",
                "batch_size": 1000,
                "remediation_type": "validation"
            },
            priority="high"
        )
        
        # Verify workflow
        assert task.task_type == "quality_improvement"
        assert request.parameters["task_id"] == "task_001"
        assert task.priority.value == request.priority

    def test_compliance_policy_validation(self):
        """Test compliance and policy validation workflow."""
        # Create compliance violation
        violation = ComplianceViolation(
            id="violation_001",
            regulation="GDPR",
            violation_type="data_retention",
            description="Data retained beyond legal limits",
            severity=Severity.HIGH,
            confidence=ConfidenceLevel.HIGH
        )
        
        # Create policy recommendation
        recommendation = PolicyRecommendation(
            id="policy_001",
            title="Update Data Retention Policy",
            description="Implement automated data deletion",
            policy_type="retention",
            confidence=ConfidenceLevel.HIGH,
            priority=Priority.HIGH,
            business_justification="Address GDPR compliance violation",
            compliance_requirements=["GDPR"]
        )
        
        # Create remediation task
        task = RemediationTask(
            id="task_002",
            title="Implement Data Deletion",
            description="Delete old records per new retention policy",
            task_type="deletion",
            priority=Priority.HIGH
        )
        
        # Verify compliance workflow
        assert violation.regulation in recommendation.compliance_requirements
        assert violation.severity == recommendation.priority
        assert task.task_type == "deletion"
        assert task.priority == Priority.HIGH
