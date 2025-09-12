"""
Unit tests for core.types.data module - Fixed to match actual implementation.
"""
import pytest
from unittest.mock import Mock
from datetime import datetime
from typing import Dict, Any, List
from core.types.data import (
    DataSource, DataField, DataSchema, DataRecord, DataIssue, 
    QualityMetrics, ComplianceViolation, RemediationTask,
    PolicyRecommendation, ProcessingRequest, AnalyticsVisualization
)
from core.types.base import (
    Severity, Priority, SensitivityLevel, DataType
)


class TestDataEnums:
    """Test data-related enum classes that actually exist."""
    
    def test_severity_enum(self):
        """Test Severity enum values."""
        assert Severity.LOW.value == "low"
        assert Severity.MEDIUM.value == "medium"
        assert Severity.HIGH.value == "high"
        assert Severity.CRITICAL.value == "critical"
    
    def test_priority_enum(self):
        """Test Priority enum values."""
        assert Priority.LOW.value == "low"
        assert Priority.MEDIUM.value == "medium"
        assert Priority.HIGH.value == "high"
        assert Priority.URGENT.value == "urgent"
    
    def test_sensitivity_level_enum(self):
        """Test SensitivityLevel enum values."""
        assert SensitivityLevel.PUBLIC.value == "public"
        assert SensitivityLevel.INTERNAL.value == "internal"
        assert SensitivityLevel.CONFIDENTIAL.value == "confidential"
        assert SensitivityLevel.RESTRICTED.value == "restricted"
    
    def test_data_type_enum(self):
        """Test DataType enum values."""
        assert DataType.STRING.value == "string"
        assert DataType.INTEGER.value == "integer"
        assert DataType.FLOAT.value == "float"
        assert DataType.BOOLEAN.value == "boolean"
        assert DataType.DATE.value == "date"
        assert DataType.DATETIME.value == "datetime"
        assert DataType.JSON.value == "json"


class TestDataField:
    """Test DataField model."""
    
    def test_data_field_creation(self):
        """Test basic DataField creation."""
        field = DataField(
            name="user_id",
            data_type=DataType.INTEGER,
            is_required=True,
            is_primary_key=True
        )
        
        assert field.name == "user_id"
        assert field.data_type == DataType.INTEGER
        assert field.is_required is True
        assert field.is_primary_key is True
        assert field.is_foreign_key is False
        assert field.sensitivity_level == SensitivityLevel.PUBLIC
    
    def test_data_field_with_validation(self):
        """Test DataField with validation rules."""
        field = DataField(
            name="email",
            data_type=DataType.STRING,
            is_required=True,
            max_length=255,
            pattern=r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
            sensitivity_level=SensitivityLevel.CONFIDENTIAL
        )
        
        assert field.name == "email"
        assert field.max_length == 255
        assert field.pattern is not None
        assert field.sensitivity_level == SensitivityLevel.CONFIDENTIAL
    
    def test_data_field_with_allowed_values(self):
        """Test DataField with constrained values."""
        field = DataField(
            name="status",
            data_type=DataType.STRING,
            allowed_values=["active", "inactive", "pending"],
            description="User account status"
        )
        
        assert field.allowed_values == ["active", "inactive", "pending"]
        assert field.description == "User account status"


class TestDataSchema:
    """Test DataSchema model."""
    
    def test_data_schema_creation(self):
        """Test basic DataSchema creation."""
        fields = [
            DataField(name="id", data_type=DataType.INTEGER, is_primary_key=True),
            DataField(name="name", data_type=DataType.STRING, is_required=True),
            DataField(name="email", data_type=DataType.STRING)
        ]
        
        schema = DataSchema(
            name="users",
            version="1.0",
            description="User table schema",
            fields=fields,
            primary_keys=["id"]
        )
        
        assert schema.name == "users"
        assert schema.version == "1.0"
        assert len(schema.fields) == 3
        assert schema.primary_keys == ["id"]
    
    def test_data_schema_with_relationships(self):
        """Test DataSchema with foreign keys."""
        fields = [
            DataField(name="id", data_type=DataType.INTEGER, is_primary_key=True),
            DataField(name="user_id", data_type=DataType.INTEGER, is_foreign_key=True),
            DataField(name="content", data_type=DataType.STRING)
        ]
        
        schema = DataSchema(
            name="posts",
            fields=fields,
            foreign_keys={"user_id": "users.id"},
            indexes=["user_id", "created_at"]
        )
        
        assert schema.foreign_keys == {"user_id": "users.id"}
        assert "user_id" in schema.indexes


class TestDataSource:
    """Test DataSource model."""
    
    def test_data_source_creation(self):
        """Test basic DataSource creation."""
        source = DataSource(
            id="db_users",
            name="User Database",
            type="database",
            connection_string="postgresql://localhost:5432/users"
        )
        
        assert source.id == "db_users"
        assert source.name == "User Database"
        assert source.type == "database"
        assert source.connection_string == "postgresql://localhost:5432/users"
    
    def test_data_source_with_schema(self):
        """Test DataSource with schema definition."""
        fields = [
            DataField(name="id", data_type=DataType.INTEGER, is_primary_key=True)
        ]
        schema = DataSchema(name="test_table", fields=fields)
        
        source = DataSource(
            id="test_source",
            name="Test Source",
            type="file",
            data_schema=schema
        )
        
        assert source.data_schema is not None
        assert source.data_schema.name == "test_table"


class TestDataRecord:
    """Test DataRecord model."""
    
    def test_data_record_creation(self):
        """Test basic DataRecord creation."""
        record = DataRecord(
            id="record_1",
            source_id="db_users",
            data={"id": 1, "name": "John Doe", "email": "john@example.com"}
        )
        
        assert record.id == "record_1"
        assert record.source_id == "db_users"
        assert record.data["name"] == "John Doe"
        assert record.created_at is not None
    
    def test_data_record_with_metadata(self):
        """Test DataRecord with additional metadata."""
        record = DataRecord(
            id="record_2",
            source_id="api_users",
            data={"user_id": 123, "status": "active"},
            version="1.0",
            metadata={"imported_from": "legacy_system", "batch_id": "batch_001"}
        )
        
        assert record.metadata["imported_from"] == "legacy_system"
        assert record.version == "1.0"


class TestDataIssue:
    """Test DataIssue model."""
    
    def test_data_issue_creation(self):
        """Test basic DataIssue creation."""
        issue = DataIssue(
            id="issue_1",
            source_id="db_users",
            field_name="email",
            issue_type="validation",
            severity=Severity.HIGH,
            description="Invalid email format"
        )
        
        assert issue.id == "issue_1"
        assert issue.field_name == "email"
        assert issue.issue_type == "validation"
        assert issue.severity == Severity.HIGH
    
    def test_data_issue_with_suggestion(self):
        """Test DataIssue with remediation suggestion."""
        issue = DataIssue(
            id="issue_2",
            source_id="db_users",
            field_name="phone",
            issue_type="missing",
            severity=Severity.MEDIUM,
            description="Missing phone number",
            suggested_fix="Add phone number validation to form"
        )
        
        assert issue.suggested_fix is not None
        assert "validation" in issue.suggested_fix


class TestQualityMetrics:
    """Test QualityMetrics model."""
    
    def test_quality_metrics_creation(self):
        """Test basic QualityMetrics creation."""
        metrics = QualityMetrics(
            source_id="db_users",
            total_records=1000,
            valid_records=950,
            invalid_records=50,
            completeness_score=0.95,
            accuracy_score=0.98
        )
        
        assert metrics.total_records == 1000
        assert metrics.valid_records == 950
        assert metrics.completeness_score == 0.95
        assert metrics.accuracy_score == 0.98
    
    def test_quality_metrics_overall_score(self):
        """Test QualityMetrics overall score calculation."""
        metrics = QualityMetrics(
            source_id="test_source",
            total_records=100,
            valid_records=95,
            invalid_records=5,
            completeness_score=0.95,
            accuracy_score=0.90,
            consistency_score=0.85,
            timeliness_score=0.88
        )
        
        # The overall score should be the average of individual scores
        expected_overall = (0.95 + 0.90 + 0.85 + 0.88) / 4
        assert abs(metrics.overall_score - expected_overall) < 0.01


class TestComplianceViolation:
    """Test ComplianceViolation model."""
    
    def test_compliance_violation_creation(self):
        """Test basic ComplianceViolation creation."""
        violation = ComplianceViolation(
            id="violation_1",
            source_id="db_users",
            rule_id="gdpr_email_consent",
            severity=Severity.HIGH,
            description="Email processing without consent"
        )
        
        assert violation.id == "violation_1"
        assert violation.rule_id == "gdpr_email_consent"
        assert violation.severity == Severity.HIGH
    
    def test_compliance_violation_with_remediation(self):
        """Test ComplianceViolation with remediation plan."""
        violation = ComplianceViolation(
            id="violation_2",
            source_id="db_users",
            rule_id="data_retention",
            severity=Severity.MEDIUM,
            description="Data retained beyond policy limit",
            remediation_required=True,
            remediation_deadline=datetime(2024, 12, 31)
        )
        
        assert violation.remediation_required is True
        assert violation.remediation_deadline is not None


class TestRemediationTask:
    """Test RemediationTask model."""
    
    def test_remediation_task_creation(self):
        """Test basic RemediationTask creation."""
        task = RemediationTask(
            id="task_1",
            source_id="db_users",
            task_type="data_cleaning",
            priority=Priority.HIGH,
            description="Clean invalid email addresses"
        )
        
        assert task.id == "task_1"
        assert task.task_type == "data_cleaning"
        assert task.priority == Priority.HIGH
        assert task.status == "pending"  # Default status
    
    def test_remediation_task_with_schedule(self):
        """Test RemediationTask with scheduling."""
        task = RemediationTask(
            id="task_2",
            source_id="db_orders",
            task_type="data_validation",
            priority=Priority.MEDIUM,
            description="Validate order data",
            scheduled_start=datetime(2024, 1, 15),
            estimated_duration_minutes=120
        )
        
        assert task.scheduled_start is not None
        assert task.estimated_duration_minutes == 120


class TestPolicyRecommendation:
    """Test PolicyRecommendation model."""
    
    def test_policy_recommendation_creation(self):
        """Test basic PolicyRecommendation creation."""
        recommendation = PolicyRecommendation(
            id="rec_1",
            source_id="db_users",
            policy_type="data_retention",
            priority=Priority.HIGH,
            title="Implement data retention policy",
            description="Set up automatic data deletion after 7 years"
        )
        
        assert recommendation.id == "rec_1"
        assert recommendation.policy_type == "data_retention"
        assert recommendation.title == "Implement data retention policy"
    
    def test_policy_recommendation_with_details(self):
        """Test PolicyRecommendation with implementation details."""
        recommendation = PolicyRecommendation(
            id="rec_2",
            source_id="db_customers",
            policy_type="access_control",
            priority=Priority.MEDIUM,
            title="Enhance access control",
            description="Implement role-based access control",
            implementation_steps=["Create roles", "Assign permissions", "Test access"],
            estimated_effort_hours=40
        )
        
        assert len(recommendation.implementation_steps) == 3
        assert recommendation.estimated_effort_hours == 40


class TestProcessingRequest:
    """Test ProcessingRequest model."""
    
    def test_processing_request_creation(self):
        """Test basic ProcessingRequest creation."""
        request = ProcessingRequest(
            id="req_1",
            source_id="db_users",
            request_type="data_quality_assessment",
            priority=Priority.HIGH,
            parameters={"check_duplicates": True, "validate_emails": True}
        )
        
        assert request.id == "req_1"
        assert request.request_type == "data_quality_assessment"
        assert request.parameters["check_duplicates"] is True
        assert request.status == "pending"
    
    def test_processing_request_with_schedule(self):
        """Test ProcessingRequest with scheduling."""
        request = ProcessingRequest(
            id="req_2",
            source_id="db_orders",
            request_type="anomaly_detection",
            priority=Priority.MEDIUM,
            parameters={"algorithm": "isolation_forest"},
            scheduled_start=datetime(2024, 2, 1),
            requester_id="user_123"
        )
        
        assert request.scheduled_start is not None
        assert request.requester_id == "user_123"


class TestAnalyticsVisualization:
    """Test AnalyticsVisualization model."""
    
    def test_analytics_visualization_creation(self):
        """Test basic AnalyticsVisualization creation."""
        viz = AnalyticsVisualization(
            id="viz_1",
            source_id="db_sales",
            title="Sales Trends",
            visualization_type="line_chart",
            data_config={"x_axis": "date", "y_axis": "revenue"}
        )
        
        assert viz.id == "viz_1"
        assert viz.title == "Sales Trends"
        assert viz.visualization_type == "line_chart"
        assert viz.data_config["x_axis"] == "date"
    
    def test_analytics_visualization_with_options(self):
        """Test AnalyticsVisualization with display options."""
        viz = AnalyticsVisualization(
            id="viz_2",
            source_id="db_users",
            title="User Growth",
            visualization_type="bar_chart",
            data_config={"groupby": "signup_month"},
            display_options={
                "theme": "dark",
                "show_legend": True,
                "colors": ["#1f77b4", "#ff7f0e"]
            }
        )
        
        assert viz.display_options["theme"] == "dark"
        assert viz.display_options["show_legend"] is True


class TestIntegrationScenarios:
    """Test integration scenarios with multiple data types."""
    
    def test_complete_data_quality_workflow(self):
        """Test a complete data quality assessment workflow."""
        # Create data source
        source = DataSource(
            id="customer_db",
            name="Customer Database",
            type="database"
        )
        
        # Create data record
        record = DataRecord(
            id="record_001",
            source_id=source.id,
            data={"id": 1, "email": "invalid-email", "phone": ""}
        )
        
        # Create data issue
        issue = DataIssue(
            id="issue_001",
            source_id=source.id,
            field_name="email",
            issue_type="validation",
            severity=Severity.HIGH,
            description="Invalid email format"
        )
        
        # Create remediation task
        task = RemediationTask(
            id="task_001",
            source_id=source.id,
            task_type="data_cleaning",
            priority=Priority.HIGH,
            description="Fix email validation issues"
        )
        
        # Verify the workflow
        assert source.id == record.source_id == issue.source_id == task.source_id
        assert issue.severity == Severity.HIGH
        assert task.priority == Priority.HIGH
    
    def test_compliance_policy_scenario(self):
        """Test compliance violation and policy recommendation."""
        # Create compliance violation
        violation = ComplianceViolation(
            id="violation_001",
            source_id="user_data",
            rule_id="gdpr_consent",
            severity=Severity.CRITICAL,
            description="Processing personal data without consent"
        )
        
        # Create policy recommendation
        recommendation = PolicyRecommendation(
            id="rec_001",
            source_id="user_data",
            policy_type="consent_management",
            priority=Priority.URGENT,
            title="Implement consent management system",
            description="Deploy consent tracking for GDPR compliance"
        )
        
        # Verify compliance scenario
        assert violation.severity == Severity.CRITICAL
        assert recommendation.priority == Priority.URGENT
        assert violation.source_id == recommendation.source_id
    
    def test_analytics_reporting_scenario(self):
        """Test analytics and reporting workflow."""
        # Create processing request for analytics
        request = ProcessingRequest(
            id="analytics_req_001",
            source_id="sales_data",
            request_type="trend_analysis",
            priority=Priority.MEDIUM,
            parameters={"time_period": "last_quarter", "metrics": ["revenue", "units"]}
        )
        
        # Create visualization
        viz = AnalyticsVisualization(
            id="sales_trends_viz",
            source_id="sales_data",
            title="Quarterly Sales Trends",
            visualization_type="multi_line_chart",
            data_config={
                "x_axis": "month",
                "y_axes": ["revenue", "units"],
                "groupby": "product_category"
            }
        )
        
        # Verify analytics scenario
        assert request.source_id == viz.source_id
        assert request.request_type == "trend_analysis"
        assert viz.visualization_type == "multi_line_chart"
