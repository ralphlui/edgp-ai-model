"""
Data Quality Agent

This agent is responsible for:
- Flagging data quality issues (anomalies, duplications)
- Firing remediation tasks for quality issues
- Providing data quality metrics to analytics agent
"""

from .agent import DataQualityAgent

__all__ = ["DataQualityAgent"]
