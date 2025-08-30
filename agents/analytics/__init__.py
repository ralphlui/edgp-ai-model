"""
Analytics Agent

This agent is responsible for:
- Providing analytics data (tabular/charts) for data quality and accuracy
- Interacting with Data Quality and Remediation agents for metrics and reports
- Dashboard generation and data visualization
"""

from .agent import AnalyticsAgent

__all__ = ["AnalyticsAgent"]
