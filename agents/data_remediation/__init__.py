"""
Data Remediation Agent

This agent is responsible for:
- Data remediation tasks after data ingestion
- Guiding users on how to remediate inaccurate data
- Providing remediation metrics to analytics agent
"""

from .agent import DataRemediationAgent

__all__ = ["DataRemediationAgent"]
