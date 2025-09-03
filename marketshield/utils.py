import hashlib
import json
from datetime import datetime
from typing import Any, Dict

def generate_hash(content: str) -> str:
    """Generate SHA-256 hash for content integrity."""
    return hashlib.sha256(content.encode('utf-8')).hexdigest()

def format_report(data: Dict[str, Any]) -> str:
    """Format analysis results into a readable report."""
    report = f"""
MarketShield Fraud Detection Report
==================================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Content Hash: {data.get('content_hash', 'N/A')}

RISK ASSESSMENT
--------------
Risk Level: {data.get('risk_assessment', {}).get('risk_level', 'Unknown')}
Risk Score: {data.get('risk_assessment', {}).get('risk_score', 0):.3f}

Key Risk Factors:
"""
    
    reasons = data.get('risk_assessment', {}).get('reasons', [])
    for i, reason in enumerate(reasons, 1):
        report += f"{i}. {reason}\n"
    
    report += "\nVERIFICATION RESULTS\n"
    report += "-------------------\n"
    
    verification = data.get('verification', {})
    
    # UPI verification
    upi_handles = verification.get('upi_handles', [])
    if upi_handles:
        report += "UPI Handles Found:\n"
        for upi in upi_handles:
            report += f"  - {upi['handle']}: {upi['status']}\n"
    
    # Registration verification
    registrations = verification.get('registrations', [])
    if registrations:
        report += "SEBI Registrations:\n"
        for reg in registrations:
            report += f"  - {reg['number']}: {reg['status']}\n"
    
    return report

def sanitize_input(content: str) -> str:
    """Sanitize input content for security."""
    # Remove potentially harmful patterns
    import re
    
    # Remove script tags and similar
    content = re.sub(r'<script.*?</script>', '', content, flags=re.IGNORECASE | re.DOTALL)
    content = re.sub(r'<.*?>', '', content)  # Remove HTML tags
    
    # Limit length
    if len(content) > 10000:
        content = content[:10000] + "... [truncated]"
    
    return content.strip()

def calculate_confidence_interval(score: float, sample_size: int = 100) -> tuple:
    """Calculate confidence interval for risk score."""
    import math
    
    # Simple confidence interval calculation
    margin = 1.96 * math.sqrt((score * (1 - score)) / sample_size)
    lower = max(0, score - margin)
    upper = min(1, score + margin)
    
    return (lower, upper)
