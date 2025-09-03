"""
MarketShield - Securities Market Fraud Detection MVP
Advanced AI-powered fraud prevention for India's securities market
"""

__version__ = "0.1.0"
__author__ = "SEBI Hackathon Team"

# Optional: You can import main classes here for easier access
from .ingest import ContentIngester
from .rag import RAGEngine
from .risk import RiskAnalyzer
from .verify import VerificationEngine
from .utils import generate_hash, format_report

__all__ = [
    'ContentIngester',
    'RAGEngine', 
    'RiskAnalyzer',
    'VerificationEngine',
    'generate_hash',
    'format_report'
]
