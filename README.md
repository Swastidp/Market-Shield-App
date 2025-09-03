# 🛡️ MarketShield - Securities Market Fraud Detection MVP

**Advanced AI-powered fraud prevention system for India's securities market**

Built for SEBI Securities Market Hackathon 2025 - Addressing the "Fraud Prevention" challenge.

## 🎯 Problem Statement

MarketShield tackles critical fraud prevention challenges in India's securities market:

- **Advisor Impersonation**: Detecting fraudulent advisors claiming SEBI registration
- **Deepfake Detection**: Identifying manipulated videos/audios of corporate leaders  
- **Social Media Fraud**: Flagging misleading stock tips in WhatsApp/Telegram groups
- **Fake Trading Apps**: Detecting apps mimicking trusted financial institutions
- **Document Verification**: Cross-verifying corporate announcements and regulatory letters

## 🚀 Key Features

### Multi-Source Content Analysis
- **URL Analysis**: Social media posts, news articles, websites
- **File Processing**: PDFs, images (OCR), text documents  
- **YouTube Integration**: Automatic transcript extraction and analysis
- **Direct Text**: WhatsApp/Telegram messages, announcements

### Advanced Fraud Detection
- **Risk Scoring**: Comprehensive 5-factor fraud assessment model
- **Pattern Recognition**: ML-powered detection of guarantee claims, impersonation, social manipulation
- **UPI Verification**: Integration with SEBI Check system (Oct 2025 rollout)
- **Registry Validation**: Cross-verification of claimed SEBI registrations

### RAG-Powered Explanations
- **Policy Context**: Citations from SEBI regulations and guidelines
- **Transparent Reasoning**: 3 specific reasons for each risk assessment
- **Audit Trail**: SHA-256 content hashing for evidence integrity

## 🛠️ Technology Stack

- **Framework**: Streamlit with Community Cloud deployment
- **AI/ML**: Google Gemini API (text-embedding-004, gemini-1.5-flash)
- **Vector Search**: FAISS IndexFlatIP for local RAG implementation
- **OCR**: EasyOCR + Pytesseract for image text extraction
- **Document Processing**: PyPDF, YouTube Transcript API
- **Security**: Cryptographic hashing, input sanitization

## ⚡ Quick Start

### Prerequisites
- Python 3.11+
- UV package manager
- Google Gemini API key

### Installation

1. **Clone and setup environment**:
