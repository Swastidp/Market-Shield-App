# 🛡️ MarketShield - Securities Market Fraud Detection MVP

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit%20Cloud-brightgreen)](https://appcom-swastidip-market-shield-app.streamlit.app/)
[![Python 3.12+](https://img.shields.io/badge/Python-3.12%2B-blue)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.49.1%2B-red)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

**Advanced AI-powered fraud prevention system for India's securities market**

*Built for SEBI Securities Market Hackathon 2025*

---

## 🚀 Live Demo

**[Try MarketShield Live →](https://appcom-swastidip-market-shield-app.streamlit.app/)**

---

## 📖 Overview

MarketShield is a comprehensive fraud detection system that automatically analyzes suspicious content from multiple channels to identify market manipulation schemes. The system provides real-time risk assessment, credential verification, and generates auditable evidence reports for regulatory compliance.

### 🎯 Problem Statement
- **₹1.38 trillion** lost to financial fraud in India annually
- **67%** increase in securities market fraud cases
- Sophisticated schemes using social media, PDFs, and multimedia content
- Manual verification is time-consuming and error-prone

### ✅ Solution
MarketShield provides automated fraud detection with:
- Multi-channel content ingestion (URLs, files, images, videos)
- AI-powered risk scoring with explainable results  
- Real-time UPI and SEBI registration verification
- Policy-aware context retrieval using RAG
- Comprehensive audit trails for compliance

---

## ✨ Key Features

### 🔍 **Multi-Channel Content Analysis**
- **URL Fetching**: Social media posts, news articles, websites
- **File Processing**: PDFs, text files, images with OCR
- **YouTube Integration**: Transcript extraction with fallbacks
- **Image Analysis**: OCR with EasyOCR + Tesseract backup
- **Video Processing**: Authenticity detection using OpenCV

### 🧠 **Advanced Risk Detection**
- **Pump-and-Dump**: Target prices, volume mentions, hype language
- **Guarantee Claims**: Unrealistic returns, time-bound promises
- **Impersonation**: SEBI/RBI claims, fake credentials
- **Social Manipulation**: Urgency tactics, FOMO, exclusive offers
- **Crypto Scams**: High-yield promises, DeFi schemes

### 🔐 **Real-Time Verification**
- **UPI Validation**: Format checking + SEBI Check simulation
- **SEBI Registry**: Registration number validation
- **Corporate Announcements**: Credibility assessment

### 🤖 **RAG-Powered Intelligence**
- **Policy Context**: Relevant SEBI guidelines retrieval
- **Semantic Search**: FAISS-powered similarity matching
- **Chunked Processing**: Handle large documents efficiently
- **Fallback Systems**: Graceful degradation when APIs unavailable

---

## 🛠️ Technology Stack

### **Frontend & Deployment**
- **Streamlit** - Interactive web interface and cloud deployment
- **Python 3.12+** - Core application runtime

### **AI & Machine Learning**
- **Google Gemini API** - Embeddings via `text-embedding-004`
- **FAISS** - Vector similarity search and indexing
- **NumPy/SciPy** - Numerical computing and statistics

### **Computer Vision & OCR**
- **OpenCV (Headless)** - Image/video processing
- **EasyOCR** - Primary OCR engine with lazy loading
- **Tesseract** - Fallback OCR system
- **PIL/Pillow** - Image preprocessing

### **Content Processing**
- **PyPDF** - PDF text extraction
- **YouTube Transcript API** - Video transcript extraction
- **yt-dlp** - Fallback subtitle extraction
- **Requests** - Web scraping with retry logic

### **Security & Performance**
- **Rate Limiting** - Per-session throttling
- **Caching System** - Memory + disk with TTL
- **Input Sanitization** - XSS prevention
- **Session Management** - Secure user tracking

---

## 🚀 Quick Start

### Prerequisites
- Python 3.12 or higher
- Git
- Gemini API key (optional, for RAG features)


