import streamlit as st
import hashlib
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import os

# Import MarketShield modules
from marketshield.ingest import ContentIngester
from marketshield.rag import RAGEngine
from marketshield.risk import RiskAnalyzer
from marketshield.verify import VerificationEngine
from marketshield.utils import generate_hash, format_report

# Page configuration
st.set_page_config(
    page_title="MarketShield - Fraud Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'rag_engine' not in st.session_state:
    try:
        st.session_state.rag_engine = RAGEngine()
    except Exception as e:
        st.error(f"Failed to initialize RAG engine: {e}")
        st.session_state.rag_engine = None

if 'risk_analyzer' not in st.session_state:
    st.session_state.risk_analyzer = RiskAnalyzer()

if 'verification_engine' not in st.session_state:
    st.session_state.verification_engine = VerificationEngine()

# Header
st.markdown("""
# 🛡️ MarketShield - Securities Market Fraud Detection
*Advanced AI-powered fraud prevention for India's securities market*
""")

# Sidebar - Input method selection
st.sidebar.header("📥 Input Method")
input_method = st.sidebar.radio(
    "Select input source:",
    ["URL Analysis", "File Upload", "YouTube Video", "Text Paste"]
)

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.header("Content Analysis")
    
    content_text = ""
    source_type = ""
    
    # Input handling based on selected method
    if input_method == "URL Analysis":
        url = st.text_input("🔗 Enter URL (social media, news, website):")
        if st.button("Analyze URL"):
            if url:
                with st.spinner("Fetching and analyzing content..."):
                    try:
                        ingester = ContentIngester()
                        content_text = ingester.fetch_url_content(url)
                        source_type = "url"
                    except Exception as e:
                        st.error(f"Failed to fetch URL: {e}")
    
    elif input_method == "File Upload":
        uploaded_file = st.file_uploader(
            "📄 Upload file (PDF, TXT, or Image):",
            type=['pdf', 'txt', 'png', 'jpg', 'jpeg']
        )
        if uploaded_file and st.button("Analyze File"):
            with st.spinner("Processing file..."):
                try:
                    ingester = ContentIngester()
                    content_text = ingester.process_file(uploaded_file)
                    source_type = "file"
                except Exception as e:
                    st.error(f"Failed to process file: {e}")
    
    elif input_method == "YouTube Video":
        youtube_url = st.text_input("🎥 Enter YouTube URL:")
        if st.button("Extract & Analyze"):
            if youtube_url:
                with st.spinner("Extracting transcript..."):
                    try:
                        ingester = ContentIngester()
                        content_text = ingester.extract_youtube_transcript(youtube_url)
                        source_type = "youtube"
                    except Exception as e:
                        st.error(f"Failed to extract transcript: {e}")
    
    elif input_method == "Text Paste":
        content_text = st.text_area(
            "📝 Paste content (WhatsApp/Telegram messages, announcements):",
            height=200
        )
        source_type = "text"
        if st.button("Analyze Text"):
            if not content_text.strip():
                st.warning("Please enter some content to analyze.")
                content_text = ""

    # Analysis results
    if content_text and content_text.strip():
        st.subheader("Analysis Results")
        
        with st.spinner("Analyzing for fraud indicators..."):
            # Generate content hash for audit trail
            content_hash = generate_hash(content_text)
            
            # Risk analysis
            risk_result = st.session_state.risk_analyzer.analyze_content(content_text)
            
            # RAG-powered explanation
            rag_context = None
            if st.session_state.rag_engine:
                rag_context = st.session_state.rag_engine.get_relevant_context(content_text)
            
            # Verification checks
            verification_results = st.session_state.verification_engine.verify_content(content_text)
            
            # Display risk badge
            risk_level = risk_result['risk_level']
            risk_score = risk_result['risk_score']
            
            if risk_level == "HIGH_RISK":
                st.error(f"🔴 **HIGH RISK** - Score: {risk_score:.2f}")
            elif risk_level == "UNCERTAIN":
                st.warning(f"🟡 **UNCERTAIN** - Score: {risk_score:.2f}")
            else:
                st.success(f"🟢 **LEGITIMATE** - Score: {risk_score:.2f}")
            
            # Risk factors
            st.subheader("🔍 Key Risk Factors")
            for i, reason in enumerate(risk_result['reasons'][:3], 1):
                st.write(f"{i}. {reason}")
            
            # Policy citations
            if rag_context:
                st.subheader("📋 SEBI Policy Context")
                for i, citation in enumerate(rag_context['citations'][:2], 1):
                    with st.expander(f"Citation {i}: {citation['title']}"):
                        st.write(citation['content'])
                        st.caption(f"Relevance: {citation['score']:.3f}")

with col2:
    st.header("Verification Status")
    
    if 'verification_results' in locals():
        # UPI Handle Verification
        st.subheader("💳 UPI Verification")
        upi_results = verification_results.get('upi_handles', [])
        if upi_results:
            for upi in upi_results:
                status_icon = "✅" if upi['verified'] else "⚠️"
                st.write(f"{status_icon} {upi['handle']} - {upi['status']}")
        else:
            st.write("No UPI handles detected")
        
        # Registry Verification
        st.subheader("📜 SEBI Registration")
        reg_results = verification_results.get('registrations', [])
        if reg_results:
            for reg in reg_results:
                status_icon = "✅" if reg['valid'] else "❌"
                st.write(f"{status_icon} {reg['number']} - {reg['status']}")
        else:
            st.write("No registration claims detected")
        
        # Deepfake Detection (placeholder)
        st.subheader("🎭 Media Authenticity")
        if source_type in ['youtube', 'file']:
            st.write("🔍 Analysis in progress...")
        else:
            st.write("Not applicable for text content")

    # Evidence Report
    st.header("📊 Evidence Report")
    
    if 'content_text' in locals() and content_text and 'risk_result' in locals():
        # Generate downloadable report
        report_data = {
            'timestamp': datetime.now().isoformat(),
            'content_hash': content_hash,
            'source_type': source_type,
            'risk_assessment': risk_result,
            'verification': verification_results if 'verification_results' in locals() else {},
            'rag_context': rag_context if rag_context else {}
        }
        
        report_json = json.dumps(report_data, indent=2)
        report_filename = f"marketshield_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        st.download_button(
            label="📥 Download Evidence Report",
            data=report_json,
            file_name=report_filename,
            mime="application/json"
        )
        
        # Display audit hash
        st.caption(f"🔐 Content Hash: `{content_hash[:16]}...`")

# Footer
st.markdown("---")
st.markdown("""
**MarketShield** - Built for SEBI Securities Market Hackathon 2025  
*Protecting investors through advanced AI-powered fraud detection*
""")
