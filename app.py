import streamlit as st
import hashlib
import json
import time
from datetime import datetime
from typing import Dict, Tuple, Optional

# Core modules
from marketshield.ingest import ContentIngester
from marketshield.rag import RAGEngine
from marketshield.risk import RiskAnalyzer
from marketshield.verify import VerificationEngine
from marketshield.utils import generate_hash, format_report

# Security / caching
from marketshield.security import SecurityManager, rate_limit
from marketshield.cache import CacheManager

# Deepfake capabilities overview (for sidebar status)
try:
    from marketshield.deepfake import DeepfakeDetector
    HAS_DEEPFAKE = True
except Exception:
    HAS_DEEPFAKE = False

# -----------------------------------------------------------------------------
# Page configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="MarketShield - Fraud Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# Initialize session state
# -----------------------------------------------------------------------------
def initialize_session_state():
    # Security Manager
    if 'security_manager' not in st.session_state:
        st.session_state.security_manager = SecurityManager()

    # Cache Manager
    if 'cache_manager' not in st.session_state:
        st.session_state.cache_manager = CacheManager()
    # Clean expired cache on startup
    st.session_state.cache_manager.clear_expired()

    # RAG Engine
    if 'rag_engine' not in st.session_state:
        try:
            with st.spinner("Initializing AI engine..."):
                st.session_state.rag_engine = RAGEngine()
        except Exception as e:
            st.error(f"⚠️ RAG engine initialization failed: {e}")
            st.session_state.rag_engine = None

    # Risk Analyzer
    if 'risk_analyzer' not in st.session_state:
        st.session_state.risk_analyzer = RiskAnalyzer()

    # Verification Engine
    if 'verification_engine' not in st.session_state:
        st.session_state.verification_engine = VerificationEngine()

    # Performance metrics
    if 'performance_metrics' not in st.session_state:
        st.session_state.performance_metrics = {
            'total_analyses': 0,
            'successful_analyses': 0,
            'avg_response_time': 0.0,
            'cache_hit_rate': 0.0,
        }

initialize_session_state()

# -----------------------------------------------------------------------------
# Helpers: analysis, fetching, file processing
# -----------------------------------------------------------------------------
@rate_limit('analysis')
def perform_comprehensive_analysis(content_text: str, source_type: str) -> Optional[Dict]:
    """Run risk analysis, RAG retrieval, and verification, with timing and caching."""
    start_time = time.time()
    try:
        st.session_state.performance_metrics['total_analyses'] += 1

        # Hash for audit trail
        content_hash = generate_hash(content_text)

        # Risk analysis (cached)
        risk_result = st.session_state.risk_analyzer.analyze_content(content_text)

        # RAG context (cached)
        rag_context = None
        if st.session_state.rag_engine:
            try:
                rag_context = st.session_state.rag_engine.get_relevant_context(content_text)
            except Exception as e:
                st.warning(f"RAG context unavailable: {e}")
                rag_context = {'citations': [], 'query': content_text}

        # Verification
        verification_results = st.session_state.verification_engine.verify_content(content_text)

        # Timing / metrics
        response_time = time.time() - start_time
        pm = st.session_state.performance_metrics
        pm['avg_response_time'] = (pm['avg_response_time'] * pm['successful_analyses'] + response_time) / (pm['successful_analyses'] + 1)
        pm['successful_analyses'] += 1

        return {
            'content_hash': content_hash,
            'risk_result': risk_result,
            'rag_context': rag_context,
            'verification_results': verification_results,
            'response_time': response_time,
            'timestamp': datetime.now(),
        }
    except Exception as e:
        st.error(f"Analysis failed: {e}")
        return None

@rate_limit('url_fetch')
def fetch_and_analyze_url(url: str) -> Optional[str]:
    """Fetch URL content with enhanced error handling."""
    try:
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
        ingester = ContentIngester()
        content = ingester.fetch_url_content(url)
        if not content or len(content.strip()) < 10:
            st.warning("⚠️ Very little content extracted from URL. Please verify the URL is accessible.")
            return None
        return content
    except Exception as e:
        st.error(f"Failed to fetch URL: {e}")
        return None

@rate_limit('file_analysis')
def process_uploaded_file(uploaded_file) -> Optional[Tuple[str, Dict]]:
    """Process uploaded file and always attempt authenticity check.

    Returns:
        (content_text, authenticity_result_dict or None)
    """
    try:
        ingester = ContentIngester()
        # Always request authenticity in the single canonical call:
        result = ingester.process_file(uploaded_file, include_authenticity_check=True)
        if isinstance(result, tuple):
            content, authenticity_result = result
        else:
            content, authenticity_result = result, None
        return content, authenticity_result
    except Exception as e:
        st.error(f"Failed to process file: {e}")
        return None, None

# -----------------------------------------------------------------------------
# Header with system status
# -----------------------------------------------------------------------------
col_header1, col_header2 = st.columns([3, 1])
with col_header1:
    st.markdown("""
# 🛡️ MarketShield - Securities Market Fraud Detection
*Advanced AI-powered fraud prevention for India's securities market*
""")

with col_header2:
    st.markdown("### System Status")
    rag_status = "🟢 Online" if st.session_state.rag_engine else "🔴 Offline"
    st.markdown(f"**AI Engine:** {rag_status}")
    cache_stats = st.session_state.cache_manager.get_stats()
    st.markdown(f"**Cache:** 🟢 {cache_stats['hit_rate']:.1%} Hit Rate")
    user_id = st.session_state.security_manager.get_user_id()
    st.markdown(f"**Session:** `{user_id[:8]}...`")

# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------
st.sidebar.header("📥 Input Method")
input_method = st.sidebar.radio(
    "Select input source:",
    ["URL Analysis", "File Upload", "YouTube Video", "Text Paste"],
)

st.sidebar.markdown("---")
st.sidebar.header("📊 Performance Metrics")
metrics = st.session_state.performance_metrics
col_metric1, col_metric2 = st.sidebar.columns(2)
with col_metric1:
    st.metric("Total Analyses", metrics['total_analyses'])
    success_rate = (metrics['successful_analyses'] / max(metrics['total_analyses'], 1))
    st.metric("Success Rate", f"{success_rate:.1%}")
with col_metric2:
    st.metric("Avg Response", f"{metrics['avg_response_time']:.2f}s")
    st.metric("Cache Hits", f"{cache_stats['hit_rate']:.1%}")

st.sidebar.markdown("---")
st.sidebar.header("🗄️ Cache Management")
if st.sidebar.button("Clear Cache"):
    st.session_state.cache_manager.memory_cache.clear()
    st.sidebar.success("Cache cleared!")
if st.sidebar.button("Clean Expired"):
    st.session_state.cache_manager.clear_expired()
    st.sidebar.success("Expired entries cleaned!")

with st.sidebar.expander("⚙️ Advanced Settings"):
    enable_detailed_logging = st.checkbox("Enable Detailed Logging", False)
    show_debug_info = st.checkbox("Show Debug Information", False)
    cache_ttl = st.slider("Cache TTL (minutes)", 5, 120, 30)

with st.sidebar.expander("🎭 Deepfake Capabilities"):
    if HAS_DEEPFAKE:
        try:
            caps = DeepfakeDetector().get_system_capabilities()
            st.write({
                "opencv_available": caps["dependencies"]["opencv"],
                "librosa_available": caps["dependencies"]["librosa"],
                "scipy_available": caps["dependencies"]["scipy"],
                "image_formats": caps["supported_formats"]["image"],
                "video_formats": caps["supported_formats"]["video"],
                "audio_formats": caps["supported_formats"]["audio"],
            })
        except Exception as e:
            st.info(f"Capability check unavailable: {e}")
    else:
        st.info("Deepfake module not importable in this environment.")

# -----------------------------------------------------------------------------
# Main content
# -----------------------------------------------------------------------------
col1, col2 = st.columns([2, 1])

with col1:
    st.header("Content Analysis")
    content_text = ""
    source_type = ""
    authenticity_result = None

    if input_method == "URL Analysis":
        url = st.text_input("🔗 Enter URL (social media, news, website):", placeholder="https://example.com/suspicious-content")
        if st.button("Analyze URL", type="primary"):
            if url:
                with st.spinner("🔍 Fetching and analyzing content..."):
                    content_text = fetch_and_analyze_url(url)
                    source_type = "url"
                if content_text:
                    st.success(f"✅ Successfully extracted {len(content_text)} characters")
            else:
                st.warning("⚠️ Please enter a valid URL.")

    elif input_method == "File Upload":
        uploaded_file = st.file_uploader(
            "📄 Upload file (PDF, TXT, Image, or Video):",
            type=['pdf', 'txt', 'png', 'jpg', 'jpeg', 'mp4', 'mov', 'avi'],
            help="Supported: PDF, text, images (OCR), short videos (authenticity)",
        )
        if uploaded_file:
            with st.expander("📋 File Details"):
                st.write(f"**Filename:** {uploaded_file.name}")
                st.write(f"**File size:** {uploaded_file.size / 1024:.1f} KB")
                st.write(f"**File type:** {uploaded_file.type}")

        if uploaded_file and st.button("Analyze File", type="primary"):
            with st.spinner("📄 Processing file..."):
                result = process_uploaded_file(uploaded_file)
                if result and result:
                    content_text, authenticity_result = result
                    source_type = "file"
                    # Persist authenticity so the right panel can render it immediately
                    st.session_state.current_authenticity_result = authenticity_result
                    st.success(f"✅ File processed successfully! Extracted {len(content_text)} characters")

    elif input_method == "YouTube Video":
        youtube_url = st.text_input("🎥 Enter YouTube URL:", placeholder="https://www.youtube.com/watch?v=VIDEO_ID")
        if st.button("Extract & Analyze", type="primary"):
            if youtube_url:
                with st.spinner("🎥 Extracting transcript..."):
                    try:
                        ingester = ContentIngester()
                        content_text = ingester.extract_youtube_transcript(youtube_url)
                        source_type = "youtube"
                        if content_text:
                            st.success(f"✅ Transcript extracted: {len(content_text)} characters")
                        else:
                            st.error("❌ No transcript available for this video")
                    except Exception as e:
                        st.error(f"❌ Failed to extract transcript: {e}")

    elif input_method == "Text Paste":
        content_text = st.text_area(
            "📝 Paste content (WhatsApp/Telegram messages, announcements):",
            height=200,
            placeholder="Paste suspicious content here for analysis...",
        )
        source_type = "text"
        if st.button("Analyze Text", type="primary"):
            if not content_text.strip():
                st.warning("⚠️ Please enter some content to analyze.")
                content_text = ""
            else:
                st.success(f"✅ Ready to analyze {len(content_text)} characters")

    # ------------------- Run analysis and display results -------------------
    if content_text and content_text.strip():
        st.markdown("---")
        st.header("🔍 Analysis Results")
        with st.spinner("🧠 Analyzing for fraud indicators..."):
            analysis_result = perform_comprehensive_analysis(content_text, source_type)

        if analysis_result is None:
            st.error("❌ Analysis aborted due to rate limiting or errors.")
        else:
            # Extract results
            content_hash = analysis_result['content_hash']
            risk_result = analysis_result['risk_result']
            rag_context = analysis_result['rag_context']
            verification_results = analysis_result['verification_results']
            response_time = analysis_result['response_time']

            # Performance indicator
            if st.session_state.get("show_debug_info", False):
                st.info(f"⚡ Analysis completed in {response_time:.2f} seconds")

            # Risk Assessment Display
            risk_level = risk_result['risk_level']
            risk_score = risk_result['risk_score']
            if risk_level in ("HIGH_RISK", "CRITICAL_RISK"):
                st.error("🔴 **HIGH RISK DETECTED**")
                st.progress(min(risk_score, 1.0), text=f"Risk Score: {risk_score:.3f}")
            elif risk_level == "UNCERTAIN":
                st.warning("🟡 **UNCERTAIN - REQUIRES REVIEW**")
                st.progress(risk_score, text=f"Risk Score: {risk_score:.3f}")
            else:
                st.success("🟢 **APPEARS LEGITIMATE**")
                st.progress(risk_score, text=f"Risk Score: {risk_score:.3f}")

            # Component scores breakdown (optional details)
            if st.session_state.get("show_debug_info", False):
                with st.expander("🔍 Detailed Risk Breakdown"):
                    component_scores = risk_result.get('component_scores', {})
                    for component, score in component_scores.items():
                        st.write(f"**{component.replace('_', ' ').title()}:** {score:.3f}")

            # Key Risk Factors
            st.subheader("⚠️ Key Risk Factors")
            reasons = risk_result.get('reasons', [])
            if reasons:
                for i, reason in enumerate(reasons[:3], 1):
                    st.write(f"{i}. {reason}")
            else:
                st.write("No significant risk factors detected.")

            # SEBI Policy Context
            if rag_context and rag_context.get('citations'):
                st.subheader("📋 Relevant SEBI Policies")
                for i, citation in enumerate(rag_context['citations'][:2], 1):
                    with st.expander(f"📄 {citation['title']} (Relevance: {citation['score']:.1%})"):
                        st.write(citation['content'])
                        st.caption(f"Source ID: {citation.get('id', 'N/A')}")

            # Store results for sidebar display
            st.session_state.current_verification_results = verification_results
            # Keep any immediate authenticity_result derived during file processing
            if authenticity_result and 'authenticity_score' in authenticity_result:
                st.session_state.current_authenticity_result = authenticity_result

with col2:
    st.header("🔐 Verification Status")

    # UPI Verification
    if hasattr(st.session_state, 'current_verification_results'):
        verification_results = st.session_state.current_verification_results
        st.subheader("💳 UPI Verification")
        upi_results = verification_results.get('upi_handles', [])
        if upi_results:
            for upi in upi_results:
                status_icon = "✅" if upi['verified'] else "⚠️"
                st.write(f"{status_icon} `{upi['handle']}`")
                st.caption(upi['status'])
        else:
            st.write("✅ No UPI handles detected")

        # SEBI Registration
        st.subheader("📜 SEBI Registration")
        reg_results = verification_results.get('registrations', [])
        if reg_results:
            for reg in reg_results:
                status_icon = "✅" if reg['valid'] else "❌"
                st.write(f"{status_icon} `{reg['number']}`")
                st.caption(reg['status'])
        else:
            st.write("✅ No registration claims detected")

        # Corporate Announcements
        corp_results = verification_results.get('corporate_announcements', {})
        if corp_results.get('is_announcement'):
            st.subheader("📢 Corporate Announcement")
            credibility = corp_results.get('credibility_score', 0)
            if credibility > 0.8:
                st.success(f"✅ High credibility ({credibility:.1%})")
            elif credibility > 0.5:
                st.warning(f"⚠️ Medium credibility ({credibility:.1%})")
            else:
                st.error(f"❌ Low credibility ({credibility:.1%})")

    # Media Authenticity
    st.subheader("🎭 Media Authenticity")
    auth_result = getattr(st.session_state, 'current_authenticity_result', None)
    if auth_result and 'authenticity_score' in auth_result:
        score = auth_result['authenticity_score']
        if score > 0.7:
            st.success(f"✅ Authentic ({score:.1%})")
        elif score > 0.4:
            st.warning(f"⚠️ Uncertain ({score:.1%})")
        else:
            st.error(f"❌ Suspicious ({score:.1%})")
        if auth_result.get('confidence'):
            st.caption(f"Confidence: {auth_result['confidence']}")
        if auth_result.get('analysis_method'):
            st.caption(f"Method: {auth_result['analysis_method']}")
    else:
        st.info("Upload an image or video to run authenticity checks.")

# -----------------------------------------------------------------------------
# Evidence Report & Audit Trail
# -----------------------------------------------------------------------------
st.markdown("---")
st.header("📊 Evidence Report & Audit Trail")

if 'content_text' in locals() and content_text and 'risk_result' in locals():
    col_report1, col_report2 = st.columns([2, 1])
    with col_report1:
        # Collect latest cache stats on demand
        cache_stats = st.session_state.cache_manager.get_stats()
        report_data = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'content_hash': generate_hash(content_text),
                'source_type': source_type,
                'analysis_version': '2.0',
                'user_session': st.session_state.security_manager.get_user_id()[:8],
            },
            'risk_assessment': risk_result,
            'verification': st.session_state.current_verification_results if hasattr(st.session_state, 'current_verification_results') else {},
            'rag_context': rag_context if 'rag_context' in locals() and rag_context else {},
            'authenticity_check': getattr(st.session_state, 'current_authenticity_result', {}) or {},
            'performance_metrics': {
                'response_time': analysis_result['response_time'] if 'analysis_result' in locals() and analysis_result else None,
                'cache_used': cache_stats['hit_rate'] > 0,
            },
        }

        with st.expander("📋 Report Preview"):
            st.json(report_data, expanded=False)

        report_json = json.dumps(report_data, indent=2, default=str)
        report_filename = f"marketshield_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        col_dl1, col_dl2 = st.columns(2)
        with col_dl1:
            st.download_button(
                label="📥 Download Evidence Report",
                data=report_json,
                file_name=report_filename,
                mime="application/json",
                type="primary",
            )
        with col_dl2:
            human_report = format_report(report_data)
            st.download_button(
                label="📄 Download Text Report",
                data=human_report,
                file_name=report_filename.replace('.json', '.txt'),
                mime="text/plain",
            )

    with col_report2:
        st.subheader("🔐 Audit Trail")
        content_hash = generate_hash(content_text)
        st.write("**Content Hash:**")
        st.code(content_hash[:16] + "...", language=None)
        st.write("**Analysis Time:**")
        st.code(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        st.write("**Cache Status:**")
        cache_stats = st.session_state.cache_manager.get_stats()
        if cache_stats['hit_rate'] > 0:
            st.success("✅ Cache utilized")
        else:
            st.info("ℹ️ Fresh analysis")

# -----------------------------------------------------------------------------
# Footer
# -----------------------------------------------------------------------------
st.markdown("---")
col_footer1, col_footer2, col_footer3 = st.columns(3)
with col_footer1:
    st.markdown("""
**MarketShield v2.0**
Built for SEBI Securities Market Hackathon 2025
""")
with col_footer2:
    st.markdown("""
**Features:**
- AI-powered fraud detection
- Real-time verification
- Comprehensive audit trails
""")
with col_footer3:
    st.markdown("""
**Security:**
- Rate limiting enabled
- Content sanitization
- Encrypted API keys
""")

# Debug info (optional)
if st.session_state.get("show_debug_info", False) and 'enable_detailed_logging' in locals() and enable_detailed_logging:
    with st.expander("🐛 Debug Information"):
        st.write("**Session State Keys:**", list(st.session_state.keys()))
        st.write("**Cache Statistics:**", st.session_state.cache_manager.get_stats())
        st.write("**Performance Metrics:**", st.session_state.performance_metrics)
