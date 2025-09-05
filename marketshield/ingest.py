import requests
import easyocr
import pytesseract
from pypdf import PdfReader
from youtube_transcript_api import YouTubeTranscriptApi
import re
import os
import time
import tempfile
from typing import Optional, Dict, Any, Tuple, Union
from PIL import Image
import io
import streamlit as st
import hashlib
import logging
from urllib.parse import urlparse, parse_qs
import html
import json
from yt_dlp import YoutubeDL

# Import our custom modules
try:
    from .security import rate_limit
    from .cache import cached_analysis
    from .deepfake import DeepfakeDetector
except ImportError:
    # Fallback decorators if modules not available
    def rate_limit(action: str):
        def decorator(func):
            return func
        return decorator
    
    def cached_analysis(analysis_type: str, ttl: int = 3600):
        def decorator(func):
            return func
        return decorator
    
    class DeepfakeDetector:
        def analyze_image_authenticity(self, image_data):
            return {'authenticity_score': 0.5, 'is_authentic': None, 'confidence': 'Low', 'analysis': {}}
        
        def analyze_video_authenticity(self, video_path: str):
            return {'authenticity_score': 0.5, 'is_authentic': None, 'confidence': 'Low', 'analysis': {}}

class ContentIngester:
    """Advanced content ingestion with security, caching, and authenticity verification."""
    
    def __init__(self):
        """Initialize the content ingester with all components."""
        self._setup_logging()
        self._initialize_ocr()
        self._initialize_deepfake_detector()
        self._setup_request_session()
        
        # Configuration
        self.max_content_length = 50000  # 50KB max content
        self.request_timeout = 30
        self.supported_image_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        self.supported_video_formats = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
        
    def _setup_logging(self):
        """Setup logging for debugging."""
        self.logger = logging.getLogger(__name__)
        
    def _initialize_ocr(self):
        """Initialize OCR engines with fallback."""
        self.ocr_reader = None
        self.use_tesseract_fallback = False
        
        try:
            self.ocr_reader = easyocr.Reader(['en'], gpu=False, verbose=False)
            self.logger.info("EasyOCR initialized successfully")
        except Exception as e:
            self.logger.warning(f"EasyOCR initialization failed: {e}. Using Tesseract fallback.")
            self.use_tesseract_fallback = True
            st.warning("⚠️ EasyOCR initialization failed. Using Tesseract as fallback.")
    
    def _initialize_deepfake_detector(self):
        """Initialize deepfake detection."""
        try:
            self.deepfake_detector = DeepfakeDetector()
        except Exception as e:
            self.logger.warning(f"Deepfake detector initialization failed: {e}")
            self.deepfake_detector = None
    
    def _setup_request_session(self):
        """Setup requests session with proper headers."""
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none'
        })

    # ==========================================================================
    # NEW ROBUST YOUTUBE TRANSCRIPT EXTRACTION METHODS
    # ==========================================================================
    
    def _env_or_secret(self, name, default=None):
        """Helper to read from env or Streamlit secrets if available."""
        try:
            return os.getenv(name) or st.secrets.get(name, default)
        except Exception:
            return os.getenv(name, default)

    def _build_proxies(self):
        """
        Build a requests-compatible proxies dict from environment variables:
          HTTPS_PROXY / HTTP_PROXY or YT_PROXY_HTTPS / YT_PROXY_HTTP
        """
        https_proxy = self._env_or_secret("YT_PROXY_HTTPS") or os.getenv("HTTPS_PROXY")
        http_proxy = self._env_or_secret("YT_PROXY_HTTP") or os.getenv("HTTP_PROXY")
        proxies = {}
        if https_proxy:
            proxies["https"] = https_proxy
        if http_proxy:
            proxies["http"] = http_proxy
        return proxies or None  # youtube-transcript-api expects None when not used

    def _cookies_path(self):
        """
        Optional cookies file path exported from browser (Netscape/cookies.txt format).
        Set via YT_COOKIES_PATH or Streamlit secrets. 
        """
        path = self._env_or_secret("YT_COOKIES_PATH")
        return path if path and os.path.exists(path) else None

    def _clean_vtt_or_srt(self, text: str) -> str:
        """Clean VTT/SRT subtitle formats to plain text."""
        # Remove WEBVTT header
        text = re.sub(r"^\s*WEBVTT.*?$", "", text, flags=re.IGNORECASE | re.MULTILINE)
        # Remove srt indices (lines with only digits)
        text = re.sub(r"(?m)^\s*\d+\s*$", "", text)
        # Remove timestamps like 00:00:01.000 --> 00:00:02.000 or 00:00:01,000 --> 00:00:02,000
        text = re.sub(r"\d{2}:\d{2}:\d{2}[.,]\d{3}\s*-->\s*\d{2}:\d{2}:\d{2}[.,]\d{3}.*", "", text)
        # Remove cues/markup
        text = re.sub(r"<[^>]+>", " ", text)
        # Remove common subtitle artifacts
        text = re.sub(r'\[Music\]|\[Applause\]|\[Laughter\]|\[.*?\]', '', text)
        # Collapse whitespace
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _fetch_text_url(self, url: str) -> str:
        """Fetch caption file (vtt/srt) and return text."""
        try:
            proxies = self._build_proxies()
            r = requests.get(url, timeout=20, proxies=proxies)
            r.raise_for_status()
            return r.text
        except Exception as e:
            raise Exception(f"Failed to fetch subtitle URL: {str(e)}")

    def _yt_dlp_transcript(self, url_or_id: str) -> str:
        """
        Fallback: use yt-dlp to get uploaded or auto-generated subtitles, 
        prefer English, else pick the first available track and return plain text.
        """
        ydl_opts = {
            "skip_download": True,
            "writesubtitles": True,          # allow authored subs
            "writeautomaticsub": True,       # allow auto subs
            "subtitlesformat": "vtt",        # easier to clean than srt
            "quiet": True,
            "no_warnings": True,
        }
        
        # Pass cookies to yt-dlp if available
        cookies = self._cookies_path()
        if cookies:
            ydl_opts["cookiefile"] = cookies
            
        # Pass proxy if available
        proxies = self._build_proxies()
        if proxies and proxies.get("https"):
            ydl_opts["proxy"] = proxies["https"]

        try:
            with YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url_or_id, download=False)
                subs = info.get("subtitles") or {}
                auto = info.get("automatic_captions") or {}
                
                # Prefer uploaded English subs, then other uploaded, then auto English, then any auto
                ordered_langs = ["en", "en-US", "en-GB", "en-IN"]
                
                def _pick_track(bucket):
                    """bucket: dict lang -> [ {url, ext, ...}, ... ]"""
                    # Try exact language matches first
                    for lang in ordered_langs:
                        for key, candidates in bucket.items():
                            if key.lower() == lang.lower() and candidates:
                                return candidates[-1].get("url")
                    
                    # Try language prefix matches (e.g., 'en' matches 'en-*')
                    for lang in ["en"]:
                        for key, candidates in bucket.items():
                            if key.lower().startswith(lang.lower()) and candidates:
                                return candidates[-1].get("url")
                    
                    # If nothing matched, pick first available language
                    for _, candidates in bucket.items():
                        if candidates:
                            return candidates[-1].get("url")
                    return None

                url_choice = _pick_track(subs) or _pick_track(auto)
                if not url_choice:
                    raise RuntimeError("No subtitles available via yt-dlp")

                raw = self._fetch_text_url(url_choice)
                cleaned = self._clean_vtt_or_srt(raw)
                
                if not cleaned or len(cleaned.strip()) < 10:
                    raise RuntimeError("Subtitle content is empty or too short")
                
                return cleaned
                
        except Exception as e:
            raise Exception(f"yt-dlp transcript extraction failed: {str(e)}")

    @rate_limit('url_fetch')
    @cached_analysis('youtube_transcript', ttl=7200)  # 2 hours cache
    def extract_youtube_transcript(self, url: str) -> str:
        """
        Robust YouTube transcript extraction pipeline:
          1) youtube-transcript-api with proxies/cookies and multi-language
          2) list_transcripts variants, including translated
          3) yt-dlp captions fallback (uploaded or auto)
        """
        try:
            self.logger.info(f"Starting YouTube transcript extraction for: {url}")
            
            # Normalize to video_id if possible
            vid = self._extract_video_id(url)
            self.logger.info(f"Extracted video ID: {vid}")
            
            proxies = self._build_proxies()
            cookies = self._cookies_path()
            
            # METHOD 1: Direct get_transcript with language preferences
            lang_prefs = ["en", "en-US", "en-GB", "en-IN"]
            try:
                self.logger.info("Method 1: Trying direct transcript retrieval")
                tlist = YouTubeTranscriptApi.get_transcript(
                    vid, 
                    languages=lang_prefs, 
                    proxies=proxies, 
                    cookies=cookies
                )
                transcript_text = " ".join(seg["text"] for seg in tlist).strip()
                if transcript_text and len(transcript_text) > 10:
                    self.logger.info("Method 1: Success")
                    return self._clean_transcript_text(transcript_text)
            except Exception as e:
                self.logger.warning(f"Method 1 failed: {e}")

            # METHOD 2: Explore transcript list and try manual/generated and translated
            try:
                self.logger.info("Method 2: Trying transcript list exploration")
                tlist_obj = YouTubeTranscriptApi.list_transcripts(vid, proxies=proxies, cookies=cookies)
                
                # Prefer manual English
                for tr in tlist_obj:
                    try:
                        if not tr.is_generated and tr.language_code in ["en", "en-US", "en-GB", "en-IN"]:
                            fetched = tr.fetch()
                            transcript_text = " ".join(seg["text"] for seg in fetched).strip()
                            if transcript_text and len(transcript_text) > 10:
                                self.logger.info("Method 2a: Manual English transcript found")
                                return self._clean_transcript_text(transcript_text)
                    except Exception:
                        continue
                
                # Then any manual
                for tr in tlist_obj:
                    try:
                        if not tr.is_generated:
                            fetched = tr.fetch()
                            transcript_text = " ".join(seg["text"] for seg in fetched).strip()
                            if transcript_text and len(transcript_text) > 10:
                                self.logger.info("Method 2b: Manual transcript found")
                                return self._clean_transcript_text(transcript_text)
                    except Exception:
                        continue
                
                # Then generated English
                for tr in tlist_obj:
                    try:
                        if tr.is_generated and tr.language_code.startswith("en"):
                            fetched = tr.fetch()
                            transcript_text = " ".join(seg["text"] for seg in fetched).strip()
                            if transcript_text and len(transcript_text) > 10:
                                self.logger.info("Method 2c: Auto-generated English transcript found")
                                return self._clean_transcript_text(transcript_text)
                    except Exception:
                        continue
                
                # Then translated to English
                for tr in tlist_obj:
                    try:
                        translated = tr.translate("en")
                        fetched = translated.fetch()
                        transcript_text = " ".join(seg["text"] for seg in fetched).strip()
                        if transcript_text and len(transcript_text) > 10:
                            self.logger.info("Method 2d: Translated transcript found")
                            return self._clean_transcript_text(transcript_text)
                    except Exception:
                        continue
                        
            except Exception as e:
                self.logger.warning(f"Method 2 failed: {e}")

            # METHOD 3: yt-dlp fallback
            try:
                self.logger.info("Method 3: Trying yt-dlp fallback")
                transcript_text = self._yt_dlp_transcript(url)
                if transcript_text and len(transcript_text.strip()) > 10:
                    self.logger.info("Method 3: Success")
                    cleaned_text = self._clean_transcript_text(transcript_text)
                    
                    # Truncate if necessary
                    if len(cleaned_text) > self.max_content_length:
                        cleaned_text = cleaned_text[:self.max_content_length] + "... [transcript truncated]"
                        st.info(f"📺 Transcript truncated to {self.max_content_length} characters")
                    
                    return cleaned_text
                    
            except Exception as e:
                self.logger.warning(f"Method 3 failed: {e}")

            # All methods failed
            error_details = self._classify_transcript_error("All transcript retrieval methods failed")
            raise Exception(f"YouTube transcript extraction failed: {error_details}")
            
        except Exception as e:
            error_msg = f"YouTube transcript extraction failed: {str(e)}"
            self.logger.error(error_msg)
            raise Exception(error_msg)

    def _extract_video_id(self, url: str) -> str:
        """Extract video ID from various YouTube URL formats."""
        # Clean URL
        url = url.strip()
        
        # YouTube URL patterns
        patterns = [
            # Standard watch URLs
            r'(?:youtube\.com/watch\?v=)([a-zA-Z0-9_-]{11})',
            r'(?:youtube\.com/watch\?.*&v=)([a-zA-Z0-9_-]{11})',
            # Short URLs
            r'(?:youtu\.be/)([a-zA-Z0-9_-]{11})',
            # Embed URLs
            r'(?:youtube\.com/embed/)([a-zA-Z0-9_-]{11})',
            # Mobile URLs
            r'(?:m\.youtube\.com/watch\?v=)([a-zA-Z0-9_-]{11})',
            # YouTube shorts
            r'(?:youtube\.com/shorts/)([a-zA-Z0-9_-]{11})',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                video_id = match.group(1)
                # Validate video ID format
                if re.match(r'^[a-zA-Z0-9_-]{11}$', video_id):
                    return video_id
        
        # If no pattern matches, check if it's a direct video ID
        if re.match(r'^[a-zA-Z0-9_-]{11}$', url):
            return url
        
        raise ValueError(f"Could not extract valid video ID from URL: {url}")
    
    def _classify_transcript_error(self, error_message: str) -> str:
        """Classify transcript extraction error for user-friendly messages."""
        error_lower = error_message.lower()
        
        if any(keyword in error_lower for keyword in ['disabled', 'not available', 'no transcript']):
            return "This video does not have transcripts/captions enabled."
        elif any(keyword in error_lower for keyword in ['private', 'unavailable', 'deleted']):
            return "This video is private, deleted, or restricted."
        elif any(keyword in error_lower for keyword in ['quota', 'limit', 'rate']):
            return "API rate limit reached. Please try again later."
        elif any(keyword in error_lower for keyword in ['network', 'connection', 'timeout']):
            return "Network connection issue. Please check your internet connection."
        elif any(keyword in error_lower for keyword in ['consent', 'cookies', 'age']):
            return "This video requires consent/age verification. Try setting YT_COOKIES_PATH."
        else:
            return "The video may not have captions available or may be restricted."
    
    def _clean_transcript_text(self, text: str) -> str:
        """Clean and normalize transcript text."""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove common transcript artifacts
        text = re.sub(r'\[Music\]|\[Applause\]|\[Laughter\]|\[.*?\]', '', text)
        
        # Fix common OCR/transcript errors
        text = re.sub(r'\bum\b|\buh\b|\ber\b', '', text)  # Remove filler words
        text = re.sub(r'\.{2,}', '.', text)  # Multiple periods
        
        # Normalize spacing
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        return text

    # ==========================================================================
    # EXISTING METHODS (URL, FILE PROCESSING, ETC.)
    # ==========================================================================
    
    @rate_limit('url_fetch')
    @cached_analysis('url_content', ttl=1800)  # 30 minutes cache
    def fetch_url_content(self, url: str) -> str:
        """Fetch and extract text content from URL with enhanced error handling."""
        try:
            # Validate URL
            parsed_url = urlparse(url)
            if not parsed_url.scheme or not parsed_url.netloc:
                raise ValueError("Invalid URL format")
            
            # Security check for suspicious URLs
            if self._is_suspicious_url(url):
                st.warning("⚠️ Potentially suspicious URL detected. Proceed with caution.")
            
            self.logger.info(f"Fetching content from: {url}")
            
            # Make request with timeout and retries
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = self.session.get(
                        url, 
                        timeout=self.request_timeout,
                        verify=True,
                        allow_redirects=True,
                        stream=True
                    )
                    response.raise_for_status()
                    
                    # Check content length
                    content_length = response.headers.get('content-length')
                    if content_length and int(content_length) > self.max_content_length * 10:  # 500KB limit for URLs
                        raise ValueError("Content too large to process")
                    
                    # Get content with encoding detection
                    content = self._get_response_content(response)
                    break
                    
                except requests.exceptions.Timeout:
                    if attempt == max_retries - 1:
                        raise Exception(f"Request timeout after {max_retries} attempts")
                    time.sleep(2 ** attempt)  # Exponential backoff
                except requests.exceptions.RequestException as e:
                    if attempt == max_retries - 1:
                        raise Exception(f"Request failed: {str(e)}")
                    time.sleep(2 ** attempt)
            
            # Extract and clean text content
            cleaned_content = self._extract_text_from_html(content)
            
            # Validate content
            if len(cleaned_content.strip()) < 10:
                raise ValueError("Insufficient content extracted from URL")
            
            # Truncate if too long
            if len(cleaned_content) > self.max_content_length:
                cleaned_content = cleaned_content[:self.max_content_length] + "... [truncated]"
                st.info(f"📄 Content truncated to {self.max_content_length} characters for analysis")
            
            self.logger.info(f"Successfully extracted {len(cleaned_content)} characters from URL")
            return cleaned_content
            
        except Exception as e:
            error_msg = f"Failed to fetch URL content: {str(e)}"
            self.logger.error(error_msg)
            raise Exception(error_msg)
    
    def _is_suspicious_url(self, url: str) -> bool:
        """Basic suspicious URL detection."""
        suspicious_patterns = [
            r'bit\.ly', r'tinyurl', r'shorturl', 
            r'\d+\.\d+\.\d+\.\d+',  # IP addresses
            r'[a-z]{20,}\.com',  # Very long domain names
            r'telegram\.me', r'telegram\.org',  # Telegram links
        ]
        
        return any(re.search(pattern, url.lower()) for pattern in suspicious_patterns)
    
    def _get_response_content(self, response) -> str:
        """Extract content with proper encoding detection."""
        # Try to get encoding from response
        encoding = response.encoding
        if not encoding or encoding.lower() == 'iso-8859-1':
            # Try to detect encoding from content
            try:
                import chardet
                raw_content = response.content
                detected = chardet.detect(raw_content)
                encoding = detected.get('encoding', 'utf-8')
            except ImportError:
                encoding = 'utf-8'
        
        try:
            return response.content.decode(encoding, errors='ignore')
        except (UnicodeDecodeError, LookupError):
            return response.content.decode('utf-8', errors='ignore')
    
    def _extract_text_from_html(self, content: str) -> str:
        """Enhanced HTML text extraction."""
        # Unescape HTML entities
        content = html.unescape(content)
        
        # Remove script and style elements completely
        content = re.sub(r'<script[^>]*>.*?</script>', '', content, flags=re.DOTALL | re.IGNORECASE)
        content = re.sub(r'<style[^>]*>.*?</style>', '', content, flags=re.DOTALL | re.IGNORECASE)
        content = re.sub(r'<noscript[^>]*>.*?</noscript>', '', content, flags=re.DOTALL | re.IGNORECASE)
        
        # Remove HTML comments
        content = re.sub(r'<!--.*?-->', '', content, flags=re.DOTALL)
        
        # Remove all HTML tags but preserve text
        content = re.sub(r'<[^>]+>', ' ', content)
        
        # Clean up whitespace
        content = re.sub(r'\s+', ' ', content)
        content = re.sub(r'\n\s*\n', '\n', content)
        
        # Remove URLs from text
        content = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', content)
        
        return content.strip()
    
    @rate_limit('analysis')
    def process_file(self, uploaded_file, include_authenticity_check: bool = True) -> Union[str, Tuple[str, Dict[str, Any]]]:
        """Process uploaded file with optional authenticity verification."""
        try:
            if uploaded_file is None:
                raise ValueError("No file uploaded")
            
            file_extension = os.path.splitext(uploaded_file.name)[1].lower()
            file_size = len(uploaded_file.read())
            uploaded_file.seek(0)  # Reset file pointer
            
            # File size check
            max_size = 50 * 1024 * 1024  # 50MB
            if file_size > max_size:
                raise ValueError(f"File too large: {file_size / (1024*1024):.1f}MB (max: 50MB)")
            
            self.logger.info(f"Processing file: {uploaded_file.name} ({file_size / 1024:.1f}KB)")
            
            content = ""
            authenticity_result = None
            
            # Process based on file type
            if file_extension == '.pdf':
                content = self._process_pdf(uploaded_file)
            elif file_extension == '.txt':
                content = self._process_text_file(uploaded_file)
            elif file_extension in self.supported_image_formats:
                content = self._process_image(uploaded_file)
                if include_authenticity_check and self.deepfake_detector:
                    authenticity_result = self._check_image_authenticity(uploaded_file)
            elif file_extension in self.supported_video_formats:
                content = "Video file uploaded - content analysis not implemented"
                if include_authenticity_check and self.deepfake_detector:
                    authenticity_result = self._check_video_authenticity(uploaded_file)
            else:
                raise ValueError(f"Unsupported file type: {file_extension}")
            
            # Validate extracted content
            if not content or len(content.strip()) < 5:
                raise ValueError("No meaningful content extracted from file")
            
            # Truncate if necessary
            if len(content) > self.max_content_length:
                content = content[:self.max_content_length] + "... [truncated]"
                st.info(f"📄 Content truncated to {self.max_content_length} characters")
            
            if include_authenticity_check and authenticity_result:
                return content, authenticity_result
            
            return content
            
        except Exception as e:
            error_msg = f"File processing failed: {str(e)}"
            self.logger.error(error_msg)
            raise Exception(error_msg)
    
    def _process_pdf(self, uploaded_file) -> str:
        """Extract text from PDF with enhanced error handling."""
        try:
            reader = PdfReader(uploaded_file)
            text_content = ""
            
            max_pages = 50  # Limit pages to prevent excessive processing
            pages_to_process = min(len(reader.pages), max_pages)
            
            for page_num in range(pages_to_process):
                try:
                    page = reader.pages[page_num]
                    page_text = page.extract_text()
                    if page_text:
                        text_content += page_text + "\n"
                except Exception as e:
                    self.logger.warning(f"Failed to extract text from page {page_num}: {e}")
                    continue
            
            if not text_content.strip():
                raise ValueError("No text could be extracted from PDF")
            
            # Clean up PDF text
            text_content = re.sub(r'\s+', ' ', text_content)
            text_content = re.sub(r'\n\s*\n', '\n', text_content)
            
            if len(reader.pages) > max_pages:
                text_content += f"\n... [PDF truncated: showing first {max_pages} of {len(reader.pages)} pages]"
            
            return text_content.strip()
            
        except Exception as e:
            raise Exception(f"PDF processing failed: {str(e)}")
    
    def _process_text_file(self, uploaded_file) -> str:
        """Process text file with encoding detection."""
        try:
            # Try different encodings
            encodings = ['utf-8', 'utf-16', 'latin-1', 'cp1252']
            
            for encoding in encodings:
                try:
                    uploaded_file.seek(0)
                    content = uploaded_file.read().decode(encoding)
                    break
                except UnicodeDecodeError:
                    continue
            else:
                # If all encodings fail, use utf-8 with error handling
                uploaded_file.seek(0)
                content = uploaded_file.read().decode('utf-8', errors='ignore')
            
            # Basic text cleaning
            content = re.sub(r'\r\n', '\n', content)
            content = re.sub(r'\r', '\n', content)
            content = content.strip()
            
            return content
            
        except Exception as e:
            raise Exception(f"Text file processing failed: {str(e)}")
    
    def _process_image(self, uploaded_file) -> str:
        """Extract text from image using OCR with fallback."""
        try:
            # Load image
            image = Image.open(uploaded_file)
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Resize if too large (max 2000px on longest side)
            max_dimension = 2000
            if max(image.size) > max_dimension:
                ratio = max_dimension / max(image.size)
                new_size = tuple(int(dim * ratio) for dim in image.size)
                image = image.resize(new_size, Image.Resampling.LANCZOS)
            
            extracted_text = ""
            
            # Try EasyOCR first
            if self.ocr_reader and not self.use_tesseract_fallback:
                try:
                    # Convert PIL image to numpy array for EasyOCR
                    import numpy as np
                    image_array = np.array(image)
                    results = self.ocr_reader.readtext(image_array, detail=0)
                    extracted_text = " ".join(results)
                except Exception as e:
                    self.logger.warning(f"EasyOCR failed: {e}. Trying Tesseract.")
                    self.use_tesseract_fallback = True
            
            # Fallback to Tesseract
            if not extracted_text or self.use_tesseract_fallback:
                try:
                    # Configure Tesseract for better accuracy
                    custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,!?@#$%^&*()_+-=[]{}|;:,.<>?/ '
                    extracted_text = pytesseract.image_to_string(image, config=custom_config)
                except Exception as e:
                    self.logger.error(f"Tesseract OCR failed: {e}")
                    extracted_text = "[OCR Error: Could not extract text from image]"
            
            # Clean extracted text
            if extracted_text and extracted_text != "[OCR Error: Could not extract text from image]":
                extracted_text = re.sub(r'\s+', ' ', extracted_text).strip()
                if len(extracted_text) < 5:
                    extracted_text = "[OCR Warning: Minimal text detected in image]"
            
            return extracted_text
            
        except Exception as e:
            return f"[Image processing error: {str(e)}]"
    
    def _check_image_authenticity(self, uploaded_file) -> Dict[str, Any]:
        """Check image authenticity using deepfake detector."""
        try:
            uploaded_file.seek(0)
            image = Image.open(uploaded_file)
            return self.deepfake_detector.analyze_image_authenticity(image)
        except Exception as e:
            return {
                'authenticity_score': 0.5,
                'is_authentic': None,
                'confidence': 'Low',
                'error': str(e)
            }
    
    def _check_video_authenticity(self, uploaded_file) -> Dict[str, Any]:
        """Check video authenticity using deepfake detector."""
        try:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                uploaded_file.seek(0)
                tmp_file.write(uploaded_file.read())
                tmp_file.flush()
                
                result = self.deepfake_detector.analyze_video_authenticity(tmp_file.name)
                
                # Clean up
                os.unlink(tmp_file.name)
                
                return result
        except Exception as e:
            return {
                'authenticity_score': 0.5,
                'is_authentic': None,
                'confidence': 'Low',
                'error': str(e)
            }
    
    def get_content_summary(self, content: str) -> Dict[str, Any]:
        """Generate content summary and metadata."""
        word_count = len(content.split())
        char_count = len(content)
        
        # Basic content analysis
        has_urls = bool(re.search(r'http[s]?://\S+', content))
        has_emails = bool(re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', content))
        has_phone_numbers = bool(re.search(r'\b\d{10,}\b', content))
        has_financial_terms = bool(re.search(r'\b(?:investment|profit|return|stock|share|money|₹|rupee|lakh|crore)\b', content, re.IGNORECASE))
        
        return {
            'word_count': word_count,
            'character_count': char_count,
            'estimated_read_time': max(1, word_count // 200),  # ~200 WPM
            'contains_urls': has_urls,
            'contains_emails': has_emails,
            'contains_phone_numbers': has_phone_numbers,
            'contains_financial_terms': has_financial_terms,
            'content_hash': hashlib.md5(content.encode()).hexdigest()[:8]
        }
    
    def __del__(self):
        """Cleanup resources."""
        if hasattr(self, 'session'):
            self.session.close()
