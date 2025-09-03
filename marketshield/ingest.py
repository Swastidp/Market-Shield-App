import requests
import easyocr
import pytesseract
from pypdf import PdfReader
from youtube_transcript_api import YouTubeTranscriptApi
import re
from typing import Optional, Dict, Any
from PIL import Image
import io
import streamlit as st

class ContentIngester:
    """Handles ingestion from multiple content sources."""
    
    def __init__(self):
        self.ocr_reader = easyocr.Reader(['en'])
    
    def fetch_url_content(self, url: str) -> str:
        """Fetch and extract text content from URL."""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            # Simple text extraction (in production, use BeautifulSoup for better parsing)
            content = response.text
            
            # Basic HTML tag removal
            content = re.sub(r'<[^>]+>', ' ', content)
            content = re.sub(r'\s+', ' ', content).strip()
            
            return content[:5000]  # Limit content length
            
        except Exception as e:
            raise Exception(f"Failed to fetch URL content: {str(e)}")
    
    def process_file(self, uploaded_file) -> str:
        """Process uploaded file and extract text content."""
        file_type = uploaded_file.type
        
        try:
            if file_type == "application/pdf":
                return self._extract_pdf_text(uploaded_file)
            elif file_type == "text/plain":
                return str(uploaded_file.read(), "utf-8")
            elif file_type.startswith("image/"):
                return self._extract_image_text(uploaded_file)
            else:
                raise Exception(f"Unsupported file type: {file_type}")
                
        except Exception as e:
            raise Exception(f"Failed to process file: {str(e)}")
    
    def _extract_pdf_text(self, pdf_file) -> str:
        """Extract text from PDF file."""
        try:
            reader = PdfReader(pdf_file)
            text = ""
            for page in reader.pages[:10]:  # Limit to first 10 pages
                text += page.extract_text() + "\n"
            return text.strip()
        except Exception as e:
            raise Exception(f"PDF extraction failed: {str(e)}")
    
    def _extract_image_text(self, image_file) -> str:
        """Extract text from image using OCR."""
        try:
            image = Image.open(image_file)
            # Use EasyOCR for better accuracy
            result = self.ocr_reader.readtext(image)
            text = " ".join([item[1] for item in result])
            return text
        except Exception as e:
            # Fallback to pytesseract
            try:
                image = Image.open(image_file)
                text = pytesseract.image_to_string(image)
                return text
            except:
                raise Exception(f"OCR extraction failed: {str(e)}")
    
    def extract_youtube_transcript(self, url: str) -> str:
        """Extract transcript from YouTube video."""
        try:
            # Extract video ID from URL
            video_id_match = re.search(r'(?:v=|\/)([0-9A-Za-z_-]{11}).*', url)
            if not video_id_match:
                raise Exception("Invalid YouTube URL")
            
            video_id = video_id_match.group(1)
            
            # Get transcript
            transcript = YouTubeTranscriptApi.get_transcript(video_id)
            
            # Combine all transcript text
            full_text = " ".join([item['text'] for item in transcript])
            return full_text
            
        except Exception as e:
            raise Exception(f"YouTube transcript extraction failed: {str(e)}")
