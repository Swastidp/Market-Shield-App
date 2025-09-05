import google.generativeai as genai
import numpy as np
import faiss
import pickle
import os
import json
import time
from typing import List, Dict, Any, Optional
import streamlit as st
import hashlib
import re
import logging

class RAGEngine:
    """RAG-powered context retrieval with content chunking for large payloads."""

    def __init__(self):
        """Initialize RAG engine with chunking support."""
        self.api_key = self._get_api_key()
        self.logger = logging.getLogger(__name__)
        
        if self.api_key:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel('gemini-1.5-flash')
            self.embedding_model = 'models/text-embedding-004'
        
        self.index = None
        self.documents = []
        
        # Chunking configuration
        self.max_chunk_size = 8000      # Safe size in characters (~2000 tokens)
        self.chunk_overlap = 200        # Overlap between chunks
        self.max_payload_size = 35000   # Safe API payload threshold (bytes)
        
        # Performance tracking
        self.cache_stats = {'hits': 0, 'misses': 0, 'chunks_created': 0}
        
        self._load_index()

    def _get_api_key(self) -> Optional[str]:
        """Get Gemini API key from environment or Streamlit secrets."""
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            try:
                api_key = st.secrets['GEMINI_API_KEY']
            except:
                st.warning("⚠️ Gemini API key not found. RAG features disabled.")
        return api_key

    def _estimate_payload_size(self, content: str) -> int:
        """Estimate API payload size in bytes including JSON overhead."""
        # Simulate the actual API payload structure
        base_payload = {
            "model": self.embedding_model,
            "content": {"parts": [{"text": content}]},
            "task_type": "retrieval_document"
        }
        # Convert to JSON and measure byte size
        payload_json = json.dumps(base_payload, ensure_ascii=False)
        return len(payload_json.encode('utf-8'))

    def _chunk_content(self, content: str) -> List[str]:
        """Split content into chunks that fit within API limits."""
        # Quick check: if content is small enough, return as-is
        if self._estimate_payload_size(content) <= self.max_payload_size:
            return [content]
        
        self.logger.info(f"Content too large ({self._estimate_payload_size(content)} bytes), chunking...")
        chunks = []
        
        # Split by sentences first for better semantic chunking
        sentences = re.split(r'(?<=[.!?])\s+', content)
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            # Test adding this sentence to current chunk
            test_chunk = ' '.join(current_chunk + [sentence])
            test_size = self._estimate_payload_size(test_chunk)
            
            if test_size > self.max_payload_size and current_chunk:
                # Current chunk is full, save it and start new one
                chunk_text = ' '.join(current_chunk)
                chunks.append(chunk_text)
                
                # Start new chunk with overlap from previous chunk
                overlap_sentences = current_chunk[-2:] if len(current_chunk) > 2 else current_chunk
                current_chunk = overlap_sentences + [sentence]
                current_size = self._estimate_payload_size(' '.join(current_chunk))
            else:
                # Add sentence to current chunk
                current_chunk.append(sentence)
                current_size = test_size
        
        # Add final chunk if it exists
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        # If chunking still produces large chunks, fall back to word-level chunking
        final_chunks = []
        for chunk in chunks:
            if self._estimate_payload_size(chunk) > self.max_payload_size:
                final_chunks.extend(self._chunk_by_words(chunk))
            else:
                final_chunks.append(chunk)
        
        self.cache_stats['chunks_created'] += len(final_chunks)
        self.logger.info(f"Created {len(final_chunks)} chunks from content")
        
        return final_chunks

    def _chunk_by_words(self, content: str) -> List[str]:
        """Fallback word-level chunking for very dense content."""
        words = content.split()
        chunks = []
        current_chunk = []
        
        for word in words:
            test_chunk = ' '.join(current_chunk + [word])
            test_size = self._estimate_payload_size(test_chunk)
            
            if test_size > self.max_payload_size and current_chunk:
                chunks.append(' '.join(current_chunk))
                # Start new chunk with small overlap
                overlap_words = current_chunk[-10:] if len(current_chunk) > 10 else current_chunk
                current_chunk = overlap_words + [word]
            else:
                current_chunk.append(word)
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks

    def _generate_embeddings_with_retry(self, content_chunks: List[str]) -> List[List[float]]:
        """Generate embeddings for content chunks with comprehensive error handling."""
        embeddings = []
        total_chunks = len(content_chunks)
        
        # Show progress for large number of chunks
        if total_chunks > 5:
            progress_bar = st.progress(0)
            status_text = st.empty()
        
        for i, chunk in enumerate(content_chunks):
            max_retries = 3
            
            # Update progress
            if total_chunks > 5:
                progress = (i + 1) / total_chunks
                progress_bar.progress(progress)
                status_text.text(f"Processing chunk {i+1} of {total_chunks}...")
            
            for attempt in range(max_retries):
                try:
                    # Final size check before API call
                    estimated_size = self._estimate_payload_size(chunk)
                    if estimated_size > self.max_payload_size:
                        st.warning(f"⚠️ Chunk {i+1} still too large ({estimated_size} bytes), truncating...")
                        # Aggressive truncation
                        chunk = chunk[:int(len(chunk) * 0.7)]
                    
                    # Make API call
                    response = genai.embed_content(
                        model=self.embedding_model,
                        content=chunk,
                        task_type="retrieval_document"
                    )
                    
                    embeddings.append(response['embedding'])
                    self.logger.debug(f"Successfully embedded chunk {i+1}")
                    break
                    
                except Exception as e:
                    error_msg = str(e).lower()
                    
                    if "payload size exceeds" in error_msg:
                        st.warning(f"⚠️ Chunk {i+1} too large, reducing size...")
                        # Reduce chunk size by 30%
                        chunk = chunk[:int(len(chunk) * 0.7)]
                        if attempt == max_retries - 1:
                            st.error(f"❌ Failed to embed chunk {i+1} after size reduction")
                            # Use zero embedding as fallback
                            embeddings.append([0.0] * 768)  # text-embedding-004 dimension
                    
                    elif "quota" in error_msg or "rate" in error_msg or "429" in error_msg:
                        wait_time = 2 ** attempt
                        st.warning(f"⚠️ Rate limit hit, waiting {wait_time}s before retry...")
                        time.sleep(wait_time)
                        if attempt == max_retries - 1:
                            st.error(f"❌ Rate limit exceeded for chunk {i+1}")
                            embeddings.append([0.0] * 768)
                    
                    elif "401" in error_msg or "api key" in error_msg:
                        st.error(f"❌ API key error: {e}")
                        embeddings.append([0.0] * 768)
                        break
                    
                    else:
                        st.error(f"❌ Embedding failed for chunk {i+1}: {e}")
                        embeddings.append([0.0] * 768)
                        break
        
        # Clean up progress indicators
        if total_chunks > 5:
            progress_bar.empty()
            status_text.empty()
        
        return embeddings

    def get_relevant_context(self, query: str, top_k: int = 3) -> Dict[str, Any]:
        """Retrieve relevant context with chunking support."""
        if not self.api_key or not self.index:
            return self._get_fallback_context()

        try:
            # Handle large query by chunking if necessary
            query_chunks = self._chunk_content(query)
            
            if len(query_chunks) > 1:
                st.info(f"📝 Query split into {len(query_chunks)} chunks for processing")
            
            # Use first chunk as primary query (most important content usually at start)
            primary_query = query_chunks[0]
            
            # Generate query embedding with retry logic
            query_embedding = None
            max_retries = 3
            
            for attempt in range(max_retries):
                try:
                    response = genai.embed_content(
                        model=self.embedding_model,
                        content=primary_query,
                        task_type="retrieval_query"
                    )
                    query_embedding = np.array([response['embedding']]).astype('float32')
                    break
                    
                except Exception as e:
                    if "payload size exceeds" in str(e).lower():
                        st.warning("⚠️ Query too large, truncating...")
                        # Aggressive truncation for query
                        primary_query = primary_query[:4000]
                        if attempt == max_retries - 1:
                            raise Exception("Query too large even after truncation")
                    elif "rate" in str(e).lower() or "quota" in str(e).lower():
                        wait_time = 2 ** attempt
                        st.warning(f"⚠️ Rate limit hit, waiting {wait_time}s...")
                        time.sleep(wait_time)
                        if attempt == max_retries - 1:
                            raise e
                    else:
                        raise e

            if query_embedding is None:
                raise Exception("Failed to generate query embedding")

            # Normalize embedding for similarity search
            faiss.normalize_L2(query_embedding)

            # Search for similar documents
            scores, indices = self.index.search(query_embedding, top_k * 2)  # Get more results to filter

            # Format results with relevance filtering
            citations = []
            seen_content = set()  # Avoid duplicate content
            
            for i, idx in enumerate(indices[0]):
                if len(citations) >= top_k:
                    break
                    
                if idx < len(self.documents) and scores[0][i] > 0.15:  # Minimum relevance threshold
                    doc = self.documents[idx]
                    
                    # Avoid duplicate content (from chunked documents)
                    content_hash = hashlib.md5(doc['content'].encode()).hexdigest()[:8]
                    if content_hash not in seen_content:
                        citations.append({
                            'title': doc['title'],
                            'content': doc['content'],
                            'score': float(scores[0][i]),
                            'id': doc['id']
                        })
                        seen_content.add(content_hash)

            self.cache_stats['hits'] += 1
            
            return {
                'citations': citations,
                'query': query,
                'chunks_processed': len(query_chunks),
                'embedding_method': 'chunked' if len(query_chunks) > 1 else 'direct',
                'relevance_threshold': 0.15,
                'total_results_found': len([s for s in scores[0] if s > 0.15])
            }

        except Exception as e:
            self.cache_stats['misses'] += 1
            st.error(f"❌ Vector search failed: {e}, falling back to keyword search")
            return self._get_fallback_context()

    def _create_demo_index(self):
        """Create a demo index with sample SEBI policy content and chunking support."""
        sample_docs = [
            {
                'title': 'SEBI Investment Adviser Regulations 2013',
                'content': '''Investment advisers are required to be registered with SEBI under these regulations.
                No person shall carry on the activity of investment adviser without obtaining a certificate of registration from SEBI.
                Investment advisers are prohibited from guaranteeing returns on investments or making misleading claims about performance.
                All investment advice must be provided in the best interest of the client with proper disclosure of conflicts of interest.
                The regulations also specify fee structures, compliance requirements, and code of conduct for registered investment advisers.''',
                'id': 'sebi_ia_2013'
            },
            {
                'title': 'SEBI Social Media Guidelines 2021',
                'content': '''SEBI-regulated intermediaries using social media platforms must ensure compliance with existing regulations.
                All promotional content must include proper disclaimers and risk warnings as specified by SEBI guidelines.
                Guaranteed return claims, misleading advertisements, and unsubstantiated performance claims are strictly prohibited.
                Intermediaries must maintain records of all social media communications and ensure they meet disclosure requirements.
                The guidelines cover platforms like Twitter, Facebook, Instagram, LinkedIn, and messaging apps like WhatsApp and Telegram.''',
                'id': 'sebi_social_2021'
            },
            {
                'title': 'SEBI UPI Validation Framework 2025',
                'content': '''Effective October 1, 2025, all UPI handles used by SEBI-registered entities for collecting payments must be validated through the SEBI Check system.
                This measure is part of the Safe Space initiative to protect investors from fraudulent payment collection.
                Unverified UPI handles used by entities claiming SEBI registration will be flagged for investigation.
                Investors are advised to verify UPI handles through the official SEBI portal before making any payments.
                The framework includes real-time validation APIs and integration with major UPI service providers.''',
                'id': 'sebi_upi_2025'
            },
            {
                'title': 'SEBI Research Analyst Regulations',
                'content': '''Research analysts must be registered with SEBI and follow prescribed standards for research reports.
                All research reports must include disclaimers about the analyst\'s position in the stock and potential conflicts of interest.
                Research analysts are prohibited from making buy/sell recommendations through unofficial channels like WhatsApp or Telegram groups.
                Investors should verify the SEBI registration of research analysts before acting on their recommendations.
                The regulations specify minimum qualifications, experience requirements, and continuing education for research analysts.''',
                'id': 'sebi_research_analysts'
            },
            {
                'title': 'SEBI Investor Protection Measures',
                'content': '''SEBI has established various investor protection measures including the Investor Protection and Education Fund.
                Investors who suffer losses due to fraudulent activities by unregistered intermediaries may be eligible for compensation.
                The key to investor protection is verification of credentials before engaging with any investment adviser or research analyst.
                SEBI regularly updates its list of defaulters and unregistered entities on its official website.
                The measures include grievance redressal mechanisms, investor education programs, and market surveillance systems.''',
                'id': 'sebi_investor_protection'
            }
        ]

        if not self.api_key:
            self.documents = sample_docs
            return

        try:
            st.info("🔄 Building RAG index with chunking support...")
            
            # Process documents with chunking
            all_embeddings = []
            processed_docs = []
            
            for doc_idx, doc in enumerate(sample_docs):
                st.text(f"Processing document {doc_idx + 1} of {len(sample_docs)}: {doc['title'][:50]}...")
                
                # Chunk the document content
                chunks = self._chunk_content(doc['content'])
                
                if len(chunks) > 1:
                    st.info(f"📄 Document split into {len(chunks)} chunks")
                
                # Generate embeddings for all chunks
                chunk_embeddings = self._generate_embeddings_with_retry(chunks)
                
                # Create document entries for each chunk
                for i, (chunk, embedding) in enumerate(zip(chunks, chunk_embeddings)):
                    chunk_doc = {
                        'title': f"{doc['title']} (Part {i+1})" if len(chunks) > 1 else doc['title'],
                        'content': chunk,
                        'id': f"{doc['id']}_chunk_{i}" if len(chunks) > 1 else doc['id'],
                        'parent_id': doc['id'],
                        'chunk_index': i,
                        'total_chunks': len(chunks)
                    }
                    processed_docs.append(chunk_doc)
                    all_embeddings.append(embedding)

            if not all_embeddings:
                st.error("❌ No embeddings generated")
                return

            # Create FAISS index
            embedding_dim = len(all_embeddings[0])
            self.index = faiss.IndexFlatIP(embedding_dim)
            
            # Convert to numpy array and normalize
            embeddings_array = np.array(all_embeddings).astype('float32')
            faiss.normalize_L2(embeddings_array)
            self.index.add(embeddings_array)
            
            self.documents = processed_docs

            st.success(f"✅ Created RAG index with {len(processed_docs)} document chunks from {len(sample_docs)} source documents")

        except Exception as e:
            st.error(f"❌ Failed to create demo index: {e}")
            self.documents = sample_docs

    def _load_index(self):
        """Load pre-built FAISS index and documents with chunking support."""
        try:
            if os.path.exists('data/faiss_index.bin') and os.path.exists('data/documents.pkl'):
                self.index = faiss.read_index('data/faiss_index.bin')
                with open('data/documents.pkl', 'rb') as f:
                    self.documents = pickle.load(f)
                st.info(f"📚 Loaded {len(self.documents)} documents from cache")
            else:
                self._create_demo_index()
        except Exception as e:
            st.error(f"❌ Failed to load RAG index: {e}")
            self._create_demo_index()

    def _get_fallback_context(self) -> Dict[str, Any]:
        """Provide fallback context when RAG is unavailable."""
        return {
            'citations': [
                {
                    'title': 'SEBI Investment Adviser Guidelines',
                    'content': 'Only SEBI-registered investment advisers can provide investment advice. Guaranteed return claims are prohibited.',
                    'score': 0.95,
                    'id': 'fallback_1'
                },
                {
                    'title': 'SEBI Investor Protection Measures',
                    'content': 'Investors should verify credentials before acting on investment advice. Be cautious of unsolicited tips and guaranteed profit claims.',
                    'score': 0.90,
                    'id': 'fallback_2'
                }
            ],
            'query': 'fallback',
            'embedding_method': 'fallback'
        }

    def get_cache_statistics(self) -> Dict[str, Any]:
        """Get performance statistics for the RAG system."""
        total_requests = self.cache_stats['hits'] + self.cache_stats['misses']
        hit_rate = self.cache_stats['hits'] / total_requests if total_requests > 0 else 0
        
        return {
            'cache_hits': self.cache_stats['hits'],
            'cache_misses': self.cache_stats['misses'],
            'hit_rate': hit_rate,
            'total_chunks_created': self.cache_stats['chunks_created'],
            'documents_loaded': len(self.documents),
            'index_available': self.index is not None
        }

    def clear_cache(self):
        """Clear the cache and reset statistics."""
        self.cache_stats = {'hits': 0, 'misses': 0, 'chunks_created': 0}
        st.success("✅ RAG cache cleared successfully")

    def save_index(self, save_path: str = 'data/'):
        """Save the current index and documents to disk."""
        try:
            os.makedirs(save_path, exist_ok=True)
            
            if self.index:
                faiss.write_index(self.index, os.path.join(save_path, 'faiss_index.bin'))
            
            if self.documents:
                with open(os.path.join(save_path, 'documents.pkl'), 'wb') as f:
                    pickle.dump(self.documents, f)
            
            st.success(f"✅ RAG index saved to {save_path}")
            
        except Exception as e:
            st.error(f"❌ Failed to save index: {e}")
