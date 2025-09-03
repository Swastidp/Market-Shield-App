
import google.generativeai as genai
import numpy as np
import faiss
import pickle
import os
from typing import List, Dict, Any, Optional
import streamlit as st

class RAGEngine:
    """RAG-powered context retrieval using SEBI policy documents."""
    
    def __init__(self):
        self.api_key = self._get_api_key()
        if self.api_key:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel('gemini-1.5-flash')
            self.embedding_model = 'models/text-embedding-004'
        
        self.index = None
        self.documents = []
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
    
    def _load_index(self):
        """Load pre-built FAISS index and documents."""
        try:
            if os.path.exists('data/faiss_index.bin') and os.path.exists('data/documents.pkl'):
                self.index = faiss.read_index('data/faiss_index.bin')
                with open('data/documents.pkl', 'rb') as f:
                    self.documents = pickle.load(f)
            else:
                # Create dummy index for demo
                self._create_demo_index()
        except Exception as e:
            st.error(f"Failed to load RAG index: {e}")
    
    def _create_demo_index(self):
        """Create a demo index with sample SEBI policy content."""
        sample_docs = [
            {
                'title': 'SEBI Investment Adviser Regulations 2013',
                'content': 'Investment advisers must be registered with SEBI. Unregistered advisers cannot provide investment advice for consideration. Guaranteed returns claims are prohibited.',
                'id': 'sebi_ia_2013'
            },
            {
                'title': 'SEBI Social Media Guidelines 2021',
                'content': 'Intermediaries using social media must ensure compliance with disclosure norms. Misleading advertisements and guaranteed return claims are strictly prohibited.',
                'id': 'sebi_social_2021'
            },
            {
                'title': 'SEBI UPI Validation Framework 2025',
                'content': 'From October 1, 2025, all UPI handles used by SEBI-registered entities must be validated through SEBI Check system for investor protection.',
                'id': 'sebi_upi_2025'
            }
        ]
        
        if not self.api_key:
            self.documents = sample_docs
            return
        
        try:
            # Generate embeddings for sample documents
            embeddings = []
            for doc in sample_docs:
                response = genai.embed_content(
                    model=self.embedding_model,
                    content=doc['content']
                )
                embeddings.append(response['embedding'])
            
            # Create FAISS index
            embedding_dim = len(embeddings[0])
            self.index = faiss.IndexFlatIP(embedding_dim)
            
            embeddings_array = np.array(embeddings).astype('float32')
            faiss.normalize_L2(embeddings_array)
            self.index.add(embeddings_array)
            
            self.documents = sample_docs
            
        except Exception as e:
            st.error(f"Failed to create demo index: {e}")
    
    def get_relevant_context(self, query: str, top_k: int = 3) -> Dict[str, Any]:
        """Retrieve relevant context for the given query."""
        if not self.api_key or not self.index:
            return self._get_fallback_context()
        
        try:
            # Generate query embedding
            response = genai.embed_content(
                model=self.embedding_model,
                content=query
            )
            
            query_embedding = np.array([response['embedding']]).astype('float32')
            faiss.normalize_L2(query_embedding)
            
            # Search for similar documents
            scores, indices = self.index.search(query_embedding, top_k)
            
            # Format results
            citations = []
            for i, idx in enumerate(indices[0]):
                if idx < len(self.documents):
                    doc = self.documents[idx]
                    citations.append({
                        'title': doc['title'],
                        'content': doc['content'],
                        'score': float(scores[0][i]),
                        'id': doc['id']
                    })
            
            return {
                'citations': citations,
                'query': query
            }
            
        except Exception as e:
            st.error(f"RAG context retrieval failed: {e}")
            return self._get_fallback_context()
    
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
            'query': 'fallback'
        }
