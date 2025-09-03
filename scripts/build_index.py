import os
# Set API key directly in code for MVP
os.environ['GEMINI_API_KEY'] = 'AIzaSyD4274nxTy2HR_z6W4KsyQ5_OuKTJaJU68'



#!/usr/bin/env python3
"""
Build FAISS index for RAG-powered fraud detection.
Run this script to initialize the knowledge base.
"""

import os
import sys
import pickle
import numpy as np
import faiss
import google.generativeai as genai
from pathlib import Path

def create_reference_documents():
    """Create sample SEBI reference documents."""
    documents = [
        {
            'title': 'SEBI (Investment Advisers) Regulations, 2013',
            'content': '''Investment advisers are required to be registered with SEBI under these regulations. 
            No person shall carry on the activity of investment adviser without obtaining a certificate of registration from SEBI. 
            Investment advisers are prohibited from guaranteeing returns on investments or making misleading claims about performance. 
            All investment advice must be provided in the best interest of the client with proper disclosure of conflicts of interest.''',
            'id': 'sebi_ia_2013',
            'category': 'regulation'
        },
        {
            'title': 'SEBI Guidelines on Social Media Usage by Intermediaries',
            'content': '''SEBI-regulated intermediaries using social media platforms must ensure compliance with existing regulations. 
            All promotional content must include proper disclaimers and risk warnings. 
            Guaranteed return claims, misleading advertisements, and unsubstantiated performance claims are strictly prohibited. 
            Intermediaries must maintain records of all social media communications and ensure they meet disclosure requirements.''',
            'id': 'sebi_social_media_2021',
            'category': 'guidelines'
        },
        {
            'title': 'SEBI UPI Validation Framework 2025',
            'content': '''Effective October 1, 2025, all UPI handles used by SEBI-registered entities for collecting payments must be validated through the SEBI Check system. 
            This measure is part of the Safe Space initiative to protect investors from fraudulent payment collection. 
            Unverified UPI handles used by entities claiming SEBI registration will be flagged for investigation. 
            Investors are advised to verify UPI handles through the official SEBI portal before making any payments.''',
            'id': 'sebi_upi_2025',
            'category': 'framework'
        },
        {
            'title': 'SEBI Guidelines on Research Analysts',
            'content': '''Research analysts must be registered with SEBI and follow prescribed standards for research reports. 
            All research reports must include disclaimers about the analyst\'s position in the stock and potential conflicts of interest. 
            Research analysts are prohibited from making buy/sell recommendations through unofficial channels like WhatsApp or Telegram groups. 
            Investors should verify the SEBI registration of research analysts before acting on their recommendations.''',
            'id': 'sebi_research_analysts',
            'category': 'regulation'
        },
        {
            'title': 'SEBI Investor Protection and Education Fund',
            'content': '''SEBI has established various investor protection measures including the Investor Protection and Education Fund. 
            Investors who suffer losses due to fraudulent activities by unregistered intermediaries may be eligible for compensation. 
            The key to investor protection is verification of credentials before engaging with any investment adviser or research analyst. 
            SEBI regularly updates its list of defaulters and unregistered entities on its official website.''',
            'id': 'sebi_investor_protection',
            'category': 'protection'
        },
        {
            'title': 'SEBI Corporate Disclosure Requirements',
            'content': '''Listed companies must make disclosures to stock exchanges in a timely manner for all material events. 
            Corporate announcements must be authentic and verified before dissemination. 
            Fabricated or misleading corporate announcements can significantly impact stock prices and constitute market manipulation. 
            Investors should verify corporate announcements through official exchange websites before making investment decisions.''',
            'id': 'sebi_corporate_disclosure',
            'category': 'disclosure'
        }
    ]
    
    return documents

def build_faiss_index(documents, api_key):
    """Build FAISS index from documents using Gemini embeddings."""
    print("Configuring Gemini API...")
    genai.configure(api_key=api_key)
    
    print("Generating embeddings...")
    embeddings = []
    
    for i, doc in enumerate(documents):
        print(f"Processing document {i+1}/{len(documents)}: {doc['title']}")
        
        try:
            response = genai.embed_content(
                model='models/text-embedding-004',
                content=doc['content']
            )
            embeddings.append(response['embedding'])
        except Exception as e:
            print(f"Error embedding document {doc['id']}: {e}")
            continue
    
    if not embeddings:
        raise Exception("No embeddings generated successfully")
    
    print("Building FAISS index...")
    embedding_dim = len(embeddings[0])
    index = faiss.IndexFlatIP(embedding_dim)
    
    embeddings_array = np.array(embeddings).astype('float32')
    faiss.normalize_L2(embeddings_array)
    index.add(embeddings_array)
    
    return index, documents

def main():
    """Main function to build the index."""
    # Check for API key
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("Error: GEMINI_API_KEY environment variable not set")
        print("Please set your Gemini API key:")
        print("export GEMINI_API_KEY='your-api-key-here'")
        sys.exit(1)
    
    # Create data directory
    data_dir = Path('data')
    data_dir.mkdir(exist_ok=True)
    
    print("Creating reference documents...")
    documents = create_reference_documents()
    
    try:
        print("Building FAISS index...")
        index, processed_docs = build_faiss_index(documents, api_key)
        
        print("Saving index and documents...")
        faiss.write_index(index, str(data_dir / 'faiss_index.bin'))
        
        with open(data_dir / 'documents.pkl', 'wb') as f:
            pickle.dump(processed_docs, f)
        
        print(f"✅ Successfully built index with {len(processed_docs)} documents")
        print(f"Index saved to: {data_dir / 'faiss_index.bin'}")
        print(f"Documents saved to: {data_dir / 'documents.pkl'}")
        
    except Exception as e:
        print(f"❌ Error building index: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
