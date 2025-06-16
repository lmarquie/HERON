# Import necessary libraries
import os
import faiss
import fitz  # PyMuPDF
import pickle
import torch
import time
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import requests
import io
import numpy as np
from config import OPENAI_API_KEY
import openai
import streamlit as st

### =================== Vector Store =================== ###
class VectorStore:
    def __init__(self, dimension: int = 1536):  # OpenAI's embedding dimension
        self.dimension = dimension
        self.index = None
        self.documents = []
        self.embedding_cache = {}
        self.similarity_cache = {}
        self.metadata_index = {}
        self.chunk_size = 500
        self.chunk_overlap = 50
        print(f"Initialized VectorStore with OpenAI embeddings")

    def get_embedding(self, text: str):
        """Get embedding using OpenAI's API"""
        if text in self.embedding_cache:
            return self.embedding_cache[text]
        
        try:
            response = openai.Embedding.create(
                input=text,
                model="text-embedding-ada-002"
            )
            embedding = np.array(response['data'][0]['embedding'])
            self.embedding_cache[text] = embedding
            return embedding
        except Exception as e:
            print(f"Error getting embedding: {str(e)}")
            return np.zeros(self.dimension)

    def add_documents(self, documents: List[Dict]):
        print(f"Adding {len(documents)} documents to vector store...")
        texts = []
        metadata_list = []
        for doc in documents:
            metadata = {
                'source': doc['metadata'].get('source', 'Unknown'),
                'path': doc['metadata'].get('path', ''),
                'date': doc['metadata'].get('date', datetime.now().isoformat())
            }
            texts.append(doc['text'])
            metadata_list.append(metadata)

        print("Converting text to vector embeddings...")
        embeddings = []
        for text in texts:
            embedding = self.get_embedding(text)
            embeddings.append(embedding)

        embeddings = np.array(embeddings)

        if self.index is None:
            self.index = faiss.IndexFlatL2(embeddings.shape[1])
            print(f"Created new FAISS index with dimension: {embeddings.shape[1]}")
        
        self.index.add(embeddings.astype('float32'))

        for i, (text, metadata) in enumerate(zip(texts, metadata_list)):
            self.documents.append({
                'text': text,
                'metadata': metadata
            })
            for key, value in metadata.items():
                if key not in self.metadata_index:
                    self.metadata_index[key] = {}
                if value not in self.metadata_index[key]:
                    self.metadata_index[key][value] = set()
                self.metadata_index[key][value].add(i)

        print(f"Successfully added {len(documents)} documents. Total documents: {len(self.documents)}")

    def search(self, query: str, k: int = 3) -> List[Dict]:
        """Search for relevant documents"""
        cache_key = f"{query}:{k}"
        if cache_key in self.similarity_cache:
            return self.similarity_cache[cache_key]

        query_embedding = self.get_embedding(query)
        
        if self.index is not None:
            initial_k = min(k * 2, len(self.documents))
            distances, indices = self.index.search(query_embedding.reshape(1, -1).astype('float32'), initial_k)
            
            results = []
            for distance, idx in zip(distances[0], indices[0]):
                if idx != -1:
                    doc = self.documents[idx]
                    results.append({
                        'text': doc['text'],
                        'metadata': doc['metadata'],
                        'score': float(1 / (1 + distance))  # Convert distance to similarity score
                    })
            
            results.sort(key=lambda x: x['score'], reverse=True)
            results = results[:k]
            
            self.similarity_cache[cache_key] = results
            return results
        
        return []

### =================== RAG System =================== ###
class RAGSystem:
    def __init__(self, settings: Optional[Dict] = None):
        self.vector_store = VectorStore()
        self.settings = settings or {
            'model': 'gpt-4',
            'temperature': 0.3,
            'chunk_size': 500,
            'chunk_overlap': 50
        }
        self.question_handler = None
        self.document_processor = None
        self.answer_generator = None
        self.sources = []
        print("Initialized RAG System")

    def process_document(self, document_text: bytes):
        """Process a document and add it to the vector store"""
        try:
            # Convert bytes to text
            if isinstance(document_text, bytes):
                document_text = document_text.decode('utf-8')
            
            # Split into chunks
            chunks = self._split_text(document_text)
            
            # Add to vector store
            documents = [{
                'text': chunk,
                'metadata': {
                    'source': 'Uploaded Document',
                    'chunk_index': i
                }
            } for i, chunk in enumerate(chunks)]
            
            self.vector_store.add_documents(documents)
            print("Document processed successfully")
            
        except Exception as e:
            print(f"Error processing document: {str(e)}")
            raise

    def _split_text(self, text: str) -> List[str]:
        """Split text into chunks"""
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            end = start + self.settings['chunk_size']
            if end > text_length:
                end = text_length
            else:
                # Try to find a good breaking point
                last_period = text.rfind('.', start, end)
                if last_period != -1 and last_period > start + self.settings['chunk_size'] // 2:
                    end = last_period + 1
            
            chunks.append(text[start:end].strip())
            start = end - self.settings['chunk_overlap']
        
        return chunks

    def get_answer(self, question: str) -> str:
        """Get answer for a question"""
        try:
            # Search for relevant documents
            results = self.vector_store.search(question)
            
            if not results:
                return "I couldn't find any relevant information to answer your question."
            
            # Prepare context from results
            context = "\n\n".join([r['text'] for r in results])
            
            # Store sources
            self.sources = [r['metadata']['source'] for r in results]
            
            # Generate answer using OpenAI
            response = openai.ChatCompletion.create(
                model=self.settings['model'],
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided context. If the answer cannot be found in the context, say so."},
                    {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
                ],
                temperature=self.settings['temperature']
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"Error getting answer: {str(e)}")
            return f"Error generating answer: {str(e)}"

    def get_sources(self) -> List[str]:
        """Get sources used for the last answer"""
        return self.sources

    def apply_settings(self, settings: Dict):
        """Apply new settings"""
        self.settings.update(settings)
        print(f"Applied new settings: {settings}")

### =================== Streamlit Interface =================== ###
def main():
    # Set page config
    st.set_page_config(
        page_title="Document Q&A System",
        page_icon="📚",
        layout="wide"
    )
    
    # Initialize session state
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = None
        st.session_state.document_processed = False
    
    # Title and description
    st.title("📚 Document Q&A System")
    st.markdown("""
    Upload a PDF document and ask questions about its content. The system will use AI to find relevant information and provide answers.
    """)
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload a PDF document",
        type=['pdf'],
        help="Upload a PDF file to analyze"
    )
    
    # Process document if uploaded
    if uploaded_file is not None and not st.session_state.document_processed:
        with st.spinner("Processing document..."):
            # Initialize RAG system
            st.session_state.rag_system = RAGSystem()
            
            # Process document
            document_text = uploaded_file.read()
            st.session_state.rag_system.process_document(document_text)
            st.session_state.document_processed = True
            
            st.success("Document processed successfully!")
    
    # Question input
    question = st.text_input(
        "Ask a question about the document",
        placeholder="Type your question here...",
        help="Enter your question about the uploaded document"
    )
    
    # Advanced settings in expander
    with st.expander("Advanced Settings"):
        col1, col2 = st.columns(2)
        
        with col1:
            # Model selection
            model = st.selectbox(
                "Select Model",
                options=["gpt-4", "gpt-3.5-turbo"],
                index=0,
                help="Choose the OpenAI model to use"
            )
            
            # Temperature slider
            temperature = st.slider(
                "Temperature",
                min_value=0.0,
                max_value=1.0,
                value=0.3,
                step=0.1,
                help="Higher values make the output more creative but less focused"
            )
        
        with col2:
            # Chunk size input
            chunk_size = st.number_input(
                "Chunk Size",
                min_value=100,
                max_value=1000,
                value=500,
                step=50,
                help="Size of text chunks for processing"
            )
            
            # Chunk overlap input
            chunk_overlap = st.number_input(
                "Chunk Overlap",
                min_value=0,
                max_value=200,
                value=50,
                step=10,
                help="Overlap between chunks to maintain context"
            )
    
    # Submit button
    if st.button("Get Answer", help="Process your question"):
        if not st.session_state.document_processed:
            st.error("Please upload a document first!")
        elif not question:
            st.error("Please enter a question!")
        else:
            with st.spinner("Generating answer..."):
                # Apply settings
                st.session_state.rag_system.apply_settings({
                    'model': model,
                    'temperature': temperature,
                    'chunk_size': chunk_size,
                    'chunk_overlap': chunk_overlap
                })
                
                # Get answer
                answer = st.session_state.rag_system.get_answer(question)
                
                # Display answer
                st.markdown("### Answer")
                st.write(answer)
                
                # Display sources
                st.markdown("### Sources")
                for source in st.session_state.rag_system.get_sources():
                    st.markdown(f"- {source}")

if __name__ == "__main__":
    main() 
