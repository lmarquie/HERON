# Import necessary libraries
import os
import faiss
import base64
import fitz  # PyMuPDF
import pickle
import torch
import time
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoModelForCausalLM, AutoTokenizer
import requests
import io
import numpy as np
import gc

### =================== Input Interface =================== ###
class InputInterface:
    def __init__(self):
        self.download_dir = "local_documents"
        os.makedirs(self.download_dir, exist_ok=True)

    def get_user_question(self) -> str:
        return input("\nEnter your question: ").strip()

    def display_answer(self, answer: str):
        print("\nAnswer:", answer)

### =================== Document Loading =================== ###
class LocalFileHandler:
    def __init__(self):
        self.base_path = os.getcwd()

    def select_folder_interactive(self):
        """Let user select a folder from local system"""
        print("\nPlease enter the path to your folder containing documents.")
        print("Example: /path/to/your/documents")
        print("Or just the folder name if it's in the current directory")

        while True:
            folder_path = input("\nEnter folder path: ").strip()

            if not os.path.isabs(folder_path):
                folder_path = os.path.join(self.base_path, folder_path)

            if os.path.exists(folder_path) and os.path.isdir(folder_path):
                print(f"✓ Selected folder: {folder_path}")
                return {
                    'name': os.path.basename(folder_path),
                    'path': folder_path
                }
            else:
                print(f" Folder not found: {folder_path}")
                print("Please try again or enter 'q' to quit")
                if folder_path.lower() == 'q':
                    return None

    def process_selected_folder(self, folder_path):
        """Process all files in the selected folder"""
        try:
            print(f"\nScanning folder: {folder_path}")
            files = []
            for item in os.listdir(folder_path):
                full_path = os.path.join(folder_path, item)
                if os.path.isfile(full_path):
                    files.append({
                        'name': item,
                        'path': full_path
                    })

            if not files:
                print("No files found in the selected folder.")
                return []

            print(f"\nFound {len(files)} files. Processing...")
            documents = []
            for file in files:
                if file['name'].endswith(('.txt', '.pdf')):
                    try:
                        if file['name'].endswith('.pdf'):
                            doc = fitz.open(file['path'])
                            content = ""
                            for page in doc:
                                content += page.get_text()
                            doc.close()
                        else:
                            with open(file['path'], 'r', encoding='utf-8') as f:
                                content = f.read()

                        documents.append({
                            'text': content,
                            'metadata': {
                                'source': file['name'],
                                'path': file['path'],
                                'date': datetime.now().isoformat()
                            }
                        })
                        print(f"✓ Processed: {file['name']}")
                    except Exception as e:
                        print(f" Error processing {file['name']}: {str(e)}")

            return documents
        except Exception as e:
            print(f" Error processing folder: {str(e)}")
            return []

### =================== Text Processing =================== ###
class TextProcessor:
    def __init__(self, chunk_size: int = 500, overlap: int = 30):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        try:
            doc = fitz.open(pdf_path)
            text = "\n".join([page.get_text() for page in doc])
            return text
        except Exception as e:
            print(f"Error extracting text from {pdf_path}: {e}")
            return ""

    def chunk_text(self, text: str) -> List[str]:
        words = text.split()
        chunks = []
        for i in range(0, len(words), self.chunk_size - self.overlap):
            chunk = " ".join(words[i:i + self.chunk_size])
            if chunk.strip():
                chunks.append(chunk)
        return chunks

    def prepare_documents(self, pdf_paths: List[str]) -> List[Dict]:
        documents = []
        for pdf_path in pdf_paths:
            text = self.extract_text_from_pdf(pdf_path)
            chunks = self.chunk_text(text)
            for i, chunk in enumerate(chunks):
                documents.append({
                    "text": chunk,
                    "metadata": {
                        "source": os.path.basename(pdf_path),
                        "chunk_id": i,
                        "timestamp": datetime.now().isoformat()
                    }
                })
        return documents

### =================== Vector Store =================== ###
class VectorStore:
    def __init__(self):
        self.chunk_size = 500
        self.chunk_overlap = 50
        self.index = None
        self.documents = []
        self.model = None
        self.cross_encoder = None
        self._initialize_models()

    def _initialize_models(self):
        """Initialize models with memory optimization"""
        try:
            # Use smaller model for free tier
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
            
            # Enable half precision
            if torch.cuda.is_available():
                self.model = self.model.half()
                self.cross_encoder = self.cross_encoder.half()
            
            # Clear CUDA cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"Error initializing models: {str(e)}")
            raise

    def add_documents(self, documents: List[Dict]):
        """Add documents to the vector store with memory optimization"""
        try:
            # Process documents in smaller batches
            batch_size = 32
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                self._process_batch(batch)
                
                # Clear memory after each batch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                
        except Exception as e:
            print(f"Error adding documents: {str(e)}")
            raise

    def _process_batch(self, batch: List[Dict]):
        """Process a batch of documents"""
        try:
            # Extract text and metadata
            texts = []
            metadatas = []
            
            for doc in batch:
                chunks = self._chunk_text(doc['text'])
                texts.extend(chunks)
                metadatas.extend([doc['metadata']] * len(chunks))
            
            # Generate embeddings in smaller sub-batches
            sub_batch_size = 16
            embeddings = []
            
            for i in range(0, len(texts), sub_batch_size):
                sub_batch = texts[i:i + sub_batch_size]
                batch_embeddings = self.model.encode(
                    sub_batch,
                    show_progress_bar=False,
                    convert_to_tensor=True
                )
                embeddings.append(batch_embeddings)
                
                # Clear memory after each sub-batch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Combine embeddings
            embeddings = torch.cat(embeddings, dim=0)
            
            # Create or update FAISS index
            if self.index is None:
                dimension = embeddings.shape[1]
                self.index = faiss.IndexFlatL2(dimension)
            
            # Add to index
            self.index.add(embeddings.cpu().numpy())
            
            # Store documents
            for text, metadata in zip(texts, metadatas):
                self.documents.append({
                    'text': text,
                    'metadata': metadata
                })
                
        except Exception as e:
            print(f"Error processing batch: {str(e)}")
            raise

    def search(self, query: str, k: int = 5) -> List[Dict]:
        """Search for similar documents with memory optimization"""
        try:
            # Generate query embedding
            query_embedding = self.model.encode(
                [query],
                show_progress_bar=False,
                convert_to_tensor=True
            )
            
            # Search in FAISS index
            distances, indices = self.index.search(
                query_embedding.cpu().numpy(),
                k
            )
            
            # Get results
            results = []
            for idx in indices[0]:
                if idx < len(self.documents):
                    results.append(self.documents[idx])
            
            # Clear memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return results
            
        except Exception as e:
            print(f"Error searching: {str(e)}")
            return []

    def _chunk_text(self, text: str) -> List[str]:
        """Split text into chunks with overlap"""
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            end = start + self.chunk_size
            if end > text_length:
                end = text_length
            chunks.append(text[start:end])
            start = end - self.chunk_overlap
            
        return chunks

### =================== OpenAI Handler =================== ###
class OpenAIHandler:
    def __init__(self):
        encoded_key = "c2stcHJvai1aWm9VV0NKTnV3NXByRURZdWVrblNhM1hXVklWdUJqVmZTUmlkd1lhanVKQjlDYjFZSUVLSWhqak1jcHBxb1dZckxNb016MDBWRFQzQmxia0ZKY3N4ZlVwQkI5V2xCUXRZNXdFd3lMbXFZaHliQ3c0TF9HT2R5ZE1HczZycjJia1diVlloc1Rha0ZwcUY3QVdUMW0zNDZmVlRqY0E="
        try:
            self.api_key = base64.b64decode(encoded_key).decode('utf-8')
            print(f"Decoded API key starts with: {self.api_key[:10]}...")  # Only print first 10 chars for security
        except Exception as e:
            print(f"Error decoding API key: {str(e)}")
            raise
            
        self.api_url = "https://api.openai.com/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        print(f"Authorization header starts with: {self.headers['Authorization'][:20]}...")  # Only print first 20 chars for security
        self.max_retries = 2
        self.timeout = 30
        self.response_cache = {}
        self.session = requests.Session()
        
        # Optimized model settings for free tier
        self.temperature = 0.1
        self.max_tokens = 300  # Reduced for memory efficiency
        self.model = "gpt-4-turbo-preview"
        self.presence_penalty = 0.1
        self.frequency_penalty = 0.1
        
        # Required attributes for compatibility
        self.sequence_length = 256
        self.batch_size = 64  # Reduced for memory efficiency
        self.use_half_precision = True

        # Rate limit settings
        self.max_context_tokens = 4000  # Reduced for memory efficiency
        self.rate_limit_delay = 15
        
        # Cache settings
        self.max_cache_size = 100  # Limit cache size
        self.cache_ttl = 3600  # Cache time-to-live in seconds

    def _clean_cache(self):
        """Clean old entries from cache"""
        current_time = time.time()
        self.response_cache = {
            k: v for k, v in self.response_cache.items()
            if current_time - v['timestamp'] < self.cache_ttl
        }
        # If still too large, remove oldest entries
        if len(self.response_cache) > self.max_cache_size:
            sorted_cache = sorted(
                self.response_cache.items(),
                key=lambda x: x[1]['timestamp']
            )
            self.response_cache = dict(sorted_cache[-self.max_cache_size:])

    def truncate_context(self, context: str) -> str:
        """Truncate context to stay within token limits while preserving important information."""
        max_chars = self.max_context_tokens * 4
        if len(context) > max_chars:
            # Try to truncate at paragraph boundaries first
            paragraphs = context.split('\n\n')
            truncated = ''
            for para in paragraphs:
                if len(truncated) + len(para) + 2 <= max_chars:
                    truncated += para + '\n\n'
                else:
                    break
            
            # If we still have space, try to add partial paragraphs
            if len(truncated) < max_chars * 0.8:
                remaining_chars = max_chars - len(truncated)
                truncated += context[len(truncated):len(truncated) + remaining_chars]
            
            return truncated + "\n[Context truncated for optimization]"
        return context

    def generate_answer(self, question: str, context: str) -> str:
        cache_key = f"{question}:{hash(context)}"
        current_time = time.time()

        # Check cache
        if cache_key in self.response_cache:
            cache_entry = self.response_cache[cache_key]
            if current_time - cache_entry['timestamp'] < self.cache_ttl:
                return cache_entry['response']

        # Clean cache before new request
        self._clean_cache()

        # Truncate context to stay within token limits
        truncated_context = self.truncate_context(context)

        for attempt in range(self.max_retries):
            try:
                # Optimized system prompt
                system_prompt = """Financial expert. Analyze documents and provide concise answers.
Focus on:
- Key financial metrics
- Important trends
- Critical insights
- Clear explanations

Guidelines:
1. Be precise and concise
2. Focus on key points
3. Explain terms briefly
4. Note uncertainties"""

                messages = [
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": f"Context:\n{truncated_context}\n\nQuestion: {question}"
                    }
                ]

                payload = {
                    "model": self.model,
                    "max_tokens": self.max_tokens,
                    "messages": messages,
                    "temperature": self.temperature,
                    "presence_penalty": self.presence_penalty,
                    "frequency_penalty": self.frequency_penalty,
                    "stream": False
                }

                response = self.session.post(
                    self.api_url,
                    headers=self.headers,
                    json=payload,
                    timeout=self.timeout
                )

                if response.status_code == 429:  # Rate limit error
                    retry_after = int(response.headers.get('Retry-After', self.rate_limit_delay))
                    print(f"Rate limit hit. Waiting {retry_after} seconds...")
                    time.sleep(retry_after)
                    continue

                response.raise_for_status()
                response_data = response.json()
                answer = response_data["choices"][0]["message"]["content"]
                
                # Cache the response
                self.response_cache[cache_key] = {
                    'response': answer,
                    'timestamp': current_time
                }
                
                return answer

            except requests.exceptions.Timeout:
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(2 ** attempt)  # Exponential backoff
            except requests.exceptions.RequestException as e:
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(2 ** attempt)  # Exponential backoff

### =================== Question Handler =================== ###
class QuestionHandler:
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store
        self.llm = OpenAIHandler()

    def process_question(self, question: str, query_type: str = "document", k: int = 3) -> str:  # Reduced k from 5 to 3
        # Get most relevant chunks
        results = self.vector_store.search(question, k=k)

        # Optimize context building
        context_parts = []
        for chunk in results:
            metadata = chunk['metadata']
            # Only include essential metadata
            source_info = f"Source: {metadata['source']}"
            if 'date' in metadata and metadata['date']:
                source_info += f" ({metadata['date']})"
            # Add chunk text with minimal formatting
            context_parts.append(f"{source_info}\n{chunk['text']}")
        
        # Join with minimal formatting
        context = "\n\n".join(context_parts)

        # Generate answer
        return self.llm.generate_answer(question, context)

### =================== Main RAG System =================== ###
class RAGSystem:
    def __init__(self, settings=None):
        self.file_handler = LocalFileHandler()
        self.vector_store = VectorStore()
        self.question_handler = QuestionHandler(self.vector_store)
        self.running = True
        
        # Initialize with default settings
        default_settings = {
            'chunk_size': 500,
            'chunk_overlap': 50,
            'model_temperature': 0.3,
            'sequence_length': 256,
            'batch_size': 128,
            'use_half_precision': True,
            'num_documents': 5,  # Fixed number of best documents to use
            'num_results': 3
        }
        
        # Update defaults with provided settings
        if settings:
            default_settings.update(settings)
        
        # Apply settings
        self.apply_settings(default_settings)
    
    def apply_settings(self, settings):
        """Apply settings to the RAG system"""
        # Update vector store settings
        self.vector_store.chunk_size = settings.get('chunk_size', 500)
        self.vector_store.chunk_overlap = settings.get('chunk_overlap', 50)
        
        # Update question handler settings
        self.question_handler.llm.temperature = settings.get('model_temperature', 0.3)
        self.question_handler.llm.sequence_length = settings.get('sequence_length', 256)
        self.question_handler.llm.batch_size = settings.get('batch_size', 128)
        self.question_handler.llm.use_half_precision = settings.get('use_half_precision', True)
        
        # Update number of documents to use
        self.num_documents = settings.get('num_documents', 5)
        
        # Store settings
        self.settings = settings

    def initialize(self):
        print("\n=== Welcome to the Local RAG System ===")

        selected_folder = self.file_handler.select_folder_interactive()
        if not selected_folder:
            print("No folder selected. Please try again.")
            return self.initialize()

        print(f"\nProcessing folder: {selected_folder['name']}")
        documents = self.file_handler.process_selected_folder(selected_folder['path'])

        if not documents:
            print("No documents found to process. Please try again.")
            return self.initialize()

        print("\nIndexing documents...")
        self.vector_store.add_documents(documents)
        print("Documents indexed successfully!")
        return True

    def show_menu(self):
        print("\n=== RAG System Menu ===")
        print("1. Ask a question")
        print("2. Select a different folder")
        print("3. Exit")
        return input("Enter your choice (1-3): ")

    def run(self):
        if not self.initialize():
            return

        while self.running:
            choice = self.show_menu()

            if choice == "1":
                question = input("\nEnter your question: ")
                answer = self.question_handler.process_question(question)
                print("\nAnswer:", answer)

            elif choice == "2":
                if self.initialize():
                    print("Successfully switched to new folder!")
                else:
                    print("Failed to switch folders.")

            elif choice == "3":
                print("\nThank you for using the RAG System. Goodbye!")
                self.running = False

            else:
                print("\nInvalid choice. Please try again.")

if __name__ == "__main__":
    rag_system = RAGSystem()
    rag_system.run() 
