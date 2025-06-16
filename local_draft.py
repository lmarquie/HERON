# Import necessary libraries
import os
import faiss
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
from config import OPENAI_API_KEY

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
    def __init__(self, dimension: int = 768):
        self.dimension = dimension
        self.index = None
        self.documents = []
        # Use a smaller, faster model
        self.bi_encoder = SentenceTransformer('all-MiniLM-L6-v2')  # Smaller and faster than multilingual
        self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-2-v2')  # Smaller model
        # Optimize model settings
        self.bi_encoder.max_seq_length = 128  # Reduced from 64 for better context
        self.cross_encoder.max_seq_length = 128
        if torch.cuda.is_available():
            self.bi_encoder = self.bi_encoder.to('cuda')
            self.cross_encoder = self.cross_encoder.to('cuda')
            self.bi_encoder = self.bi_encoder.half()
            self.cross_encoder = self.cross_encoder.half()
        self.embedding_cache = {}
        self.similarity_cache = {}
        self.metadata_index = {}
        self.chunk_size = 500
        self.chunk_overlap = 50
        print(f"Initialized VectorStore with optimized models on {self.bi_encoder.device}")

    def save_local(self, path: str):
        """Save the vector store to a local directory"""
        if not os.path.exists(path):
            os.makedirs(path)
        
        # Save the index
        if self.index is not None:
            faiss.write_index(self.index, os.path.join(path, 'index.faiss'))
        
        # Save the documents and metadata
        with open(os.path.join(path, 'documents.pkl'), 'wb') as f:
            pickle.dump({
                'documents': self.documents,
                'metadata_index': self.metadata_index,
                'chunk_size': self.chunk_size,
                'chunk_overlap': self.chunk_overlap
            }, f)
        
        print(f"Vector store saved to {path}")

    def load_local(self, path: str):
        """Load the vector store from a local directory"""
        # Load the index
        index_path = os.path.join(path, 'index.faiss')
        if os.path.exists(index_path):
            self.index = faiss.read_index(index_path)
        
        # Load the documents and metadata
        documents_path = os.path.join(path, 'documents.pkl')
        if os.path.exists(documents_path):
            with open(documents_path, 'rb') as f:
                data = pickle.load(f)
                self.documents = data['documents']
                self.metadata_index = data['metadata_index']
                self.chunk_size = data.get('chunk_size', 500)
                self.chunk_overlap = data.get('chunk_overlap', 50)
        
        # Reinitialize models with proper device placement
        self.bi_encoder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-TinyBERT-L-2-v2')
        self.bi_encoder.max_seq_length = 64
        self.cross_encoder.max_seq_length = 64
        
        if torch.cuda.is_available():
            self.bi_encoder = self.bi_encoder.to('cuda')
            self.cross_encoder = self.cross_encoder.to('cuda')
            self.bi_encoder = self.bi_encoder.half()
            self.cross_encoder = self.cross_encoder.half()
        
        print(f"Vector store loaded from {path}")

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
        # Increased batch size for faster processing
        embeddings = self.bi_encoder.encode(texts,
                                          show_progress_bar=True,
                                          batch_size=64,  # Increased from 32
                                          convert_to_numpy=True,
                                          normalize_embeddings=True)

        if self.index is None:
            # Use a more efficient index type
            if torch.cuda.is_available():
                res = faiss.StandardGpuResources()
                quantizer = faiss.IndexFlatIP(embeddings.shape[1])
                self.index = faiss.IndexIVFFlat(quantizer, embeddings.shape[1],
                                              min(100, len(embeddings)),
                                              faiss.METRIC_INNER_PRODUCT)
                self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
                if not self.index.is_trained and len(embeddings) > 0:
                    self.index.train(embeddings.astype('float32'))
            else:
                self.index = faiss.IndexFlatIP(embeddings.shape[1])
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
        # Check cache first
        cache_key = f"{query}:{k}"
        if cache_key in self.similarity_cache:
            return self.similarity_cache[cache_key]

        # Get query embedding
        query_embedding = self.get_embedding(query)
        
        # Search in FAISS index
        if self.index is not None:
            # Use a larger initial search to get more candidates
            initial_k = min(k * 2, len(self.documents))
            scores, indices = self.index.search(query_embedding.reshape(1, -1).astype('float32'), initial_k)
            
            # Get documents and their scores
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx != -1:  # FAISS returns -1 for empty slots
                    doc = self.documents[idx]
                    results.append({
                        'text': doc['text'],
                        'metadata': doc['metadata'],
                        'score': float(score)
                    })
            
            # Sort by score in descending order and take top k
            results.sort(key=lambda x: x['score'], reverse=True)
            results = results[:k]
            
            # Cache the results
            self.similarity_cache[cache_key] = results
            return results
        
        return []

    def get_embedding(self, text: str):
        if text in self.embedding_cache:
            return self.embedding_cache[text]
        else:
            embedding = self.bi_encoder.encode([text],
                                              batch_size=1,
                                              convert_to_numpy=True,
                                              normalize_embeddings=True)
            self.embedding_cache[text] = embedding
            return embedding

### =================== Claude Handler =================== ###
class ClaudeHandler:
    def __init__(self):
        self.api_key = OPENAI_API_KEY  # This will now work for both local and Litstreams
        self.api_url = "https://api.openai.com/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        self.max_retries = 10
        self.timeout = 600
        self.response_cache = {}
        self.session = requests.Session()

    def generate_answer(self, question: str, context: str) -> str:
        cache_key = f"{question}:{hash(context)}"

        if cache_key in self.response_cache:
            return self.response_cache[cache_key]

        for attempt in range(self.max_retries):
            try:
                messages = [
                    {
                        "role": "system",
                        "content": """You are a document analysis expert. Your task is to analyze documents and provide detailed, accurate answers to questions about them.

Key capabilities:
- Analyze documents and extract relevant information
- Provide clear and concise answers
- Explain complex concepts in simple terms
- Highlight important insights
- Provide context for your answers

Guidelines:
1. Base your answers primarily on the provided context
2. Be precise with information and data
3. Explain terms when used
4. Highlight any uncertainties or missing information
5. Provide relevant context for your analysis
6. Be clear about assumptions made
7. If the context doesn't contain enough information, say so clearly"""
                    },
                    {
                        "role": "user",
                        "content": f"Here are the relevant documents:\n\n{context}\n\nQuestion: {question}"
                    }
                ]

                payload = {
                    "model": "gpt-4-turbo-preview",
                    "messages": messages,
                    "temperature": 0.3,
                    "max_tokens": 2000
                }

                response = self.session.post(
                    self.api_url,
                    headers=self.headers,
                    json=payload,
                    timeout=self.timeout
                )

                response.raise_for_status()
                response_data = response.json()
                answer = response_data["choices"][0]["message"]["content"]
                self.response_cache[cache_key] = answer
                return answer

            except requests.exceptions.Timeout:
                if attempt < self.max_retries - 1:
                    print(f"Request timed out. Retrying... (Attempt {attempt + 1}/{self.max_retries})")
                    time.sleep(2 ** attempt)
                    continue
                else:
                    return "Error: Request timed out after multiple attempts. Please try again."

            except requests.exceptions.RequestException as e:
                error_msg = f"API Error: {str(e)}"
                if hasattr(e, 'response') and e.response is not None:
                    error_msg += f"\nResponse: {e.response.text}"
                return f"Error generating answer: {error_msg}"

            except Exception as e:
                return f"Error generating answer: {str(e)}"

        return "Error: Maximum retry attempts reached. Please try again."

### =================== Question Handler =================== ###
class QuestionHandler:
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store
        self.llm = ClaudeHandler()

    def process_question(self, question: str, query_type: str = "document", k: int = 5) -> str:
        results = self.vector_store.search(question, k=k)

        context_parts = []
        for chunk in results:
            metadata = chunk['metadata']
            source_info = f"Source: {metadata['source']}"
            if 'date' in metadata:
                source_info += f" from {metadata['date']}"
            context_parts.append(f"{source_info}\n{chunk['text']}")
        context = "\n\n".join(context_parts)

        answer = self.llm.generate_answer(question, context)
        return answer

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
            'doc_percentage': 15,
            'num_results': 3
        }
        
        # Update defaults with provided settings
        if settings:
            default_settings.update(settings)
        
        # Apply settings
        self.apply_settings(default_settings)
    
    def apply_settings(self, settings):
        """Apply new settings to the RAG system components."""
        if not settings:
            return
            
        print("\n=== Applying Settings ===")
        print(f"Settings received: {settings}")
            
        # Update vector store settings
        if hasattr(self, 'vector_store') and self.vector_store is not None:
            old_chunk_size = self.vector_store.chunk_size
            old_chunk_overlap = self.vector_store.chunk_overlap
            
            self.vector_store.chunk_size = settings.get('chunk_size', self.vector_store.chunk_size)
            self.vector_store.chunk_overlap = settings.get('chunk_overlap', self.vector_store.chunk_overlap)
            
            print(f"Vector Store Settings Updated:")
            print(f"  Chunk Size: {old_chunk_size} -> {self.vector_store.chunk_size}")
            print(f"  Chunk Overlap: {old_chunk_overlap} -> {self.vector_store.chunk_overlap}")
            
        # Update question handler settings
        if hasattr(self, 'question_handler') and self.question_handler is not None:
            if hasattr(self.question_handler, 'llm') and self.question_handler.llm is not None:
                # Store the temperature setting in the class for later use
                self.question_handler.llm.temperature = settings.get('model_temperature', 0.3)
                print(f"LLM Settings Updated:")
                print(f"  Temperature: {self.question_handler.llm.temperature}")
        print("=== Settings Applied ===\n")

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
