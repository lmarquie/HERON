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

class WebFileHandler(LocalFileHandler):
    def __init__(self):
        super().__init__()
        self.temp_dir = "temp"
        os.makedirs(self.temp_dir, exist_ok=True)

    def process_uploaded_files(self, uploaded_files):
        """Process files uploaded through Streamlit's file uploader"""
        if not uploaded_files:
            return []

        documents = []
        for uploaded_file in uploaded_files:
            try:
                # Save the uploaded file to a temporary location
                temp_path = os.path.join(self.temp_dir, uploaded_file.name)
                with open(temp_path, 'wb') as f:
                    f.write(uploaded_file.getbuffer())

                # Process the file based on its type
                if uploaded_file.name.endswith('.pdf'):
                    doc = fitz.open(temp_path)
                    content = ""
                    for page in doc:
                        content += page.get_text()
                    doc.close()
                elif uploaded_file.name.endswith('.txt'):
                    with open(temp_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                else:
                    continue

                documents.append({
                    'text': content,
                    'metadata': {
                        'source': uploaded_file.name,
                        'path': temp_path,
                        'date': datetime.now().isoformat()
                    }
                })

                # Clean up the temporary file
                os.remove(temp_path)

            except Exception as e:
                print(f"Error processing {uploaded_file.name}: {str(e)}")
                continue

        return documents

### =================== Text Processing =================== ###
class TextProcessor:
    def __init__(self, chunk_size: int = 800, overlap: int = 100):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.sentence_splitter = None  # Will be initialized on first use

    def _initialize_sentence_splitter(self):
        """Lazy initialization of sentence splitter"""
        if self.sentence_splitter is None:
            from nltk.tokenize import sent_tokenize
            import nltk
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                nltk.download('punkt')
            self.sentence_splitter = sent_tokenize

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        try:
            doc = fitz.open(pdf_path)
            text = ""
            for page in doc:
                text += page.get_text("text")  # Using "text" mode for better formatting
            doc.close()
            return text
        except Exception as e:
            print(f"Error extracting text from {pdf_path}: {e}")
            return ""

    def chunk_text(self, text: str) -> List[str]:
        """Improved text chunking that respects sentence boundaries"""
        self._initialize_sentence_splitter()
        
        # Split into sentences first
        sentences = self.sentence_splitter(text)
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence.split())
            
            # If adding this sentence would exceed chunk size, save current chunk
            if current_length + sentence_length > self.chunk_size and current_chunk:
                chunks.append(" ".join(current_chunk))
                # Keep overlap sentences for context
                overlap_sentences = []
                overlap_length = 0
                for s in reversed(current_chunk):
                    if overlap_length + len(s.split()) <= self.overlap:
                        overlap_sentences.insert(0, s)
                        overlap_length += len(s.split())
                    else:
                        break
                current_chunk = overlap_sentences
                current_length = overlap_length
            
            current_chunk.append(sentence)
            current_length += sentence_length
        
        # Add the last chunk if it exists
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
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
                        "timestamp": datetime.now().isoformat(),
                        "chunk_size": len(chunk.split())
                    }
                })
        return documents

### =================== Vector Store =================== ###
class VectorStore:
    def __init__(self, dimension: int = 1536):  # OpenAI embeddings are 1536-dimensional
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)  # Using L2 distance for better accuracy
        self.documents = []
        self.embedding_model = None
        self.cross_encoder = None
        self.chunk_size = 800  # Default chunk size
        self.chunk_overlap = 100  # Default chunk overlap
        self._initialize_models()

    def _initialize_models(self):
        """Initialize embedding models with optimized settings"""
        if self.embedding_model is None:
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            # Optimize model for inference
            self.embedding_model.eval()
            if torch.cuda.is_available():
                self.embedding_model = self.embedding_model.cuda()
        
        if self.cross_encoder is None:
            self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
            # Optimize model for inference
            self.cross_encoder.eval()
            if torch.cuda.is_available():
                self.cross_encoder = self.cross_encoder.cuda()

    def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding with optimized batch processing"""
        with torch.no_grad():  # Disable gradient calculation for inference
            embeddings = self.embedding_model.encode(
                [text],
                convert_to_tensor=True,
                show_progress_bar=False,
                normalize_embeddings=True  # Normalize for better cosine similarity
            )
            if torch.cuda.is_available():
                embeddings = embeddings.cpu()
            return embeddings.numpy()

    def add_documents(self, documents: List[Dict]):
        """Add documents with optimized batch processing"""
        if not documents:
            return

        # Prepare texts for batch processing
        texts = [doc["text"] for doc in documents]
        
        # Get embeddings in batch
        with torch.no_grad():
            embeddings = self.embedding_model.encode(
                texts,
                convert_to_tensor=True,
                show_progress_bar=False,
                normalize_embeddings=True
            )
            if torch.cuda.is_available():
                embeddings = embeddings.cpu()
            embeddings = embeddings.numpy()

        # Add to FAISS index
        self.index.add(embeddings)
        self.documents.extend(documents)

    def search(self, query: str, k: int = 3) -> List[Dict]:
        """Enhanced search with reranking"""
        # Get query embedding
        query_embedding = self.get_embedding(query)
        
        # First stage: FAISS search
        distances, indices = self.index.search(query_embedding, k * 2)  # Get more candidates for reranking
        
        # Get candidate documents
        candidates = [self.documents[i] for i in indices[0]]
        
        # Second stage: Cross-encoder reranking
        pairs = [(query, doc["text"]) for doc in candidates]
        with torch.no_grad():
            scores = self.cross_encoder.predict(pairs)
        
        # Combine scores and sort
        scored_candidates = list(zip(candidates, scores))
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Return top k results
        return [doc for doc, _ in scored_candidates[:k]]

    def save_local(self, path: str):
        """Save the vector store to a local file"""
        try:
            with open(path, 'wb') as f:
                pickle.dump({
                    'documents': self.documents,
                    'embedding_model': self.embedding_model,
                    'cross_encoder': self.cross_encoder,
                    'chunk_size': self.chunk_size,
                    'chunk_overlap': self.chunk_overlap
                }, f)
            print(f"Vector store saved to {path}")
        except Exception as e:
            print(f"Error saving vector store: {e}")

    def load_local(self, path: str):
        """Load the vector store from a local file"""
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
                self.documents = data['documents']
                self.embedding_model = data['embedding_model']
                self.cross_encoder = data['cross_encoder']
                self.chunk_size = data.get('chunk_size', 800)
                self.chunk_overlap = data.get('chunk_overlap', 100)
            print(f"Vector store loaded from {path}")
        except Exception as e:
            print(f"Error loading vector store: {e}")

### =================== Claude Handler =================== ###
class ClaudeHandler:
    def __init__(self):
        self.api_key = OPENAI_API_KEY
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
                # Truncate context if it's too long
                max_context_length = 20000  # Reduced from 40000 to ~5k tokens
                if len(context) > max_context_length:
                    context = context[:max_context_length] + "..."

                messages = [
                    {
                        "role": "system",
                        "content": "Answer based on the context."  # Minimal system prompt
                    },
                    {
                        "role": "user",
                        "content": f"Context: {context}\nQ: {question}"  # Simplified format
                    }
                ]

                payload = {
                    "model": "gpt-4-turbo-preview",
                    "messages": messages,
                    "temperature": 0.3,
                    "max_tokens": 1000  # Reduced from 2000
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
        self.llm = None
        self._initialize_llm()

    def _initialize_llm(self):
        """Initialize the language model with optimized settings"""
        if self.llm is None:
            self.llm = AutoModelForCausalLM.from_pretrained(
                "gpt2",  # Using a smaller model for faster inference
                torch_dtype=torch.float16,  # Use half precision for faster inference
                low_cpu_mem_usage=True
            )
            if torch.cuda.is_available():
                self.llm = self.llm.cuda()
            self.llm.eval()  # Set to evaluation mode

    def process_question(self, question: str, query_type: str = "document", k: int = 5) -> str:
        """Process a single question"""
        try:
            # Get relevant documents
            results = self.vector_store.search(question, k=k)
            
            # Prepare context
            context = "\n".join([r["text"] for r in results])
            
            # Generate answer
            with torch.no_grad():
                answer = self.llm.generate_answer(question, context)
            
            return answer
        except Exception as e:
            print(f"Error processing question: {e}")
            return "Error processing question"

    def process_questions_batch(self, questions: List[str], query_type: str = "document", k: int = 5) -> List[str]:
        """Process multiple questions in parallel"""
        try:
            # Get relevant documents for all questions at once
            all_results = []
            for question in questions:
                results = self.vector_store.search(question, k=k)
                all_results.append(results)
            
            # Prepare contexts
            contexts = ["\n".join([r["text"] for r in results]) for results in all_results]
            
            # Generate answers in parallel
            with torch.no_grad():
                answers = []
                for question, context in zip(questions, contexts):
                    answer = self.llm.generate_answer(question, context)
                    answers.append(answer)
            
            return answers
        except Exception as e:
            print(f"Error processing questions batch: {e}")
            return ["Error processing question"] * len(questions)

### =================== Main RAG System =================== ###
class RAGSystem:
    def __init__(self, settings=None, is_web=False):
        self.file_handler = WebFileHandler() if is_web else LocalFileHandler()
        self.vector_store = VectorStore()
        self.question_handler = QuestionHandler(self.vector_store)
        self.running = True
        self.is_web = is_web
        
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

    def process_web_uploads(self, uploaded_files):
        """Process files uploaded through the web interface"""
        if not self.is_web:
            return False
            
        documents = self.file_handler.process_uploaded_files(uploaded_files)
        if documents:
            self.vector_store.add_documents(documents)
            return True
        return False

if __name__ == "__main__": 
    rag_system = RAGSystem()
    rag_system.run() 
