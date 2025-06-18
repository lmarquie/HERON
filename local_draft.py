# Import necessary libraries
import os
import faiss
import fitz  # PyMuPDF
import time
from typing import List, Dict
from datetime import datetime
from sentence_transformers import SentenceTransformer
import requests
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

### =================== Text Processing =================== ###
class TextProcessor:
    def __init__(self, chunk_size: int = 500, overlap: int = 30):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF - text only, no image processing."""
        try:
            print(f"Processing PDF: {pdf_path}")
            doc = fitz.open(pdf_path)
            text_content = []
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text = page.get_text()
                text_content.append(f"Page {page_num + 1}: {text}")
            
            doc.close()
            final_content = "\n".join(text_content)
            print(f"PDF processing completed. Total content length: {len(final_content)}")
            return final_content
            
        except Exception as e:
            print(f"Error extracting text from PDF: {str(e)}")
            return f"Error extracting text from PDF: {str(e)}"

    def chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks."""
        if not text.strip():
            return []
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            if end >= len(text):
                chunks.append(text[start:])
                break
            
            # Try to break at a sentence boundary
            last_period = text.rfind('.', start, end)
            last_newline = text.rfind('\n', start, end)
            
            if last_period > start and last_period > last_newline:
                end = last_period + 1
            elif last_newline > start:
                end = last_newline + 1
            
            chunks.append(text[start:end])
            start = end - self.overlap
            
            if start >= len(text):
                break
        
        return chunks

    def prepare_documents(self, pdf_paths: List[str]) -> List[Dict]:
        """Prepare documents for vector storage."""
        documents = []
        
        for pdf_path in pdf_paths:
            try:
                # Extract text from PDF
                text_content = self.extract_text_from_pdf(pdf_path)
                
                if text_content.strip():
                    # Split into chunks
                    chunks = self.chunk_text(text_content)
                    
                    for i, chunk in enumerate(chunks):
                        if chunk.strip():
                            documents.append({
                                'text': chunk.strip(),
                                'metadata': {
                                    'source': pdf_path,
                                    'chunk_id': i,
                                    'total_chunks': len(chunks),
                                    'date': datetime.now().strftime('%Y-%m-%d')
                                }
                            })
                
            except Exception as e:
                print(f"Error processing {pdf_path}: {str(e)}")
                continue
        
        return documents

### =================== Document Loading =================== ###
class LocalFileHandler:
    def __init__(self):
        self.text_processor = TextProcessor()

    def select_folder_interactive(self):
        """Allow user to select a folder interactively"""
        print("\nPlease select a folder containing your documents:")
        print("1. Enter folder path manually")
        print("2. Browse current directory")
        
        choice = input("Enter your choice (1-2): ")
        
        if choice == "1":
            folder_path = input("Enter the folder path: ").strip()
            if os.path.exists(folder_path) and os.path.isdir(folder_path):
                return {
                    'name': os.path.basename(folder_path),
                    'path': folder_path
                }
            else:
                print("Invalid folder path. Please try again.")
                return self.select_folder_interactive()
        
        elif choice == "2":
            current_dir = os.getcwd()
            print(f"\nCurrent directory: {current_dir}")
            print("\nAvailable folders:")
            
            folders = [item for item in os.listdir(current_dir) 
                      if os.path.isdir(os.path.join(current_dir, item))]
            
            if not folders:
                print("No folders found in current directory.")
                return self.select_folder_interactive()
            
            for i, folder in enumerate(folders, 1):
                print(f"{i}. {folder}")
            
            try:
                folder_choice = int(input(f"\nSelect folder (1-{len(folders)}): "))
                if 1 <= folder_choice <= len(folders):
                    selected_folder = folders[folder_choice - 1]
                    folder_path = os.path.join(current_dir, selected_folder)
                    return {
                        'name': selected_folder,
                        'path': folder_path
                    }
                else:
                    print("Invalid choice. Please try again.")
                    return self.select_folder_interactive()
            except ValueError:
                print("Invalid input. Please try again.")
                return self.select_folder_interactive()
        
        else:
            print("Invalid choice. Please try again.")
            return self.select_folder_interactive()

    def process_selected_folder(self, folder_path):
        """Process all files in the selected folder using TextProcessor with image recognition"""
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

            print(f"\nFound {len(files)} files. Processing with image recognition...")
            documents = []
            for file in files:
                if file['name'].endswith(('.txt', '.pdf')):
                    try:
                        if file['name'].endswith('.pdf'):
                            # Use TextProcessor for PDF processing with image recognition
                            content = self.text_processor.extract_text_from_pdf(file['path'])
                        else:
                            # For text files, use simple text extraction
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
        self.text_processor = TextProcessor()

    def process_uploaded_files(self, uploaded_files):
        """Process files uploaded through Streamlit."""
        documents = []
        
        for uploaded_file in uploaded_files:
            try:
                # Save uploaded file temporarily
                temp_path = f"temp/{uploaded_file.name}"
                os.makedirs("temp", exist_ok=True)
                
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Extract text from PDF
                text_content = self.text_processor.extract_text_from_pdf(temp_path)
                
                if text_content.strip():
                    # Split into chunks
                    chunks = self.text_processor.chunk_text(text_content)
                    
                    for i, chunk in enumerate(chunks):
                        if chunk.strip():
                            documents.append({
                                'text': chunk.strip(),
                                'metadata': {
                                    'source': uploaded_file.name,
                                    'chunk_id': i,
                                    'total_chunks': len(chunks),
                                    'date': datetime.now().strftime('%Y-%m-%d')
                                }
                            })
                
                # Clean up temp file
                os.remove(temp_path)
                
            except Exception as e:
                print(f"Error processing {uploaded_file.name}: {str(e)}")
                continue
        
        return documents

### =================== Vector Store =================== ###
class VectorStore:
    def __init__(self, dimension: int = 768):
        self.documents = []
        self.embeddings = None
        self.index = None
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.chunk_size = 500
        self.chunk_overlap = 50

    def add_documents(self, documents: List[Dict]):
        """Add documents to the vector store."""
        if not documents:
            return
            
        print(f"Adding {len(documents)} documents to vector store...")
        
        # Extract text from documents
        texts = [doc['text'] for doc in documents]
        
        # Generate embeddings
        print("Converting text to vector embeddings...")
        embeddings = self.model.encode(texts, show_progress_bar=True)
        
        # Store documents and embeddings
        self.documents.extend(documents)
        
        if self.embeddings is None:
            self.embeddings = embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, embeddings])
        
        # Create FAISS index with correct dimension
        embedding_dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(embedding_dim)
        self.index.add(self.embeddings.astype('float32'))
        
        print(f"Successfully added {len(documents)} documents. Total documents: {len(self.documents)}")

    def search(self, query: str, k: int = 3) -> List[Dict]:
        """Search for similar documents."""
        if not self.documents or self.index is None:
            return []
        
        # Generate query embedding
        query_embedding = self.model.encode([query])
        
        # Search
        scores, indices = self.index.search(query_embedding.astype('float32'), k)
        
        # Return results
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.documents):
                results.append({
                    'text': self.documents[idx]['text'],
                    'metadata': self.documents[idx].get('metadata', {}),
                    'score': float(scores[0][i])
                })
        
        return results

### =================== Claude Handler =================== ###
class ClaudeHandler:
    def __init__(self, system_prompt=None):
        self.api_url = "https://api.openai.com/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json"
        }
        self.system_prompt = system_prompt or "You are a helpful assistant. Answer questions based on the provided context."

    def generate_answer(self, question: str, context: str) -> str:
        try:
            messages = [
                {
                    "role": "system",
                    "content": self.system_prompt
                },
                {
                    "role": "user",
                    "content": f"Context: {context}\n\nQuestion: {question}"
                }
            ]

            payload = {
                "model": "gpt-4-turbo-preview",
                "messages": messages,
                "temperature": 0.3,
                "max_tokens": 1000
            }

            response = requests.post(
                self.api_url,
                headers=self.headers,
                json=payload,
                timeout=30
            )

            response.raise_for_status()
            response_data = response.json()
            return response_data["choices"][0]["message"]["content"]

        except Exception as e:
            return f"Error generating answer: {str(e)}"

### =================== Question Handler =================== ###
class QuestionHandler:
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store
        self.llm = ClaudeHandler()
        self.conversation_history = []

    def process_question(self, question: str, query_type: str = "document", k: int = 5) -> str:
        results = self.vector_store.search(question, k=k)
        
        if not results:
            return "No relevant information found in the documents."
        
        # Simple context building
        context = "\n".join([chunk['text'] for chunk in results])
        answer = self.llm.generate_answer(question, context)
        
        # Store conversation history
        self.conversation_history.append({
            'question': question,
            'answer': answer,
            'context': context
        })
        
        return answer

    def process_follow_up(self, follow_up_question: str, k: int = 5) -> str:
        """Process a follow-up question using conversation history and document context."""
        if not self.conversation_history:
            return "No previous conversation to follow up on. Please ask a question first."
        
        # Get recent conversation context
        recent_context = ""
        for i, conv in enumerate(self.conversation_history[-3:]):  # Last 3 exchanges
            recent_context += f"Previous Q: {conv['question']}\nPrevious A: {conv['answer']}\n\n"
        
        # Get document context
        results = self.vector_store.search(follow_up_question, k=k)
        document_context = "\n".join([chunk['text'] for chunk in results]) if results else ""
        
        # Combine contexts
        full_context = f"Conversation History:\n{recent_context}\nDocument Context:\n{document_context}"
        
        # Generate follow-up answer
        answer = self.llm.generate_answer(follow_up_question, full_context)
        
        # Store in conversation history
        self.conversation_history.append({
            'question': follow_up_question,
            'answer': answer,
            'context': full_context
        })
        
        return answer

    def clear_conversation_history(self):
        """Clear the conversation history."""
        self.conversation_history = []

### =================== Main RAG System =================== ###
class RAGSystem:
    def __init__(self, settings=None, is_web=False, use_vision_api=True):
        self.file_handler = WebFileHandler() if is_web else LocalFileHandler()
        self.vector_store = VectorStore()
        self.question_handler = QuestionHandler(self.vector_store)
        self.is_web = is_web
        self.conversation_history = []

    def process_web_uploads(self, uploaded_files):
        """Process files uploaded through the web interface"""
        if not self.is_web:
            return False
            
        documents = self.file_handler.process_uploaded_files(uploaded_files)
        if documents:
            self.vector_store.add_documents(documents)
            return True
        return False

    def add_to_conversation_history(self, question, answer, question_type="initial"):
        """Add a Q&A pair to conversation history"""
        self.conversation_history.append({
            'question': question,
            'answer': answer,
            'type': question_type
        })

    def get_conversation_history(self):
        """Get the conversation history"""
        return self.conversation_history

    def clear_conversation_history(self):
        """Clear the conversation history"""
        self.conversation_history = []

if __name__ == "__main__": 
    rag_system = RAGSystem()
    rag_system.run() 
