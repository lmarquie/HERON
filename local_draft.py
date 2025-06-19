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
from PIL import Image
import io
import base64
import re
import traceback

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
        self.extracted_images = {}  # Store images for display
        self.image_descriptions = {}  # Store semantic descriptions for search

    def clean_extracted_text(self, text: str) -> str:
        """Clean up extracted text to fix common PDF extraction issues."""
        try:
            # Fix concatenated words by adding spaces between transitions from lowercase to uppercase
            text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
            
            # Fix cases where numbers are concatenated with words
            text = re.sub(r'([a-zA-Z])(\d)', r'\1 \2', text)
            text = re.sub(r'(\d)([a-zA-Z])', r'\1 \2', text)
            
            # Fix cases where parentheses are concatenated with words
            text = re.sub(r'([a-zA-Z])(\()', r'\1 \2', text)
            text = re.sub(r'(\))([a-zA-Z])', r'\1 \2', text)
            
            # Remove excessive whitespace
            text = ' '.join(text.split())
            
            # Fix common PDF artifacts
            text = text.replace('•', '- ')  # Replace bullets with dashes
            text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
            text = re.sub(r'([a-z])([A-Z][a-z])', r'\1 \2', text)  # Split camelCase
            
            return text
        except Exception as e:
            print(f"Error cleaning text: {e}")
            return text

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text and images from PDF - simple and fast."""
        try:
            print(f"Processing PDF: {pdf_path}")
            doc = fitz.open(pdf_path)
            text_content = []
            image_content = []
            
            # Create images directory
            os.makedirs("images", exist_ok=True)
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                
                # Extract text and clean it
                try:
                    text = page.get_text()
                    cleaned_text = self.clean_extracted_text(text)
                except Exception as e:
                    print(f"Error extracting text from page {page_num + 1}: {e}")
                    traceback.print_exc()
                    cleaned_text = "[Error extracting text from this page]"
                text_content.append(f"Page {page_num + 1}: {cleaned_text}")
                
                # Extract images (simple approach)
                image_list = page.get_images()
                for img_index, img in enumerate(image_list):
                    try:
                        xref = img[0]
                        pix = fitz.Pixmap(doc, xref)
                        
                        if pix.n - pix.alpha < 4:  # GRAY or RGB
                            img_data = pix.tobytes("png")
                            img_pil = Image.open(io.BytesIO(img_data))
                            
                            # Check if image is not just blank/white space
                            if not self.is_blank_image(img_pil):
                                # Save image for display
                                img_filename = f"page_{page_num + 1}_image_{img_index + 1}.png"
                                img_path = os.path.join("images", img_filename)
                                img_pil.save(img_path)
                                
                                # Store image info for retrieval
                                img_key = f"page_{page_num + 1}_image_{img_index + 1}"
                                self.extracted_images[img_key] = {
                                    'path': img_path,
                                    'page': page_num + 1,
                                    'image_num': img_index + 1
                                }
                                
                                # Enhanced image analysis using Vision API
                                try:
                                    img_analysis = self.analyze_image_detailed(img_pil)
                                except Exception as e:
                                    print(f"Error analyzing image on page {page_num + 1}, image {img_index + 1}: {e}")
                                    traceback.print_exc()
                                    img_analysis = None
                                if img_analysis:
                                    # Store semantic description for search
                                    self.image_descriptions[img_key] = img_analysis
                                    image_content.append(f"Page {page_num + 1}, Image {img_index + 1}: {img_analysis}")
                                else:
                                    # If not a figure/graph, remove from extracted images
                                    if img_key in self.extracted_images:
                                        del self.extracted_images[img_key]
                                    # Delete the saved image file
                                    if os.path.exists(img_path):
                                        os.remove(img_path)
                                    continue
                        
                        pix = None
                    except Exception as e:
                        print(f"Error processing image on page {page_num + 1}, image {img_index + 1}: {e}")
                        traceback.print_exc()
                        continue
            
            doc.close()
            
            # Combine text and image content
            final_content = "\n".join(text_content)
            if image_content:
                final_content += "\n\nImage Analysis:\n" + "\n".join(image_content)
            
            print(f"PDF processing completed. Total content length: {len(final_content)}")
            return final_content
            
        except Exception as e:
            print(f"Error extracting text from PDF: {str(e)}")
            traceback.print_exc()
            return f"Error extracting text from PDF: {str(e)}"

    def analyze_image_detailed(self, img_pil):
        """Enhanced image analysis using Vision API for semantic search - focused on figures and graphs."""
        try:
            # Resize image to reduce API costs and speed
            max_size = 512
            img_pil.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            
            # Convert to base64
            buffer = io.BytesIO()
            img_pil.save(buffer, format='PNG')
            img_base64 = base64.b64encode(buffer.getvalue()).decode()
            
            # Enhanced Vision API call focused on figures and graphs
            headers = {
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": "gpt-4o",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Analyze this image and determine if it contains figures, graphs, charts, tables, or data visualizations. If it does, provide a detailed description including: 1) Type of visual (chart, graph, table, diagram, etc.) 2) Main subject/topic 3) Key data points or information shown 4) Any text or labels visible 5) Business context if applicable (revenue, growth, metrics, etc.). If this is NOT a figure/graph/chart (e.g., it's just text, a photo, or decorative element), respond with 'NOT_A_FIGURE'. Be specific and searchable."
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{img_base64}"
                                }
                            }
                        ]
                    }
                ],
                "max_tokens": 200
            }
            
            try:
                response = requests.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=15
                )
            except Exception as e:
                print(f"Error making Vision API request: {e}")
                traceback.print_exc()
                return None
            
            if response.status_code == 200:
                result = response.json()
                content = result["choices"][0]["message"]["content"]
                
                # Only return description if it's actually a figure/graph
                if "NOT_A_FIGURE" not in content.upper():
                    return content
                else:
                    return None
            else:
                print(f"Vision API returned status {response.status_code}: {response.text}")
                return None
                
        except Exception as e:
            print(f"Error analyzing image: {e}")
            return None

    def analyze_image_simple(self, img_pil):
        """Simple image analysis using Vision API - minimal and fast."""
        try:
            # Resize image to reduce API costs and speed
            max_size = 512
            img_pil.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            
            # Convert to base64
            buffer = io.BytesIO()
            img_pil.save(buffer, format='PNG')
            img_base64 = base64.b64encode(buffer.getvalue()).decode()
            
            # Simple Vision API call
            headers = {
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": "gpt-4o",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Describe this image briefly in one sentence. Focus on charts, tables, graphs, or key visual elements."
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{img_base64}"
                                }
                            }
                        ]
                    }
                ],
                "max_tokens": 100
            }
            
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"]
            else:
                return None
                
        except Exception as e:
            print(f"Error analyzing image: {e}")
            return None

    def search_images_semantically(self, query: str, top_k: int = 3):
        """Search for images based on semantic similarity to the query."""
        try:
            if not self.image_descriptions:
                return []
            
            # Use OpenAI embeddings to compare query with image descriptions
            headers = {
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json"
            }
            
            # Get query embedding
            query_payload = {
                "model": "text-embedding-3-small",
                "input": query
            }
            
            query_response = requests.post(
                "https://api.openai.com/v1/embeddings",
                headers=headers,
                json=query_payload,
                timeout=10
            )
            
            if query_response.status_code != 200:
                return []
            
            query_embedding = query_response.json()["data"][0]["embedding"]
            
            # Get embeddings for all image descriptions
            descriptions = list(self.image_descriptions.values())
            description_payload = {
                "model": "text-embedding-3-small",
                "input": descriptions
            }
            
            desc_response = requests.post(
                "https://api.openai.com/v1/embeddings",
                headers=headers,
                json=description_payload,
                timeout=15
            )
            
            if desc_response.status_code != 200:
                return []
            
            description_embeddings = [item["embedding"] for item in desc_response.json()["data"]]
            
            # Calculate cosine similarities
            similarities = []
            for i, desc_embedding in enumerate(description_embeddings):
                similarity = self.cosine_similarity(query_embedding, desc_embedding)
                similarities.append((similarity, i))
            
            # Sort by similarity and get top results
            similarities.sort(reverse=True)
            top_indices = [idx for _, idx in similarities[:top_k]]
            
            # Return matching images
            results = []
            img_keys = list(self.extracted_images.keys())
            for idx in top_indices:
                if idx < len(img_keys):
                    img_key = img_keys[idx]
                    img_info = self.extracted_images[img_key].copy()
                    img_info['description'] = self.image_descriptions[img_key]
                    img_info['similarity_score'] = similarities[idx][0]
                    results.append(img_info)
            
            return results
            
        except Exception as e:
            print(f"Error in semantic image search: {e}")
            return []

    def cosine_similarity(self, vec1, vec2):
        """Calculate cosine similarity between two vectors."""
        try:
            vec1 = np.array(vec1)
            vec2 = np.array(vec2)
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            return dot_product / (norm1 * norm2)
        except Exception as e:
            print(f"Error calculating cosine similarity: {e}")
            return 0.0

    def is_blank_image(self, img_pil):
        """Check if image is mostly blank/white space."""
        try:
            # Convert to grayscale for easier processing
            gray_img = img_pil.convert('L')
            
            # Get pixel data
            pixels = list(gray_img.getdata())
            
            # Count non-white pixels (assuming white is close to 255)
            white_threshold = 250  # Very close to white
            non_white_pixels = sum(1 for pixel in pixels if pixel < white_threshold)
            
            # Calculate percentage of non-white pixels
            total_pixels = len(pixels)
            non_white_percentage = (non_white_pixels / total_pixels) * 100
            
            # Consider image blank if less than 5% non-white pixels
            return non_white_percentage < 5
            
        except Exception as e:
            print(f"Error checking if image is blank: {e}")
            return False  # If we can't check, assume it's not blank

    def get_image_path(self, page_num, image_num=None):
        """Get the path to a specific image for display."""
        if image_num:
            key = f"page_{page_num}_image_{image_num}"
        else:
            # Find first image on the page
            key = None
            for k in self.extracted_images.keys():
                if k.startswith(f"page_{page_num}_"):
                    key = k
                    break
        
        if key and key in self.extracted_images:
            return self.extracted_images[key]['path']
        return None

    def get_all_images(self):
        """Get all extracted images info."""
        return self.extracted_images

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
        self.text_processor = TextProcessor()  # Use default behavior (analyze images during upload)

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
        self.chunk_size = 500
        self.chunk_overlap = 50
        
        # Simple error handling for model loading
        try:
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None

    def add_documents(self, documents: List[Dict]):
        """Add documents to the vector store."""
        if not documents:
            return
            
        if self.model is None:
            print("Error: No embedding model available")
            return
            
        print(f"Adding {len(documents)} documents to vector store...")
        
        # Extract text from documents
        texts = [doc['text'] for doc in documents]
        
        # Generate embeddings
        print("Converting text to vector embeddings...")
        try:
            embeddings = self.model.encode(texts, show_progress_bar=True)
        except Exception as e:
            print(f"Error generating embeddings: {e}")
            return
        
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
        if not self.documents or self.index is None or self.model is None:
            return []
        
        try:
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
        except Exception as e:
            print(f"Error during search: {e}")
            return []

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
        self.use_vision_api = use_vision_api

    def process_web_uploads(self, uploaded_files):
        """Process files uploaded through the web interface"""
        if not self.is_web:
            return False
            
        documents = self.file_handler.process_uploaded_files(uploaded_files)
        if documents:
            self.vector_store.add_documents(documents)
            return True
        return False

    def get_image_path(self, page_num, image_num=None):
        """Get image path for display - delegates to TextProcessor."""
        if hasattr(self.file_handler, 'text_processor'):
            return self.file_handler.text_processor.get_image_path(page_num, image_num)
        return None

    def get_all_images(self):
        """Get all extracted images info."""
        if hasattr(self.file_handler, 'text_processor'):
            return self.file_handler.text_processor.get_all_images()
        return {}

    def search_images_semantically(self, query: str, top_k: int = 3):
        """Search for images based on semantic similarity to the query."""
        if hasattr(self.file_handler, 'text_processor'):
            return self.file_handler.text_processor.search_images_semantically(query, top_k)
        return []

    def is_semantic_image_request(self, question: str):
        """Determine if the question is asking for semantic image search."""
        question_lower = question.lower()
        
        # Explicit image/visual keywords that must be present
        explicit_image_keywords = [
            'show me', 'find', 'where is', 'locate', 'display',
            'image', 'picture', 'chart', 'graph', 'table', 'figure',
            'diagram', 'visualization', 'plot', 'bar chart', 'line chart',
            'pie chart', 'scatter plot', 'histogram', 'dashboard'
        ]
        
        # Check if question explicitly asks for images/visuals
        is_asking_for_visual = any(keyword in question_lower for keyword in explicit_image_keywords)
        
        # Only proceed if explicitly asking for visual content
        return is_asking_for_visual

    def handle_semantic_image_search(self, question: str):
        """Handle semantic image search requests."""
        try:
            # Search for semantically similar images
            matching_images = self.search_images_semantically(question, top_k=3)
            
            if matching_images:
                # Filter out low similarity scores
                good_matches = [img for img in matching_images if img.get('similarity_score', 0) > 0.3]
                
                if good_matches:
                    return good_matches
                else:
                    return f"I found some images but they don't seem to match your request closely enough. Try being more specific about what you're looking for."
            else:
                return "I couldn't find any images matching your request. Try rephrasing or being more specific."
                
        except Exception as e:
            return f"Error searching for images: {str(e)}"

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
