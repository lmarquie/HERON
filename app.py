# Import necessary libraries
import os
import faiss
import fitz  # PyMuPDF
from typing import List, Dict
from datetime import datetime
from sentence_transformers import SentenceTransformer
import requests
import numpy as np
from PIL import Image
import io
import base64
import cv2
import pytesseract
from difflib import SequenceMatcher
from config import OPENAI_API_KEY

### =================== Text Processing =================== ###
class TextProcessor:
    def __init__(self, chunk_size: int = 1500, overlap: int = 30):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.extracted_images = {}  # Store images for display
        self.image_descriptions = {}  # Store semantic descriptions for search
        self.images_analyzed = set()  # Track which images have been analyzed

    def extract_text_from_pdf(self, pdf_path: str, enable_image_processing: bool = False) -> str:
        """Extract text and images from PDF, detect tables/graphs, and run OCR on each region (header + body)."""
        try:
            doc = fitz.open(pdf_path)
            text_content = []
            
            if enable_image_processing:
                os.makedirs("images", exist_ok=True)
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text = page.get_text()
                text_content.append(f"Page {page_num + 1}: {text}")

                # Only process images if enabled (much faster without this)
                if enable_image_processing:
                    # Render page as image with lower DPI for speed
                    pix = page.get_pixmap(dpi=150)  # Reduced from 200 to 150
                    img_data = pix.tobytes("png")
                    img_array = np.frombuffer(img_data, np.uint8)
                    
                    # Check image size to prevent processing extremely large images
                    if len(img_array) > 25 * 1024 * 1024:  # Reduced to 25MB limit
                        continue
                    
                    # Add error handling for large PNG chunks
                    try:
                        img_cv = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                        if img_cv is not None:
                            detected_regions = self.detect_tables_and_graphs(img_cv)
                            for idx, (x, y, w, h) in enumerate(detected_regions):
                                crop = img_cv[y:y+h, x:x+w]
                                img_filename = f"page_{page_num + 1}_detected_{idx + 1}.png"
                                img_path = os.path.join("images", img_filename)
                                cv2.imwrite(img_path, crop)
                                img_key = f"page_{page_num + 1}_detected_{idx + 1}"
                                # OCR on the cropped region (body)
                                ocr_text = self.ocr_image(crop)
                                # OCR on the header (top 20%)
                                header_h = max(1, int(h * 0.2))
                                header_crop = crop[0:header_h, :]
                                ocr_header = self.ocr_image(header_crop)
                                self.extracted_images[img_key] = {
                                    'path': img_path,
                                    'page': page_num + 1,
                                    'image_num': idx + 1,
                                    'description': 'Detected table/graph region (OpenCV)',
                                    'ocr_text': ocr_text,
                                    'ocr_header': ocr_header,
                                }
                    except Exception as e:
                        continue
                        
            doc.close()
            final_content = "\n".join(text_content)
            return final_content
        except Exception as e:
            return f"Error extracting text from PDF: {str(e)}"

    def detect_tables_and_graphs(self, img_cv):
        """Detect likely tables/graphs in a page image using OpenCV (returns list of bounding boxes)."""
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150, apertureSize=3)
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        regions = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            # Heuristic: Only keep large regions (likely tables/graphs)
            if w > 100 and h > 60 and w*h > 10000:
                regions.append((x, y, w, h))
        # Optionally, merge overlapping/close rectangles here
        return regions

    def ocr_image(self, img_cv):
        """Run OCR on a cropped image region using Tesseract if available, else fallback to empty string."""
        try:
            # Convert to RGB for pytesseract
            img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
            text = pytesseract.image_to_string(img_rgb)
            return text.strip()
        except Exception as e:
            return ""

    def ensure_images_analyzed(self):
        """Analyze all images that have not yet been analyzed with Vision API."""
        for img_key, img_info in list(self.extracted_images.items()):
            if img_key in self.images_analyzed:
                continue
            try:
                img_path = img_info['path']
                if os.path.exists(img_path):
                    img_pil = Image.open(img_path)
                    img_analysis = self.analyze_image_detailed(img_pil)
                    if img_analysis:
                        self.image_descriptions[img_key] = img_analysis
                    else:
                        # If not a figure/graph, remove from extracted images
                        del self.extracted_images[img_key]
                        if os.path.exists(img_path):
                            os.remove(img_path)
                self.images_analyzed.add(img_key)
            except Exception as e:
                continue

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
            
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=15
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result["choices"][0]["message"]["content"]
                
                # Only return description if it's actually a figure/graph
                if "NOT_A_FIGURE" not in content.upper():
                    return content
                else:
                    return None
            else:
                return None
                
        except Exception as e:
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
            return None

    def fuzzy_score(self, a, b):
        """Return a fuzzy match ratio between two strings (0-100)."""
        return int(SequenceMatcher(None, a, b).ratio() * 100)

    def search_images_semantically(self, query: str, top_k: int = 3, min_score: int = 180):
        """Find the most relevant detected region by fuzzy matching query to header/body OCR, prioritizing header. Only return if score is high enough."""
        query_lc = query.lower()
        best_score = -1
        best_img_info = None
        for img_info in self.extracted_images.values():
            ocr_header = img_info.get('ocr_header', '').lower()
            ocr_text = img_info.get('ocr_text', '').lower()
            header_score = self.fuzzy_score(query_lc, ocr_header)
            body_score = self.fuzzy_score(query_lc, ocr_text)
            combined_score = header_score * 2 + body_score  # Prioritize header
            if combined_score > best_score:
                best_score = combined_score
                best_img_info = img_info
        if best_img_info and best_score >= min_score:
            return [best_img_info]
        return []

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
                continue
        
        return documents

### =================== Document Loading =================== ###
class WebFileHandler:
    def __init__(self):
        self.text_processor = TextProcessor()
        self.saved_pdf_paths = []  # Track saved PDF paths for later image processing

    def process_uploaded_files(self, uploaded_files):
        """Process files uploaded through Streamlit."""
        documents = []
        self.saved_pdf_paths = []  # Reset for new uploads
        
        for uploaded_file in uploaded_files:
            try:
                # Save uploaded file temporarily
                temp_path = f"temp/{uploaded_file.name}"
                os.makedirs("temp", exist_ok=True)
                
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Save the path for later image processing
                self.saved_pdf_paths.append(temp_path)
                
                # Extract text from PDF (disable image processing for speed)
                text_content = self.text_processor.extract_text_from_pdf(temp_path, enable_image_processing=False)
                
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
                
            except Exception as e:
                continue
        
        return documents

    def get_saved_pdf_paths(self):
        """Get the paths of saved PDF files for image processing."""
        return self.saved_pdf_paths

### =================== Vector Store =================== ###
class VectorStore:
    def __init__(self, dimension: int = 768):
        self.documents = []
        self.embeddings = None
        self.index = None
        self.chunk_size = 500
        self.chunk_overlap = 50
        # Restore local embedding model
        try:
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            self.model = None

    def add_documents(self, documents: List[Dict]):
        """Add documents to the vector store using local SentenceTransformer."""
        if not documents:
            return
        if self.model is None:
            return
        texts = [doc['text'] for doc in documents]
        try:
            embeddings = self.model.encode(texts, show_progress_bar=True)
        except Exception as e:
            return
        self.documents.extend(documents)
        if self.embeddings is None:
            self.embeddings = embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, embeddings])
        embedding_dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(embedding_dim)
        self.index.add(self.embeddings.astype('float32'))

    def search(self, query: str, k: int = 3) -> List[Dict]:
        """Search for similar documents using local SentenceTransformer."""
        if not self.documents or self.index is None or self.model is None:
            return []
        try:
            query_embedding = self.model.encode([query])
            scores, indices = self.index.search(query_embedding.astype('float32'), k)
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
        self.file_handler = WebFileHandler()
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

    def search_images_semantically(self, query: str, top_k: int = 3, min_score: int = 180):
        """Find the most relevant detected region by fuzzy matching query to header/body OCR, prioritizing header. Only return if score is high enough."""
        query_lc = query.lower()
        best_score = -1
        best_img_info = None
        for img_info in self.file_handler.text_processor.extracted_images.values():
            ocr_header = img_info.get('ocr_header', '').lower()
            ocr_text = img_info.get('ocr_text', '').lower()
            header_score = self.file_handler.text_processor.fuzzy_score(query_lc, ocr_header)
            body_score = self.file_handler.text_processor.fuzzy_score(query_lc, ocr_text)
            combined_score = header_score * 2 + body_score  # Prioritize header
            if combined_score > best_score:
                best_score = combined_score
                best_img_info = img_info
        if best_img_info and best_score >= min_score:
            return [best_img_info]
        return []

    def is_semantic_image_request(self, question: str):
        """Determine if the question is asking for semantic image search."""
        question_lower = question.lower()
        
        # Keywords that suggest looking for figures, graphs, and charts
        semantic_keywords = [
            'revenue', 'profit', 'growth', 'sales', 'earnings', 'income',
            'chart', 'graph', 'table', 'data', 'metrics', 'performance',
            'financial', 'business', 'quarterly', 'annual', 'report',
            'trend', 'comparison', 'analysis', 'statistics', 'figures',
            'figure', 'diagram', 'visualization', 'plot', 'bar chart',
            'line chart', 'pie chart', 'scatter plot', 'histogram',
            'dashboard', 'kpi', 'key performance indicator'
        ]
        
        # Check if question contains semantic keywords
        has_semantic_keywords = any(keyword in question_lower for keyword in semantic_keywords)
        
        # Check if it's asking to show/find something specific
        is_asking_for_specific = any(word in question_lower for word in ['show me', 'find', 'where is', 'locate', 'display'])
        
        return has_semantic_keywords and is_asking_for_specific

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

    def process_images_on_demand(self, pdf_path: str):
        """Process images from a specific PDF only when needed."""
        try:
            # Process images with the text processor
            self.file_handler.text_processor.extract_text_from_pdf(pdf_path, enable_image_processing=True)
            return True
        except Exception as e:
            return False

    def is_image_related_question(self, question: str) -> bool:
        """Check if a question is related to images, charts, graphs, etc."""
        question_lower = question.lower()
        
        # Keywords that suggest image-related questions
        image_keywords = [
            'show', 'display', 'image', 'picture', 'chart', 'graph', 'table', 'figure',
            'diagram', 'visual', 'plot', 'graphic', 'photo', 'screenshot', 'illustration',
            'revenue chart', 'profit graph', 'sales figure', 'data visualization',
            'bar chart', 'line graph', 'pie chart', 'scatter plot', 'histogram'
        ]
        
        return any(keyword in question_lower for keyword in image_keywords)

    def get_pdf_path_for_question(self, question: str) -> str:
        """Get the PDF path that should be processed for an image question."""
        # Get saved PDF paths from the file handler
        if hasattr(self.file_handler, 'get_saved_pdf_paths'):
            saved_paths = self.file_handler.get_saved_pdf_paths()
            if saved_paths:
                # For now, return the first PDF
                # In a more sophisticated version, you could analyze which PDF is most relevant
                return saved_paths[0]
        return None

if __name__ == "__main__": 
    rag_system = RAGSystem()
    rag_system.run() 
