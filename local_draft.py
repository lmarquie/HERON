# Import necessary libraries
import os
import faiss
import fitz  # PyMuPDF
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import requests
import numpy as np
from PIL import Image
import io
import base64
import cv2
import pytesseract
from difflib import SequenceMatcher
from config import OPENAI_API_KEY
import logging
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import openai
import pdfplumber
from abc import ABC, abstractmethod
from docx import Document as DocxDocument
from pptx import Presentation
import streamlit as st

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Reduce verbose httpx logging
logging.getLogger("httpx").setLevel(logging.WARNING)

### =================== Text Processing =================== ###
class TextProcessor:
    def __init__(self, chunk_size: int = 1500, overlap: int = 30):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.extracted_images = {}  # Store images for display
        self.image_descriptions = {}  # Store semantic descriptions for search
        self.images_analyzed = set()  # Track which images have been analyzed
        self.processing_lock = threading.Lock()  # Thread safety for image processing
        self.error_count = 0  # Track errors for better error handling

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
                    try:
                        # Render page as image with lower DPI for speed
                        pix = page.get_pixmap(dpi=150)  # Reduced from 200 to 150
                        img_data = pix.tobytes("png")
                        img_array = np.frombuffer(img_data, np.uint8)
                        
                        # Check image size to prevent processing extremely large images
                        if len(img_array) > 25 * 1024 * 1024:  # Reduced to 25MB limit
                            continue
                        
                        # Add error handling for large PNG chunks
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
                        logger.warning(f"Error processing images on page {page_num + 1}: {str(e)}")
                        continue
                        
            doc.close()
            final_content = "\n".join(text_content)
            return final_content
        except Exception as e:
            self.error_count += 1
            logger.error(f"Error extracting text from PDF: {str(e)}")
            return f"Error extracting text from PDF: {str(e)}"

    def detect_tables_and_graphs(self, img_cv):
        """Detect likely tables/graphs in a page image using OpenCV (returns list of bounding boxes)."""
        try:
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
        except Exception as e:
            logger.warning(f"Error detecting tables/graphs: {str(e)}")
            return []

    def ocr_image(self, img_cv):
        """Run OCR on a cropped image region using Tesseract if available, else fallback to empty string."""
        try:
            # Convert to RGB for pytesseract
            img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
            text = pytesseract.image_to_string(img_rgb)
            return text.strip()
        except Exception as e:
            logger.warning(f"OCR error: {str(e)}")
            return ""

    def ensure_images_analyzed(self):
        """Analyze all images that have not yet been analyzed with Vision API."""
        with self.processing_lock:
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
                    logger.warning(f"Error analyzing image {img_key}: {str(e)}")
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
                "max_tokens": 5000
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
            logger.warning(f"Vision API error: {str(e)}")
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
                "max_tokens": 5000
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
            logger.warning(f"Simple Vision API error: {str(e)}")
            return None

    def fuzzy_score(self, a, b):
        """Return a fuzzy match ratio between two strings (0-100)."""
        return int(SequenceMatcher(None, a, b).ratio() * 100)

    def search_images_semantically(self, query: str, top_k: int = 3, min_score: int = 180):
        """Find the most relevant detected region by fuzzy matching query to header/body OCR, prioritizing header. Only return if score is high enough."""
        try:
            query_lc = query.lower()
            
            # Check if this is a general image request (not specific)
            general_image_requests = ['image', 'picture', 'show me', 'display', 'give me', 'single image', 'any image']
            is_general_request = any(phrase in query_lc for phrase in general_image_requests)
            
            # If it's a general request, return any available images
            if is_general_request:
                all_images = list(self.extracted_images.values())
                if all_images:
                    return all_images[:top_k]  # Return first few images
                else:
                    return []
            
            # For specific requests, use semantic matching
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
        except Exception as e:
            logger.error(f"Error in semantic image search: {str(e)}")
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
                logger.error(f"Error preparing document {pdf_path}: {str(e)}")
                continue
        
        return documents

### =================== Document Loading =================== ###
class WebFileHandler:
    def __init__(self):
        self.text_processor = TextProcessor()
        self.saved_pdf_paths = []  # Track saved PDF paths for later image processing
        self.processing_status = {}  # Track processing status per file
        self.executor = ThreadPoolExecutor(max_workers=3)  # Limit concurrent processing
        self.images_processed = False  # Track if images have been processed

    def process_uploaded_files(self, uploaded_files):
        """Process files uploaded through Streamlit with improved error handling."""
        documents = []
        self.saved_pdf_paths = []  # Reset for new uploads
        self.processing_status = {}
        self.images_processed = False  # Reset image processing flag
        
        # Process files in parallel for better performance
        futures = []
        for uploaded_file in uploaded_files:
            future = self.executor.submit(self._process_single_file, uploaded_file)
            futures.append((uploaded_file.name, future))
        
        # Collect results
        for filename, future in futures:
            try:
                result = future.result(timeout=60)  # 60 second timeout per file
                if result:
                    documents.extend(result)
                    self.processing_status[filename] = "success"
                else:
                    self.processing_status[filename] = "failed"
            except Exception as e:
                logger.error(f"Error processing {filename}: {str(e)}")
                self.processing_status[filename] = "failed"
        
        return documents

    def _process_single_file(self, uploaded_file):
        """Process a single uploaded file."""
        try:
            ext = os.path.splitext(uploaded_file.name)[1].lower()
            temp_path = f"temp/{uploaded_file.name}"
            os.makedirs("temp", exist_ok=True)
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            self.saved_pdf_paths.append(temp_path)

            if ext == ".pdf":
                # Only extract text, not images (faster upload)
                text_content = self.text_processor.extract_text_from_pdf(temp_path, enable_image_processing=False)
            elif ext == ".docx":
                text_content = extract_text_from_docx(temp_path)
            elif ext in [".pptx"]:
                text_content = extract_text_from_pptx(temp_path)
            else:
                logger.warning(f"Unsupported file type: {uploaded_file.name}")
                return None

            if text_content and text_content.strip():
                chunks = self.text_processor.chunk_text(text_content)
                documents = []
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
                return documents
            else:
                return None

        except Exception as e:
            logger.error(f"Error processing file {uploaded_file.name}: {str(e)}")
            return None

    def process_images_on_demand(self):
        """Process images from all saved PDFs when requested."""
        if self.images_processed:
            return True  # Already processed
            
        try:
            for pdf_path in self.saved_pdf_paths:
                if os.path.exists(pdf_path):
                    # Process images with the text processor
                    self.text_processor.extract_text_from_pdf(pdf_path, enable_image_processing=True)
            
            self.images_processed = True
            return True
        except Exception as e:
            logger.error(f"Error processing images on demand: {str(e)}")
            return False

    def get_saved_pdf_paths(self):
        """Get the paths of saved PDF files for image processing."""
        return self.saved_pdf_paths

    def get_processing_status(self):
        """Get the processing status of uploaded files."""
        return self.processing_status

### =================== Vector Store =================== ###
class VectorStore:
    def __init__(self, dimension: int = 768):
        self.documents = []
        self.embeddings = None
        self.index = None
        self.chunk_size = 500
        self.chunk_overlap = 50
        self.model = None
        self.initialized = False
        
        # Initialize embedding model with retry logic
        self._initialize_model()

    def _initialize_model(self):
        """Initialize the embedding model with retry logic."""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Test the OpenAI API connection
                client = openai.OpenAI(api_key=OPENAI_API_KEY)
                # Test with a simple embedding
                response = client.embeddings.create(input="test", model="text-embedding-ada-002")
                self.initialized = True
                logger.info("OpenAI embedding model initialized successfully")
                break
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed to initialize OpenAI model: {str(e)}")
                if attempt == max_retries - 1:
                    logger.error("Failed to initialize OpenAI embedding model after all retries")
                    self.initialized = False
                else:
                    time.sleep(1)  # Wait before retry

    def get_openai_embedding_single(self, text, model="text-embedding-ada-002"):
        """Get a single embedding from OpenAI API."""
        try:
            client = openai.OpenAI(api_key=OPENAI_API_KEY)
            response = client.embeddings.create(input=text, model=model)
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error getting OpenAI embedding: {str(e)}")
            return None

    def get_openai_embeddings_batch(self, texts, model="text-embedding-ada-002"):
        """Get embeddings from OpenAI API in batches."""
        try:
            client = openai.OpenAI(api_key=OPENAI_API_KEY)
            response = client.embeddings.create(input=texts, model=model)
            return [data.embedding for data in response.data]
        except Exception as e:
            logger.error(f"Error getting OpenAI embeddings batch: {str(e)}")
            return None

    def add_documents(self, documents: List[Dict]):
        """Add documents to the vector store using OpenAI embeddings with batching."""
        if not documents:
            return
        
        if not self.initialized:
            logger.error("Model not initialized, cannot add documents")
            return
        
        try:
            texts = [doc['text'] for doc in documents]
            
            # Get all embeddings in a single batch call
            embeddings = self.get_openai_embeddings_batch(texts)
            
            if embeddings is None or len(embeddings) != len(documents):
                logger.error("Failed to get embeddings for all documents")
                return
            
            self.documents.extend(documents)
            embeddings = np.array(embeddings)
            
            if self.embeddings is None:
                self.embeddings = embeddings
            else:
                self.embeddings = np.vstack([self.embeddings, embeddings])
            
            embedding_dim = self.embeddings.shape[1]
            self.index = faiss.IndexFlatIP(embedding_dim)
            self.index.add(self.embeddings.astype('float32'))
            
            logger.info(f"Added {len(documents)} documents to vector store (OpenAI embeddings - batch)")
            
        except Exception as e:
            logger.error(f"Error adding documents to vector store: {str(e)}")

    def search(self, query: str, k: int = 3) -> List[Dict]:
        """Search for similar documents using OpenAI embeddings."""
        if not self.documents or self.index is None or not self.initialized:
            return []
        
        try:
            query_embedding = self.get_openai_embedding_single(query)
            if query_embedding is None:
                return []
            
            query_embedding = np.array([query_embedding])
            
            # Safety check: ensure we have valid embeddings
            if self.embeddings is None or len(self.embeddings) == 0:
                logger.warning("No embeddings available for search")
                return []
            
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
            logger.error(f"Error searching vector store: {str(e)}")
            return []

    def is_ready(self) -> bool:
        """Check if the vector store is properly initialized and ready for use."""
        return (
            self.initialized and
            self.documents is not None and 
            len(self.documents) > 0 and 
            self.embeddings is not None and 
            len(self.embeddings) > 0 and 
            self.index is not None
        )

### =================== Claude Handler =================== ###
class ClaudeHandler:
    def __init__(self, system_prompt=None):
        self.api_url = "https://api.openai.com/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json"
        }
        # Force citation in every answer
        self.system_prompt = (
            system_prompt or
            "You are a helpful assistant. Answer questions based on the provided context. "
            "Always cite the document source (filename and page or chunk) for every fact or statement in your answer. "
            "If you use multiple sources, cite each one clearly."
        )
        self.response_cache = {}  # Simple cache for repeated queries

    def generate_answer(self, question: str, context: str, normalize_length: bool = True) -> str:
        try:
            # Check cache first
            cache_key = f"{question[:100]}_{hash(context[:500])}"
            if cache_key in self.response_cache:
                return self.response_cache[cache_key]
            
            # Determine response length based on question type
            max_tokens = self._get_response_length(question, normalize_length)
            
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
                "max_tokens": max_tokens
            }

            response = requests.post(
                self.api_url,
                headers=self.headers,
                json=payload,
                timeout=30
            )

            response.raise_for_status()
            response_data = response.json()
            answer = response_data["choices"][0]["message"]["content"]
            
            # Cache the response
            self.response_cache[cache_key] = answer
            
            return answer

        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            return f"Error generating answer: {str(e)}"

    def _get_response_length(self, question: str, normalize_length: bool) -> int:
        """Determine appropriate response length based on question type."""
        if not normalize_length:
            return 1000
        
        question_lower = question.lower()
        
        # Short responses for simple questions
        if any(word in question_lower for word in ['what is', 'define', 'explain briefly', 'summarize']):
            return 300
        
        # Medium responses for analysis questions
        if any(word in question_lower for word in ['analyze', 'compare', 'discuss', 'evaluate']):
            return 600
        
        # Long responses for complex questions
        if any(word in question_lower for word in ['detailed', 'comprehensive', 'thorough']):
            return 1000
        
        # Default medium length
        return 500

### =================== Question Handler =================== ###
class QuestionHandler:
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store
        self.llm = ClaudeHandler()
        self.conversation_history = []
        self.error_count = 0
        self.max_retries = 3

    def process_question(self, question: str, query_type: str = "document", k: int = 5, normalize_length: bool = True) -> str:
        for attempt in range(self.max_retries):
            try:
                results = self.vector_store.search(question, k=k)
                
                if not results:
                    return "No relevant information found in the documents."
                
                # Build context with metadata for citation
                context = "\n".join([
                    f"[source: {chunk['metadata'].get('source', 'unknown')}, chunk: {chunk['metadata'].get('chunk_id', '?')}]\n{chunk['text']}"
                    for chunk in results
                ])
                answer = self.llm.generate_answer(question, context, normalize_length)
                
                # Store conversation history
                self.conversation_history.append({
                    'question': question,
                    'answer': answer,
                    'context': context,
                    'timestamp': datetime.now().isoformat()
                })
                
                return answer
                
            except Exception as e:
                self.error_count += 1
                logger.error(f"Error processing question (attempt {attempt + 1}): {str(e)}")
                if attempt == self.max_retries - 1:
                    return f"Error processing question after {self.max_retries} attempts: {str(e)}"
                time.sleep(1)  # Wait before retry

    def process_follow_up(self, follow_up_question: str, k: int = 5, normalize_length: bool = True) -> str:
        """Process a follow-up question using conversation history and document context."""
        if not self.conversation_history:
            return "No previous conversation to follow up on. Please ask a question first."
        
        try:
            # Get recent conversation context
            recent_context = ""
            for i, conv in enumerate(self.conversation_history[-3:]):  # Last 3 exchanges
                recent_context += f"Previous Q: {conv['question']}\nPrevious A: {conv['answer']}\n\n"
            
            # Get document context
            results = self.vector_store.search(follow_up_question, k=k)
            document_context = "\n".join([
                f"[source: {chunk['metadata'].get('source', 'unknown')}, chunk: {chunk['metadata'].get('chunk_id', '?')}]\n{chunk['text']}"
                for chunk in results
            ]) if results else ""
            
            # Combine contexts
            full_context = f"Conversation History:\n{recent_context}\nDocument Context:\n{document_context}"
            
            # Generate follow-up answer
            answer = self.llm.generate_answer(follow_up_question, full_context, normalize_length)
            
            # Store in conversation history
            self.conversation_history.append({
                'question': follow_up_question,
                'answer': answer,
                'context': full_context,
                'timestamp': datetime.now().isoformat()
            })
            
            return answer
            
        except Exception as e:
            self.error_count += 1
            logger.error(f"Error processing follow-up question: {str(e)}")
            return f"Error processing follow-up question: {str(e)}"

    def clear_conversation_history(self):
        """Clear the conversation history."""
        self.conversation_history = []

    def get_conversation_stats(self):
        """Get conversation statistics."""
        return {
            'total_questions': len(self.conversation_history),
            'error_count': self.error_count,
            'last_question_time': self.conversation_history[-1]['timestamp'] if self.conversation_history else None
        }

### =================== Session Manager =================== ###
class SessionManager:
    def __init__(self):
        self.sessions = {}
        self.session_timeout = 3600  # 1 hour timeout
        self.cleanup_interval = 300  # Clean up every 5 minutes
        self.last_cleanup = time.time()

    def create_session(self, session_id: str) -> Dict:
        """Create a new session."""
        self.sessions[session_id] = {
            'created_at': time.time(),
            'last_activity': time.time(),
            'rag_system': None,
            'documents_loaded': False,
            'conversation_history': []
        }
        return self.sessions[session_id]

    def get_session(self, session_id: str) -> Optional[Dict]:
        """Get an existing session."""
        self._cleanup_expired_sessions()
        
        if session_id in self.sessions:
            self.sessions[session_id]['last_activity'] = time.time()
            return self.sessions[session_id]
        return None

    def update_session(self, session_id: str, updates: Dict):
        """Update session data."""
        if session_id in self.sessions:
            self.sessions[session_id].update(updates)
            self.sessions[session_id]['last_activity'] = time.time()

    def delete_session(self, session_id: str):
        """Delete a session."""
        if session_id in self.sessions:
            del self.sessions[session_id]

    def _cleanup_expired_sessions(self):
        """Clean up expired sessions."""
        current_time = time.time()
        if current_time - self.last_cleanup > self.cleanup_interval:
            expired_sessions = []
            for session_id, session_data in self.sessions.items():
                if current_time - session_data['last_activity'] > self.session_timeout:
                    expired_sessions.append(session_id)
            
            for session_id in expired_sessions:
                del self.sessions[session_id]
            
            self.last_cleanup = current_time

### =================== Main RAG System =================== ###
class RAGSystem:
    def __init__(self, settings=None, is_web=False, use_vision_api=True, session_id: str = None, internet_mode=False):
        self.file_handler = WebFileHandler() if is_web else None
        self.vector_store = VectorStore()
        self.question_handler = QuestionHandler(self.vector_store)
        self.is_web = is_web
        self.conversation_history = []
        self.session_id = session_id
        self.session_manager = SessionManager()
        self.error_handling_enabled = True
        self.internet_mode = internet_mode  # New: Enable internet search mode
        self.performance_metrics = {
            'total_queries': 0,
            'avg_response_time': 0,
            'error_count': 0
        }

    def process_web_uploads(self, uploaded_files):
        """Process files uploaded through the web interface with improved error handling."""
        if not self.is_web:
            return False
            
        try:
            start_time = time.time()
            documents = self.file_handler.process_uploaded_files(uploaded_files)
            
            if documents:
                self.vector_store.add_documents(documents)
                
                # Update performance metrics
                processing_time = time.time() - start_time
                self.performance_metrics['total_queries'] += 1
                
                return True
            else:
                logger.warning("No documents were successfully processed")
                return False
                
        except Exception as e:
            logger.error(f"Error processing web uploads: {str(e)}")
            self.performance_metrics['error_count'] += 1
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
        try:
            query_lc = query.lower()
            
            # Check if this is a general image request (not specific)
            general_image_requests = ['image', 'picture', 'show me', 'display', 'give me', 'single image', 'any image']
            is_general_request = any(phrase in query_lc for phrase in general_image_requests)
            
            # If it's a general request, return any available images
            if is_general_request:
                all_images = list(self.file_handler.text_processor.extracted_images.values())
                if all_images:
                    return all_images[:top_k]  # Return first few images
                else:
                    return []
            
            # For specific requests, use semantic matching
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
        except Exception as e:
            logger.error(f"Error in semantic image search: {str(e)}")
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
        """Handle semantic image search requests with on-demand image processing."""
        try:
            # First, ensure images are processed if this is an image request
            if not hasattr(self.file_handler, 'images_processed') or not self.file_handler.images_processed:
                with st.spinner("Scanning document for images..."):
                    self.process_images_on_demand()
            
            # Get all available images first
            all_images = self.file_handler.text_processor.extracted_images
            
            if not all_images:
                return "No images were found in the uploaded documents."
            
            # Search for semantically similar images
            matching_images = self.search_images_semantically(question, top_k=3)
            
            if matching_images:
                return matching_images
            else:
                # If no semantic matches but we have images, return a general message
                return f"I found {len(all_images)} image(s) in the document, but none seem to match your specific request. Try asking for 'any image' or 'show me an image' to see what's available."
                
        except Exception as e:
            logger.error(f"Error in semantic image search: {str(e)}")
            return f"Error searching for images: {str(e)}"

    def add_to_conversation_history(self, question, answer, question_type="initial", mode="document"):
        """Add to conversation history with mode tracking."""
        self.conversation_history.append({
            'question': question,
            'answer': answer,
            'question_type': question_type,
            'mode': mode,  # Track which mode was used
            'timestamp': datetime.now().isoformat()
        })

    def get_conversation_history(self):
        """Get the conversation history"""
        return self.conversation_history

    def clear_conversation_history(self):
        """Clear the conversation history"""
        self.conversation_history = []

    def process_images_on_demand(self):
        """Process images from all saved PDFs when requested."""
        if hasattr(self.file_handler, 'process_images_on_demand'):
            return self.file_handler.process_images_on_demand()
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

    def get_performance_metrics(self):
        """Get performance metrics for monitoring."""
        return self.performance_metrics

    def reset_performance_metrics(self):
        """Reset performance metrics."""
        self.performance_metrics = {
            'total_queries': 0,
            'avg_response_time': 0,
            'error_count': 0
        }

    def set_internet_mode(self, enabled: bool):
        """Toggle internet mode on/off."""
        self.internet_mode = enabled
        logger.info(f"Internet mode {'enabled' if enabled else 'disabled'}")

    def is_internet_mode_enabled(self) -> bool:
        """Check if internet mode is enabled."""
        return self.internet_mode

    def generate_internet_answer(self, question: str) -> str:
        """Generate answer using internet search when no documents are available."""
        try:
            # Use OpenAI's web browsing capabilities
            client = openai.OpenAI(api_key=OPENAI_API_KEY)
            
            # Create a system prompt for internet search
            system_prompt = (
                "You are a helpful assistant with access to the internet. "
                "Answer questions based on current information from the web. "
                "You must always cite your sources with URLs in every answer. "
                "If you use multiple sources, cite each one clearly. "
                "If you cannot find relevant information, say so clearly."
            )
            
            response = client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question}
                ],
                temperature=0.3,
                max_tokens=5000
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error generating internet answer: {str(e)}")
            return f"Error accessing internet: {str(e)}"

    def process_question_with_mode(self, question: str, normalize_length: bool = True) -> str:
        """Process question using either document mode or internet mode."""
        # Check for image/graph requests first
        if self.is_image_related_question(question) or self.is_semantic_image_request(question):
            logger.info("Processing image/graph request")
            answer = self.handle_semantic_image_search(question)
            self.add_to_conversation_history(question, answer, "image_search")
            return answer
            
        if self.internet_mode:
            # Use internet mode
            logger.info("Processing question using internet mode")
            answer = self.generate_internet_answer(question)
            self.add_to_conversation_history(question, answer, "internet")
            return answer
        else:
            # Use document mode (existing logic)
            if not self.vector_store.is_ready():
                answer = "No documents loaded. Please upload documents first or enable internet mode."
                self.add_to_conversation_history(question, answer, "error", "document")
                return answer
            
            logger.info("Processing question using document mode")
            answer = self.question_handler.process_question(question, normalize_length=normalize_length)
            self.add_to_conversation_history(question, answer, "document")
            return answer

    def process_follow_up_with_mode(self, follow_up_question: str, normalize_length: bool = True) -> str:
        """Process follow-up question using either document mode or internet mode."""
        # Check for image/graph requests first
        if self.is_image_related_question(follow_up_question) or self.is_semantic_image_request(follow_up_question):
            logger.info("Processing follow-up image/graph request")
            answer = self.handle_semantic_image_search(follow_up_question)
            self.add_to_conversation_history(follow_up_question, answer, "image_search_followup")
            return answer
            
        if self.internet_mode:
            # Use internet mode for follow-up
            logger.info("Processing follow-up using internet mode")
            answer = self.generate_internet_answer(follow_up_question)
            self.add_to_conversation_history(follow_up_question, answer, "internet_followup")
            return answer
        else:
            # Use document mode (existing logic)
            if not self.vector_store.is_ready():
                answer = "No documents loaded. Please upload documents first or enable internet mode."
                self.add_to_conversation_history(follow_up_question, answer, "error", "document")
                return answer
            
            logger.info("Processing follow-up using document mode")
            answer = self.question_handler.process_follow_up(follow_up_question, normalize_length=normalize_length)
            self.add_to_conversation_history(follow_up_question, answer, "document_followup")
            return answer

    def get_mode_status(self) -> Dict:
        """Get current mode status and information."""
        return {
            'internet_mode': self.internet_mode,
            'documents_loaded': self.vector_store.is_ready(),
            'total_documents': len(self.vector_store.documents) if self.vector_store.documents else 0,
            'mode_description': 'Internet Search' if self.internet_mode else 'Document Search'
        }

    def handle_follow_up(self, follow_up_question: str, normalize_length: bool = True):
        """Encapsulate all follow-up logic: timing, error handling, metrics, and answer."""
        import time
        start_time = time.time()
        try:
            answer = self.process_follow_up_with_mode(follow_up_question, normalize_length=normalize_length)
            response_time = time.time() - start_time
            if not hasattr(self, 'performance_metrics'):
                self.performance_metrics = {}
            self.performance_metrics['last_response_time'] = response_time
            return answer
        except Exception as e:
            if hasattr(self, 'performance_metrics'):
                self.performance_metrics['error_count'] = self.performance_metrics.get('error_count', 0) + 1
            import logging
            logging.getLogger(__name__).error(f"Error generating follow-up: {str(e)}")
            return f"Error: {str(e)}"

if __name__ == "__main__": 
    rag_system = RAGSystem()
    rag_system.run() 
