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
import openpyxl
import pandas as pd
import re
import tiktoken

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Reduce verbose httpx logging
logging.getLogger("httpx").setLevel(logging.WARNING)

def batch_documents_by_token_limit(documents, max_tokens=250000):
    enc = tiktoken.get_encoding("cl100k_base")
    batches = []
    current_batch = []
    current_tokens = 0
    for doc in documents:
        tokens = len(enc.encode(doc['text']))
        if tokens > max_tokens:
            print(f"Skipping chunk with {tokens} tokens (too large for a single batch)")
            continue  # Skip this chunk
        if current_tokens + tokens > max_tokens and current_batch:
            batches.append(current_batch)
            current_batch = []
            current_tokens = 0
        current_batch.append(doc)
        current_tokens += tokens
    if current_batch:
        batches.append(current_batch)
    return batches

### =================== Text Processing =================== ###
class TextProcessor:
    def __init__(self, chunk_size: int = 500, overlap: int = 30):
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
            
            # Disable image processing for now
            enable_image_processing = False
            
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
                "max_tokens": 16384
            }
            
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=100
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
                "max_tokens": 16384
            }
            
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=100
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
                # Extract text from PDF, but also get per-page text
                doc = fitz.open(pdf_path)
                for page_num in range(len(doc)):
                    page = doc.load_page(page_num)
                    text = page.get_text()
                    if text.strip():
                        # Chunk this page's text
                        chunks = self.chunk_text(text)
                        for i, chunk in enumerate(chunks):
                            if chunk.strip():
                                documents.append({
                                    'text': chunk.strip(),
                                    'metadata': {
                                        'source': pdf_path,
                                        'chunk_id': i,
                                        'total_chunks': len(chunks),
                                        'date': datetime.now().strftime('%Y-%m-%d'),
                                        'page': page_num + 1
                                    }
                                })
                doc.close()
            except Exception as e:
                logger.error(f"Error preparing document {pdf_path}: {str(e)}")
                continue
        return documents

    def extract_page_images(self, pdf_path: str) -> Dict[int, str]:
        """Extract images from each page of a PDF and save them for later display.
        This is a much simpler and more reliable approach than text highlighting.
        
        Returns:
            Dict mapping page numbers to image file paths
        """
        page_images = {}
        try:
            doc = fitz.open(pdf_path)
            os.makedirs("temp", exist_ok=True)
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                
                # Render page as image
                pix = page.get_pixmap(dpi=150)  # Lower DPI for speed
                img_path = f"temp/page_{page_num + 1}_full.png"
                pix.save(img_path)
                page_images[page_num + 1] = img_path
                
            doc.close()
            return page_images
            
        except Exception as e:
            logger.error(f"Error extracting page images from {pdf_path}: {str(e)}")
            return {}

### =================== Document Loading =================== ###
class WebFileHandler:
    def __init__(self):
        self.text_processor = TextProcessor()
        self.saved_pdf_paths = []  # Track saved PDF paths for later image processing
        self.processing_status = {}  # Track processing status per file
        self.executor = ThreadPoolExecutor(max_workers=3)  # Limit concurrent processing

    def process_uploaded_files(self, uploaded_files):
        """Process files uploaded through Streamlit with improved error handling."""
        documents = []
        self.saved_pdf_paths = []  # Reset for new uploads
        self.processing_status = {}
        
        # Process files in parallel for better performasence
        futures = []
        for uploaded_file in uploaded_files:
            future = self.executor.submit(self._process_single_file, uploaded_file)
            futures.append((uploaded_file.name, future))
        
        # Collect results
        for filename, future in futures:
            try:
                result = future.result(timeout=100)  # 60 second timeout per file
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
                # Process PDF page by page like prepare_documents does
                documents = []
                doc = fitz.open(temp_path)
                for page_num in range(len(doc)):
                    page = doc.load_page(page_num)
                    text = page.get_text()
                    if text.strip():
                        # Chunk this page's text
                        chunks = self.text_processor.chunk_text(text)
                        for i, chunk in enumerate(chunks):
                            if chunk.strip():
                                documents.append({
                                    'text': chunk.strip(),
                                    'metadata': {
                                        'source': uploaded_file.name,
                                        'chunk_id': i,
                                        'total_chunks': len(chunks),
                                        'date': datetime.now().strftime('%Y-%m-%d'),
                                        'page': page_num + 1
                                    }
                                })
                doc.close()
                return documents
            elif ext in [".doc", ".docx"]:
                text_content = extract_text_from_docx(temp_path)
            elif ext in [".ppt", ".pptx"]:
                text_content = extract_text_from_pptx(temp_path)
            elif ext == ".xlsx":
                text_content = extract_text_from_xlsx(temp_path)
            elif ext == ".xls":
                text_content = extract_text_from_xls(temp_path)
            else:
                logger.warning(f"Unsupported file type: {uploaded_file.name}")
                return None

            # For non-PDF files, use the old approach
            if text_content and text_content.strip():
                chunks = self.text_processor.chunk_text(text_content)
                documents = []
                for i, chunk in enumerate(chunks):
                    if chunk.strip():
                        # Extract page number from chunk text
                        page_num = None
                        page_match = re.search(r'Page (\d+):', chunk)
                        if page_match:
                            page_num = int(page_match.group(1))
                        
                        documents.append({
                            'text': chunk.strip(),
                            'metadata': {
                                'source': uploaded_file.name,
                                'chunk_id': i,
                                'total_chunks': len(chunks),
                                'date': datetime.now().strftime('%Y-%m-%d'),
                                'page': page_num
                            }
                        })
                return documents
            else:
                return None

        except Exception as e:
            logger.error(f"Error processing file {uploaded_file.name}: {str(e)}")
            return None

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
        if not documents:
            return

        if not self.initialized:
            logger.error("Model not initialized, cannot add documents")
            return

        try:
            # Batch documents to stay under the token limit
            batches = batch_documents_by_token_limit(documents, max_tokens=250000)
            print(f"Batching {len(documents)} documents into {len(batches)} batches")
            for i, batch in enumerate(batches):
                print(f"Batch {i+1} has {len(batch)} documents")
                for doc in batch:
                    tokens = len(tiktoken.get_encoding("cl100k_base").encode(doc['text']))
                    if tokens > 20000:
                        print(f"Large chunk: {tokens} tokens, first 100 chars: {doc['text'][:100]}")
                texts = [doc['text'] for doc in batch]
                embeddings = self.get_openai_embeddings_batch(texts)
                if embeddings is None or len(embeddings) != len(batch):
                    logger.error("Failed to get embeddings for all documents in batch")
                    continue

                self.documents.extend(batch)
                embeddings = np.array(embeddings)

                if self.embeddings is None:
                    self.embeddings = embeddings
                else:
                    self.embeddings = np.vstack([self.embeddings, embeddings])

                embedding_dim = self.embeddings.shape[1]
                if self.index is None:
                    self.index = faiss.IndexFlatIP(embedding_dim)
                self.index.add(embeddings.astype('float32'))

                logger.info(f"Added {len(batch)} documents to vector store (OpenAI embeddings - batch)")

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
                timeout=100
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
        if not normalize_length:
            return 1000

        question_lower = question.lower()

        # Short responses for simple questions
        if any(word in question_lower for word in ['what is', 'define', 'explain briefly', 'summarize']):
            return 1000

        # Medium responses for analysis questions
        if any(word in question_lower for word in ['analyze', 'compare', 'discuss', 'evaluate']):
            return 2000

        # Long responses for complex questions
        if any(word in question_lower for word in ['detailed', 'comprehensive', 'thorough']):
            return 4000

        # Default medium length
        return 2000

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
                # Store conversation history with chunk metadata from top result
                top_chunk = results[0] if results else None
                self.conversation_history.append({
                    'question': question,
                    'answer': answer,
                    'context': context,
                    'timestamp': datetime.now().isoformat(),
                    'source': top_chunk['metadata'].get('source') if top_chunk else None,
                    'chunk_id': top_chunk['metadata'].get('chunk_id') if top_chunk else None,
                    'total_chunks': top_chunk['metadata'].get('total_chunks') if top_chunk else None,
                    'chunk_text': top_chunk['text'] if top_chunk else None,
                    'page': top_chunk['metadata'].get('page') if top_chunk and 'page' in top_chunk['metadata'] else None
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
            # Store in conversation history with chunk metadata from top result
            top_chunk = results[0] if results else None
            self.conversation_history.append({
                'question': follow_up_question,
                'answer': answer,
                'context': full_context,
                'timestamp': datetime.now().isoformat(),
                'source': top_chunk['metadata'].get('source') if top_chunk else None,
                'chunk_id': top_chunk['metadata'].get('chunk_id') if top_chunk else None,
                'total_chunks': top_chunk['metadata'].get('total_chunks') if top_chunk else None,
                'chunk_text': top_chunk['text'] if top_chunk else None,
                'page': top_chunk['metadata'].get('page') if top_chunk and 'page' in top_chunk['metadata'] else None
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

    def add_to_conversation_history(self, question, answer, question_type="initial", mode="document", source=None, page=None, chunk_text=None):
        """Add to conversation history with mode tracking."""
        entry = {
            'question': question,
            'answer': answer,
            'question_type': question_type,
            'mode': mode,  # Track which mode was used
            'timestamp': datetime.now().isoformat()
        }
        
        # Add chunk metadata if provided (for source requests)
        if source:
            entry['source'] = source
        if page:
            entry['page'] = page
        if chunk_text:
            entry['chunk_text'] = chunk_text
            
        self.conversation_history.append(entry)

    def get_conversation_history(self):
        """Get the conversation history"""
        return self.conversation_history

    def clear_conversation_history(self):
        """Clear the conversation history"""
        self.conversation_history = []

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
        """Set internet mode on or off."""
        self.internet_mode = enabled

    def is_internet_mode_enabled(self) -> bool:
        """Check if internet mode is enabled."""
        return self.internet_mode

    def generate_internet_answer(self, question: str) -> str:
        """Generate answer using internet search."""
        try:
            # Use the ClaudeHandler with internet mode system prompt
            internet_llm = ClaudeHandler(
                system_prompt="You are a helpful assistant. Answer questions based on current internet knowledge. "
                "Always include URLs/sources for every fact or statement in your answer. "
                "If you use multiple sources, cite each one clearly with the URL."
            )
            
            # For now, return a simple response indicating internet mode
            return f"Internet mode is enabled. This would search the web for: {question}"
            
        except Exception as e:
            logger.error(f"Error generating internet answer: {str(e)}")
            return f"Error generating internet answer: {str(e)}"

    def process_question_with_mode(self, question: str, normalize_length: bool = True) -> str:
        """Process question using either document mode or internet mode."""
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

# --- Add helper functions for text extraction ---
def extract_text_from_docx(path):
    try:
        doc = DocxDocument(path)
        return '\n'.join([p.text for p in doc.paragraphs if p.text.strip()])
    except Exception as e:
        return f"Error extracting text from Word: {str(e)}"

def extract_text_from_pptx(path):
    try:
        prs = Presentation(path)
        text_runs = []
        for slide in prs.slides:
            for shape in slide.shapes:
                if hasattr(shape, "text"):
                    text_runs.append(shape.text)
        return '\n'.join([t for t in text_runs if t.strip()])
    except Exception as e:
        return f"Error extracting text from PowerPoint: {str(e)}"

def extract_text_from_xlsx(path):
    try:
        wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
        text = []
        for ws in wb.worksheets:
            for row in ws.iter_rows(values_only=True):
                row_text = ' '.join([str(cell) for cell in row if cell is not None])
                if row_text.strip():
                    text.append(row_text)
        return '\n'.join(text)
    except Exception as e:
        # Fallback to pandas for .xls or if openpyxl fails
        try:
            df = pd.read_excel(path, engine='openpyxl')
            return df.to_string(index=False)
        except Exception as e2:
            return f"Error extracting text from Excel: {str(e)}; {str(e2)}"

def extract_text_from_xls(path):
    try:
        df = pd.read_excel(path)
        return df.to_string(index=False)
    except Exception as e:
        return f"Error extracting text from Excel: {str(e)}"

def render_pdf_page_with_highlight(pdf_path, page_num, highlight_text=None):
    """Render a PDF page as an image, optionally highlighting the given text."""
    doc = fitz.open(pdf_path)
    page = doc.load_page(page_num - 1)  # 0-based index
    if highlight_text:
        text_instances = page.search_for(highlight_text)
        for inst in text_instances:
            page.add_highlight_annot(inst)
    pix = page.get_pixmap(dpi=200)
    img_path = f"temp/page_{page_num}_highlighted.png" if highlight_text else f"temp/page_{page_num}_screenshot.png"
    pix.save(img_path)
    doc.close()
    return img_path

def render_chunk_source_image(source_path, page_num, chunk_text, chunk_id=None, total_chunks=None):
    """Render a PDF page as an image, highlighting the chunk text if possible.
    
    Args:
        source_path: Path to the PDF file
        page_num: Page number (1-based)
        chunk_text: The text content of the chunk to highlight
        chunk_id: Optional chunk identifier within the page
        total_chunks: Optional total number of chunks on the page
    """
    import fitz
    import os
    os.makedirs("temp", exist_ok=True)
    
    try:
        doc = fitz.open(source_path)
        page = doc.load_page(page_num - 1)  # 0-based index
        
        # Try to highlight all instances of the chunk text
        if chunk_text:
            # Clean the chunk text for better matching
            clean_chunk_text = chunk_text.strip()
            if len(clean_chunk_text) > 50:  # Only highlight if chunk is substantial
                text_instances = page.search_for(clean_chunk_text)
                for inst in text_instances:
                    page.add_highlight_annot(inst)
            else:
                # For short chunks, try to highlight key phrases
                words = clean_chunk_text.split()
                if len(words) > 3:
                    # Highlight first few words as a fallback
                    key_phrase = " ".join(words[:3])
                    text_instances = page.search_for(key_phrase)
                    for inst in text_instances:
                        page.add_highlight_annot(inst)
        
        # Create filename with metadata
        filename_parts = [f"page_{page_num}"]
        if chunk_id is not None:
            filename_parts.append(f"chunk_{chunk_id}")
        if total_chunks is not None:
            filename_parts.append(f"of_{total_chunks}")
        filename_parts.append("highlighted.png")
        
        img_path = f"temp/_{'_'.join(filename_parts)}"
        
        pix = page.get_pixmap(dpi=200)
        pix.save(img_path)
        doc.close()
        return img_path
        
    except Exception as e:
        logger.error(f"Error rendering chunk source image: {str(e)}")
        # Fallback to basic rendering without highlighting
        try:
            doc = fitz.open(source_path)
            page = doc.load_page(page_num - 1)
            pix = page.get_pixmap(dpi=200)
            img_path = f"temp/page_{page_num}_fallback.png"
            pix.save(img_path)
            doc.close()
            return img_path
        except Exception as e2:
            logger.error(f"Fallback rendering also failed: {str(e2)}")
            return None

def get_page_image_simple(source_path, page_num):
    """Get a simple page image without highlighting - much more reliable.
    
    Args:
        source_path: Path to the PDF file
        page_num: Page number (1-based)
    
    Returns:
        Path to the page image file
    """
    import fitz
    import os
    os.makedirs("temp", exist_ok=True)
    
    try:
        doc = fitz.open(source_path)
        page = doc.load_page(page_num - 1)  # 0-based index
        
        # Simple page rendering
        pix = page.get_pixmap(dpi=150)
        img_path = f"temp/page_{page_num}_simple.png"
        pix.save(img_path)
        doc.close()
        return img_path
        
    except Exception as e:
        logger.error(f"Error getting page image: {str(e)}")
        return None

if __name__ == "__main__": 
    st.set_page_config(
        page_title="HERON",
        layout="wide"
    )
    rag_system = RAGSystem()
    rag_system.run() 
