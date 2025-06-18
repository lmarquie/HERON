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
import base64
from PIL import Image
import pytesseract
from pdf2image import convert_from_path

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
        self.image_analysis_cache = {}  # Cache for image analysis results

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF including images using OCR and vision analysis."""
        try:
            print(f"Processing PDF: {pdf_path}")
            # Extract regular text
            doc = fitz.open(pdf_path)
            text_content = []
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                
                # Extract regular text
                text = page.get_text()
                text_content.append(f"Page {page_num + 1} Text:\n{text}\n")
                
                # Try multiple methods to extract images
                images_found = 0
                
                # Method 1: get_images()
                image_list = page.get_images()
                print(f"Page {page_num + 1}: Found {len(image_list)} images via get_images()")
                
                if image_list:
                    text_content.append(f"Page {page_num + 1} Images:\n")
                    
                    for img_index, img in enumerate(image_list):
                        try:
                            print(f"Processing image {img_index + 1} on page {page_num + 1}")
                            # Get image data
                            xref = img[0]
                            pix = fitz.Pixmap(doc, xref)
                            
                            # Handle different image formats
                            if pix.n - pix.alpha < 4:  # GRAY or RGB
                                print(f"Image {img_index + 1}: RGB/Gray format")
                                # Convert to PIL Image
                                img_data = pix.tobytes("png")
                                pil_image = Image.open(io.BytesIO(img_data))
                                
                                # Analyze image with OCR and vision
                                image_analysis = self.analyze_image(pil_image, page_num + 1, img_index + 1)
                                text_content.append(image_analysis)
                                images_found += 1
                            elif pix.n == 4:  # CMYK
                                print(f"Image {img_index + 1}: CMYK format, converting to RGB")
                                # Convert CMYK to RGB
                                pix_rgb = fitz.Pixmap(fitz.csRGB, pix)
                                img_data = pix_rgb.tobytes("png")
                                pil_image = Image.open(io.BytesIO(img_data))
                                
                                # Analyze image with OCR and vision
                                image_analysis = self.analyze_image(pil_image, page_num + 1, img_index + 1)
                                text_content.append(image_analysis)
                                images_found += 1
                                
                                pix_rgb = None  # Free memory
                            else:
                                print(f"Image {img_index + 1}: Skipped (format: {pix.n} channels)")
                            
                            pix = None  # Free memory
                        except Exception as e:
                            print(f"Error processing image {img_index + 1}: {str(e)}")
                            text_content.append(f"Error processing image {img_index + 1}: {str(e)}\n")
                
                # Method 2: get_image_info()
                try:
                    image_blocks = page.get_image_info()
                    print(f"Page {page_num + 1}: Found {len(image_blocks)} images via get_image_info()")
                    if image_blocks:
                        if not image_list:  # Only add header if we didn't already
                            text_content.append(f"Page {page_num + 1} Images:\n")
                        
                        for block_index, block in enumerate(image_blocks):
                            try:
                                print(f"Processing image block {block_index + 1} on page {page_num + 1}")
                                # Try to extract the image - check if xref exists
                                if "xref" in block:
                                    pix = fitz.Pixmap(doc, block["xref"])
                                    if pix.n - pix.alpha < 4:  # GRAY or RGB
                                        img_data = pix.tobytes("png")
                                        pil_image = Image.open(io.BytesIO(img_data))
                                        
                                        # Analyze image
                                        image_analysis = self.analyze_image(pil_image, page_num + 1, f"block_{block_index + 1}")
                                        text_content.append(image_analysis)
                                        images_found += 1
                                    
                                    pix = None  # Free memory
                                else:
                                    print(f"Image block {block_index + 1}: No xref found in block")
                            except Exception as e:
                                print(f"Error processing image block {block_index + 1}: {str(e)}")
                                text_content.append(f"Error processing image block {block_index + 1}: {str(e)}\n")
                except Exception as e:
                    print(f"Error with get_image_info() on page {page_num + 1}: {str(e)}")
                
                # Method 3: get_drawings() - for vector graphics that might be charts
                try:
                    drawings = page.get_drawings()
                    print(f"Page {page_num + 1}: Found {len(drawings)} drawings")
                    if drawings and len(drawings) > 5:  # If there are many drawing elements, might be a chart
                        text_content.append(f"Page {page_num + 1} Vector Graphics:\n")
                        text_content.append(f"Vector graphics detected - possible chart or diagram with {len(drawings)} elements\n")
                except Exception as e:
                    print(f"Error with get_drawings() on page {page_num + 1}: {str(e)}")
                
                # Method 4: Convert page to image and analyze if no images found AND page has substantial content
                if images_found == 0:
                    # Only convert to image if page has substantial text content (likely to have charts/tables)
                    page_text = page.get_text().strip()
                    if len(page_text) > 100:  # Only process pages with substantial text
                        print(f"Page {page_num + 1}: No images found but substantial text, converting page to image for analysis")
                        try:
                            # Convert page to image
                            mat = fitz.Matrix(2, 2)  # 2x zoom for better quality
                            pix = page.get_pixmap(matrix=mat)
                            img_data = pix.tobytes("png")
                            pil_image = Image.open(io.BytesIO(img_data))
                            
                            # Save the page image
                            page_filename = f"page_{page_num + 1}_full.png"
                            page_path = os.path.join("app_data", "images", page_filename)
                            os.makedirs(os.path.dirname(page_path), exist_ok=True)
                            pil_image.save(page_path)
                            
                            # Add page image reference
                            text_content.append(f"Page {page_num + 1} Full Page Image:\n")
                            text_content.append(f"IMAGE_REF:{page_filename}\n")
                            
                            # Analyze the page image for charts/tables
                            page_analysis = self.analyze_image(pil_image, page_num + 1, "full_page")
                            text_content.append(page_analysis)
                            
                            pix = None  # Free memory
                        except Exception as e:
                            print(f"Error converting page {page_num + 1} to image: {str(e)}")
                    else:
                        print(f"Page {page_num + 1}: No images found and minimal text, skipping page conversion")
            
            doc.close()
            final_content = "\n".join(text_content)
            print(f"PDF processing completed. Total content length: {len(final_content)}")
            return final_content
            
        except Exception as e:
            print(f"Error extracting text from PDF: {str(e)}")
            return f"Error extracting text from PDF: {str(e)}"

    def analyze_image(self, image: Image.Image, page_num: int, img_index) -> str:
        """Analyze image using OCR and vision APIs for comprehensive understanding."""
        analysis_parts = []
        
        try:
            print(f"Starting analysis of image {img_index} on page {page_num}")
            # Convert img_index to string for filename
            img_index_str = str(img_index)
            
            # Create cache key based on image hash
            import hashlib
            img_hash = hashlib.md5(image.tobytes()).hexdigest()
            cache_key = f"{img_hash}_{page_num}_{img_index_str}"
            
            # Check cache first
            if cache_key in self.image_analysis_cache:
                print(f"Using cached analysis for image {img_index_str}")
                return self.image_analysis_cache[cache_key]
            
            # Save image for display in answers
            image_filename = f"image_page_{page_num}_img_{img_index_str}.png"
            image_path = os.path.join("app_data", "images", image_filename)
            os.makedirs(os.path.dirname(image_path), exist_ok=True)
            image.save(image_path)
            print(f"Image saved to: {image_path}")
            
            # Add image reference for display
            analysis_parts.append(f"IMAGE_REF:{image_filename}")
            print(f"Added image reference: {image_filename}")
            
            # 1. OCR Text Extraction (optional - only if Tesseract is available)
            try:
                # Check if tesseract is available
                import subprocess
                result = subprocess.run(['tesseract', '--version'], capture_output=True, text=True)
                if result.returncode == 0:
                    print("Running OCR...")
                    ocr_text = pytesseract.image_to_string(image)
                    if ocr_text.strip():
                        analysis_parts.append(f"OCR Text: {ocr_text.strip()}")
                        print(f"OCR found text: {len(ocr_text.strip())} characters")
                    else:
                        print("OCR found no text")
                else:
                    print("Tesseract not available, skipping OCR")
            except Exception as e:
                print(f"OCR not available: {str(e)}")
                # Don't add OCR error to analysis since it's optional
            
            # 2. Table Detection and Extraction (optional - only if Tesseract is available)
            try:
                # Check if tesseract is available
                import subprocess
                result = subprocess.run(['tesseract', '--version'], capture_output=True, text=True)
                if result.returncode == 0:
                    print("Running table detection...")
                    table_data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
                    # Process table data if detected
                    if any(float(conf) > 60 for conf in table_data['conf'] if conf != '-1'):
                        analysis_parts.append("Table detected - data extracted via OCR")
                        print("Table detected via OCR")
                    else:
                        print("No table detected via OCR")
                else:
                    print("Tesseract not available, skipping table detection")
            except Exception as e:
                print(f"Table detection not available: {str(e)}")
                # Don't add table detection error to analysis since it's optional
            
            # 3. Vision API Analysis (OpenAI GPT-4 Vision) - This is the main analysis
            # Check if Vision API should be skipped (can be set via environment variable)
            skip_vision_api = os.getenv('SKIP_VISION_API', 'false').lower() == 'true'
            if skip_vision_api:
                print("Vision API analysis skipped due to SKIP_VISION_API setting")
                analysis_parts.append("Vision Analysis: Skipped (disabled)")
            else:
                try:
                    print("Running vision API analysis...")
                    vision_analysis = self.analyze_image_with_vision_api(image)
                    if vision_analysis:
                        analysis_parts.append(f"Vision Analysis: {vision_analysis}")
                        print(f"Vision API analysis completed: {len(vision_analysis)} characters")
                    else:
                        print("Vision API analysis failed or returned no results")
                except Exception as e:
                    print(f"Vision API failed: {str(e)}")
                    # If vision API fails, continue without it
                    analysis_parts.append(f"Vision Analysis: [Vision API failed: {str(e)}]")
            
            # 4. Chart/Graph Detection (with error handling)
            try:
                print("Running chart detection...")
                chart_analysis = self.detect_charts_and_graphs(image)
                if chart_analysis:
                    analysis_parts.append(f"Chart Analysis: {chart_analysis}")
                    print(f"Chart analysis completed: {len(chart_analysis)} characters")
                else:
                    print("Chart detection found no charts")
            except Exception as e:
                print(f"Chart detection failed: {str(e)}")
                # If chart detection fails, continue without it
                pass
            
            if analysis_parts:
                result = f"Image {img_index_str} Analysis:\n" + "\n".join(analysis_parts) + "\n"
                print(f"Image analysis completed successfully. Result length: {len(result)}")
                
                # Cache the result
                self.image_analysis_cache[cache_key] = result
                print(f"Cached analysis for image {img_index_str}")
                
                return result
            else:
                result = f"Image {img_index_str}: No text or significant content detected\n"
                print("Image analysis completed but found no significant content")
                
                # Cache the result
                self.image_analysis_cache[cache_key] = result
                
                return result
                
        except Exception as e:
            error_msg = f"Image {img_index_str} Analysis Error: {str(e)}\n"
            print(f"Image analysis error: {str(e)}")
            return error_msg

    def analyze_image_with_vision_api(self, image: Image.Image) -> str:
        """Use OpenAI's GPT-4 Vision API for comprehensive image analysis."""
        try:
            print("Preparing image for Vision API...")
            
            # Validate and prepare image
            if image.mode not in ['RGB', 'RGBA']:
                image = image.convert('RGB')
                print("Converted image to RGB mode")
            
            # Resize image if it's too large (Vision API has size limits)
            max_size = 1024
            if image.width > max_size or image.height > max_size:
                image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
                print(f"Resized image to {image.width}x{image.height}")
            
            # Convert image to base64
            buffered = io.BytesIO()
            image.save(buffered, format="PNG", optimize=True)
            img_str = base64.b64encode(buffered.getvalue()).decode()
            print(f"Image converted to base64, size: {len(img_str)} characters")
            
            # Check if image data is reasonable size
            if len(img_str) > 20000000:  # ~20MB limit
                print("Warning: Image data is very large, may cause API issues")
            
            # Check if API key is available
            if not OPENAI_API_KEY:
                return "Vision API Error: No API key configured"
            
            # Use the newer OpenAI library approach with timeout
            try:
                from openai import OpenAI
                client = OpenAI(api_key=OPENAI_API_KEY, timeout=30.0)  # 30 second timeout
                
                print("Sending request to Vision API...")
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {
                            "role": "system", 
                            "content": "You are a financial document analysis expert. Analyze images comprehensively and provide detailed insights."
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": """Analyze this image from a financial document and provide a detailed report with:

**Observations:**
- Any text, numbers, dates, or data visible
- Chart types, graph elements, or table structures
- Visual elements like logos, signatures, or formatting

**Important Details:**
- Key financial metrics, trends, or data points
- Specific values, percentages, or figures
- Document structure and layout

**Analysis:**
- What type of financial content this represents
- Key insights or patterns identified
- Any notable findings or anomalies

Be specific about what you see and provide actionable insights."""
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/png;base64,{img_str}"
                                    }
                                }
                            ]
                        }
                    ],
                    temperature=0.1,
                    max_tokens=1000
                )
                
                content = response.choices[0].message.content
                print(f"Vision API response received: {len(content)} characters")
                
                # Check if response is generic
                if "unable to display or analyze images" in content.lower() or "cannot see" in content.lower():
                    print("Warning: Vision API returned generic response")
                    return "Vision API returned generic response - image may not have been processed correctly"
                
                return content
                
            except ImportError:
                # Fallback to requests if openai library not available
                print("OpenAI library not available, using requests fallback")
                return self._analyze_image_with_requests(image, img_str)
            except Exception as e:
                print(f"Vision API call failed: {str(e)}")
                return f"Vision API Error: {str(e)}"
                
        except Exception as e:
            error_msg = f"Vision API Error: {str(e)}"
            print(f"Vision API exception: {error_msg}")
            return error_msg
    
    def _analyze_image_with_requests(self, image: Image.Image, img_str: str) -> str:
        """Fallback method using requests library."""
        try:
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
                                "text": "Analyze this financial document image comprehensively and provide detailed insights about any text, charts, tables, or data visible."
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{img_str}"
                                }
                            }
                        ]
                    }
                ],
                "max_tokens": 1000,
                "temperature": 0.1
            }
            
            print("Sending request to Vision API via requests...")
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=30  # 30 second timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result["choices"][0]["message"]["content"]
                print(f"Vision API response received via requests: {len(content)} characters")
                return content
            else:
                error_msg = f"Vision API Error: {response.status_code} - {response.text}"
                print(f"Vision API error: {error_msg}")
                return error_msg
                
        except requests.exceptions.Timeout:
            error_msg = "Vision API Error: Request timed out after 30 seconds"
            print(error_msg)
            return error_msg
        except Exception as e:
            error_msg = f"Vision API Error: {str(e)}"
            print(f"Vision API exception: {error_msg}")
            return error_msg

    def detect_charts_and_graphs(self, image: Image.Image) -> str:
        """Detect and analyze charts, graphs, and visual data."""
        try:
            # Convert to grayscale for analysis
            gray_image = image.convert('L')
            width, height = image.size
            
            # Simple heuristics for chart detection
            chart_indicators = []
            
            # Check image size - charts are usually larger than small icons
            if width > 200 and height > 200:
                chart_indicators.append("large image size")
            
            # Check for color variations (common in graphs)
            if image.mode in ['RGB', 'RGBA']:
                # Get color statistics
                colors = image.getcolors(maxcolors=1000)
                if colors and len(colors) > 10:  # Multiple colors suggest charts
                    chart_indicators.append("multiple colors detected")
            
            # Basic pattern detection
            if chart_indicators:
                return f"Chart/Graph detected: {', '.join(chart_indicators)}"
            else:
                return "No specific chart patterns detected"
                
        except Exception as e:
            return f"Chart detection error: {str(e)}"

    def chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks."""
        if not text:
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
        
        return chunks

    def prepare_documents(self, pdf_paths: List[str]) -> List[Dict]:
        """Prepare documents for vector storage with enhanced image recognition."""
        documents = []
        
        for pdf_path in pdf_paths:
            try:
                # Extract text including image analysis
                text = self.extract_text_from_pdf(pdf_path)
                
                if text.strip():
                    # Split into chunks
                    chunks = self.chunk_text(text)
                    
                    for i, chunk in enumerate(chunks):
                        if chunk.strip():
                            documents.append({
                                'text': chunk,
                                'metadata': {
                                    'source': os.path.basename(pdf_path),
                                    'page': i + 1,
                                    'chunk': i + 1,
                                    'date': datetime.now().strftime('%Y-%m-%d'),
                                    'type': 'pdf_with_images'
                                }
                            })
                
            except Exception as e:
                print(f"Error processing {pdf_path}: {str(e)}")
        
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
        self.temp_dir = "temp"
        os.makedirs(self.temp_dir, exist_ok=True)

    def process_uploaded_files(self, uploaded_files):
        """Process files uploaded through Streamlit's file uploader using TextProcessor with image recognition"""
        if not uploaded_files:
            return []

        documents = []
        for uploaded_file in uploaded_files:
            try:
                # Save the uploaded file to a temporary location
                temp_path = os.path.join(self.temp_dir, uploaded_file.name)
                with open(temp_path, 'wb') as f:
                    f.write(uploaded_file.getbuffer())

                # Process the file based on its type using TextProcessor
                if uploaded_file.name.endswith('.pdf'):
                    # Use TextProcessor for PDF processing with image recognition
                    content = self.text_processor.extract_text_from_pdf(temp_path)
                elif uploaded_file.name.endswith('.txt'):
                    # For text files, use simple text extraction
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

### =================== Vector Store =================== ###
class VectorStore:
    def __init__(self, dimension: int = 768):
        self.dimension = dimension
        self.index = None
        self.documents = []
        
        # Use a smaller, faster model with local caching
        model_name = 'paraphrase-MiniLM-L3-v2'  # Smaller model than all-MiniLM-L6-v2
        cache_dir = os.path.join(os.path.expanduser('~'), '.cache', 'sentence_transformers')
        os.makedirs(cache_dir, exist_ok=True)
        
        try:
            # Try to load from cache first
            self.bi_encoder = SentenceTransformer(model_name, cache_folder=cache_dir)
            # Use a more powerful cross-encoder for better ranking
            self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', cache_folder=cache_dir)
        except Exception as e:
            print(f"Error loading models: {str(e)}")
            print("Falling back to smaller models...")
            # Fallback to even smaller models if the main ones fail
            self.bi_encoder = SentenceTransformer('paraphrase-MiniLM-L3-v2', cache_folder=cache_dir)
            self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-TinyBERT-L-2-v2', cache_folder=cache_dir)
        
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

    def add_documents(self, documents: List[Dict]):
        """Add documents to the vector store"""
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
        self.bi_encoder = SentenceTransformer('paraphrase-MiniLM-L3-v2')
        self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        self.bi_encoder.max_seq_length = 128
        self.cross_encoder.max_seq_length = 128
        
        if torch.cuda.is_available():
            self.bi_encoder = self.bi_encoder.to('cuda')
            self.cross_encoder = self.cross_encoder.to('cuda')
            self.bi_encoder = self.bi_encoder.half()
            self.cross_encoder = self.cross_encoder.half()
        
        print(f"Vector store loaded from {path}")

    def preprocess_query(self, query: str) -> str:
        """Preprocess the query to improve search relevance"""
        # Convert to lowercase
        query = query.lower()
        
        # Add financial context if query seems financial
        financial_terms = ['revenue', 'profit', 'earnings', 'income', 'financial', 'numbers', 'quarterly', 'annual', 'report']
        if any(term in query for term in financial_terms):
            query = f"financial report {query}"
        
        return query

    def search(self, query: str, k: int = 3) -> List[Dict]:
        """Search for relevant documents with improved ranking"""
        # Check cache first
        cache_key = f"{query}:{k}"
        if cache_key in self.similarity_cache:
            return self.similarity_cache[cache_key]

        # Preprocess query
        processed_query = self.preprocess_query(query)
        
        # Get query embedding
        query_embedding = self.get_embedding(processed_query)
        
        # Search in FAISS index
        if self.index is not None:
            # Use a larger initial search to get more candidates
            initial_k = min(k * 4, len(self.documents))  # Increased from k*2 to k*4
            scores, indices = self.index.search(query_embedding.reshape(1, -1).astype('float32'), initial_k)
            
            # Get documents and their scores
            candidates = []
            for score, idx in zip(scores[0], indices[0]):
                if idx != -1:  # FAISS returns -1 for empty slots
                    doc = self.documents[idx]
                    candidates.append({
                        'text': doc['text'],
                        'metadata': doc['metadata'],
                        'score': float(score)
                    })
            
            # Re-rank using cross-encoder
            if candidates:
                # Prepare pairs for cross-encoder
                pairs = [(processed_query, candidate['text']) for candidate in candidates]
                
                # Get cross-encoder scores
                cross_scores = self.cross_encoder.predict(pairs)
                
                # Update scores with cross-encoder scores
                for candidate, cross_score in zip(candidates, cross_scores):
                    # Combine bi-encoder and cross-encoder scores
                    candidate['score'] = 0.3 * candidate['score'] + 0.7 * float(cross_score)
            
            # Sort by combined score in descending order and take top k
            candidates.sort(key=lambda x: x['score'], reverse=True)
            results = candidates[:k]
            
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
    def __init__(self, system_prompt=None):
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
        self.system_prompt = system_prompt or "Answer based on the context."

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
                        "content": self.system_prompt
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
        financial_system_prompt = """You are a financial analysis expert. Your task is to analyze financial documents and provide detailed, accurate answers to questions about them.

Key capabilities:
- Analyze financial statements, reports, and documents
- Extract and interpret financial metrics and data
- Identify trends, risks, and opportunities
- Compare financial performance across periods
- Explain financial concepts and terminology
- Provide context for financial decisions
- Highlight important financial insights

Guidelines:
1. Base your answers primarily on the provided context
2. Be precise with numbers and financial data
3. Explain financial terms when used
4. Highlight any uncertainties or missing information
5. Provide relevant context for your analysis
6. Be clear about assumptions made
7. If the context doesn't contain enough information, say so clearly

Answer based on the context provided."""
        self.llm = ClaudeHandler(system_prompt=financial_system_prompt)

    def process_question(self, question: str, query_type: str = "document", k: int = 5) -> str:
        results = self.vector_store.search(question, k=k)

        # Combine results into a single context with size limits
        context_parts = []
        total_length = 0
        max_context_length = 40000  # Approximately 10k tokens
        
        for chunk in results:
            chunk_text = chunk['text']
            # If adding this chunk would exceed the limit, truncate it
            if total_length + len(chunk_text) > max_context_length:
                remaining_length = max_context_length - total_length
                if remaining_length > 100:  # Only add if we have at least 100 chars left
                    chunk_text = chunk_text[:remaining_length] + "..."
                else:
                    break
            context_parts.append(chunk_text)
            total_length += len(chunk_text)
            
            if total_length >= max_context_length:
                break
        
        context = "\n".join(context_parts)
        answer = self.llm.generate_answer(question, context)
        return answer

### =================== Main RAG System =================== ###
class RAGSystem:
    def __init__(self, settings=None, is_web=False, use_vision_api=True):
        self.file_handler = WebFileHandler() if is_web else LocalFileHandler()
        self.vector_store = VectorStore()
        self.question_handler = QuestionHandler(self.vector_store)
        self.running = True
        self.is_web = is_web
        self.use_vision_api = use_vision_api
        
        # Set environment variable for TextProcessor
        if not use_vision_api:
            os.environ['SKIP_VISION_API'] = 'true'
            print("Vision API disabled for this session")
        else:
            # Remove the environment variable if it exists
            os.environ.pop('SKIP_VISION_API', None)
            print("Vision API enabled for this session")
        
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
