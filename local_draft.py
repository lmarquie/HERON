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
from duckduckgo_search import DDGS
import speech_recognition as sr
from pydub import AudioSegment
import whisper
import tempfile
import subprocess
from pdf2image import convert_from_path
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.colors import black, blue
from reportlab.lib.enums import TA_LEFT, TA_CENTER

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Reduce verbose httpx logging
logging.getLogger("httpx").setLevel(logging.WARNING)

### =================== Text Processing =================== ###
class TextProcessor:
    def __init__(self, chunk_size: int = 2000, overlap: int = 100):
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
        """Split text into fixed-size chunks (no overlap, no sentence boundary logic)."""
        if not text.strip():
            return []
        chunk_size = self.chunk_size
        return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

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

### =================== Document Loading =================== ###
class AudioProcessor:
    def __init__(self):
        self.whisper_model = None
        self.recognizer = sr.Recognizer()
        self.ffmpeg_available = self._check_ffmpeg()
        self._model_cache = {}  # Cache for different model sizes
        
    def _check_ffmpeg(self):
        """Check if FFmpeg is available."""
        try:
            import subprocess
            subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False
    
    def load_whisper_model(self, model_size="tiny"):
        """Load Whisper model with caching - use tiny for speed."""
        if model_size in self._model_cache:
            self.whisper_model = self._model_cache[model_size]
            return
        
        try:
            logger.info(f"Loading Whisper model: {model_size}")
            # Use tiny model for maximum speed (still very accurate)
            self.whisper_model = whisper.load_model(model_size)
            self._model_cache[model_size] = self.whisper_model
            logger.info("Whisper model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading Whisper model: {str(e)}")
    
    def convert_audio_format(self, audio_path: str, target_format: str = "wav") -> str:
        """Convert audio to WAV format using FFmpeg with optimized settings."""
        try:
            if not self.ffmpeg_available:
                logger.error("FFmpeg not available for audio conversion")
                return audio_path
            
            # Create output path
            base_name = os.path.splitext(audio_path)[0]
            output_path = f"{base_name}.{target_format}"
            
            logger.info(f"Converting {audio_path} to {output_path}")
            
            # Use optimized FFmpeg settings for speed
            cmd = [
                'ffmpeg', '-i', audio_path,
                '-acodec', 'pcm_s16le',  # 16-bit PCM
                '-ar', '16000',          # 16kHz sample rate (Whisper's preferred)
                '-ac', '1',              # Mono
                '-y',                    # Overwrite output
                '-threads', '4',         # Use multiple threads
                '-preset', 'ultrafast',  # Fastest encoding preset
                output_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            if result.returncode != 0:
                logger.error(f"FFmpeg conversion failed: {result.stderr}")
                return audio_path
            
            logger.info(f"Successfully converted to {output_path}")
            return output_path
            
        except subprocess.TimeoutExpired:
            logger.error("FFmpeg conversion timed out")
            return audio_path
        except Exception as e:
            logger.error(f"Error converting audio format: {str(e)}")
            return audio_path
    
    def transcribe_with_whisper(self, audio_path: str) -> str:
        """Transcribe audio using Whisper with speed optimizations."""
        try:
            if self.whisper_model is None:
                self.load_whisper_model("tiny")  # Use tiny model for speed
            
            logger.info(f"Transcribing audio with Whisper: {audio_path}")
            
            # Check if file exists and has content
            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
            
            file_size = os.path.getsize(audio_path)
            if file_size == 0:
                raise ValueError(f"Audio file is empty: {audio_path}")
            
            # Get audio duration
            import librosa
            duration = librosa.get_duration(path=audio_path)
            logger.info(f"Audio duration: {duration:.2f} seconds ({duration/60:.2f} minutes)")
            
            print(f"🎵 Transcribing {duration/60:.1f} minute audio file...")
            
            # SPEED OPTIMIZED settings
            result = self.whisper_model.transcribe(
                audio_path,
                language="en",
                task="transcribe",
                verbose=False,  # Disable verbose for speed
                fp16=False,  # Force FP32 for stability
                condition_on_previous_text=False,  # Disable for speed
                temperature=0.0,  # Deterministic for speed
                compression_ratio_threshold=1.0,  # More lenient for speed
                logprob_threshold=-2.0,  # More lenient for speed
                no_speech_threshold=0.8,  # More lenient for speed
                word_timestamps=False,  # Disable for speed
                prepend_punctuations=False,  # Disable for speed
                append_punctuations=False  # Disable for speed
            )
            
            logger.info("Whisper transcription completed")
            
            transcription = result.get('text', '').strip()
            
            if not transcription:
                logger.warning("Whisper returned empty transcription")
                return "No speech detected in audio file."
            
            logger.info(f"Transcription completed: {len(transcription)} characters")
            print(f"🎉 Transcription completed! Total: {len(transcription)} characters")
            return transcription
            
        except Exception as e:
            # FIX: Capture the actual error details
            import traceback
            error_details = traceback.format_exc()
            logger.error(f"Error transcribing with Whisper: {str(e)}")
            logger.error(f"Full traceback: {error_details}")
            print(f"❌ Transcription error: {str(e)}")
            
            # If it's a tensor error, try with even more conservative settings
            if "tensor" in str(e).lower() and "reshape" in str(e).lower():
                print("🔄 Trying with ultra-conservative settings...")
                try:
                    result = self.whisper_model.transcribe(
                        audio_path,
                        language="en",
                        task="transcribe",
                        verbose=False,
                        fp16=False,
                        condition_on_previous_text=False,
                        temperature=0.0,
                        compression_ratio_threshold=1.0,
                        logprob_threshold=-2.0,
                        no_speech_threshold=0.8
                    )
                    transcription = result.get('text', '').strip()
                    if transcription:
                        return transcription
                except Exception as e2:
                    logger.error(f"Second attempt also failed: {str(e2)}")
            
            return f"Error transcribing audio: {str(e)}"
    
    def transcribe_with_speechrecognition(self, audio_path: str) -> str:
        """Transcribe audio using SpeechRecognition (Google Speech API)."""
        try:
            logger.info(f"Transcribing audio with SpeechRecognition: {audio_path}")
            
            with sr.AudioFile(audio_path) as source:
                audio = self.recognizer.record(source)
                transcription = self.recognizer.recognize_google(audio)
            
            logger.info(f"SpeechRecognition transcription completed: {len(transcription)} characters")
            return transcription
            
        except Exception as e:
            logger.error(f"Error transcribing with SpeechRecognition: {str(e)}")
            return f"Error transcribing audio: {str(e)}"
    
    def preprocess_audio_for_speed(self, audio_path: str) -> str:
        """Preprocess audio to speed up transcription."""
        try:
            import librosa
            import soundfile as sf
            
            # Load audio with lower sample rate for speed
            audio, sr = librosa.load(audio_path, sr=16000)  # 16kHz is sufficient for speech
            
            # Apply noise reduction and normalization
            # Simple normalization
            audio = librosa.util.normalize(audio)
            
            # Save preprocessed audio
            preprocessed_path = audio_path.replace('.', '_preprocessed.')
            sf.write(preprocessed_path, audio, sr)
            
            logger.info(f"Audio preprocessed and saved to: {preprocessed_path}")
            return preprocessed_path
            
        except Exception as e:
            logger.warning(f"Audio preprocessing failed: {str(e)}")
            return audio_path  # Return original if preprocessing fails

    def transcribe_audio(self, audio_path: str, method: str = "whisper") -> str:
        """Transcribe audio with preprocessing for speed."""
        try:
            if method == "whisper":
                # Preprocess audio for speed
                preprocessed_path = self.preprocess_audio_for_speed(audio_path)
                
                # Transcribe with Whisper
                transcription = self.transcribe_with_whisper(preprocessed_path)
                
                # Clean up preprocessed file
                if preprocessed_path != audio_path and os.path.exists(preprocessed_path):
                    try:
                        os.remove(preprocessed_path)
                    except:
                        pass
                
                return transcription
            else:
                return self.transcribe_with_speechrecognition(audio_path)
                
        except Exception as e:
            logger.error(f"Error transcribing audio: {str(e)}")
            return f"Error transcribing audio: {str(e)}"
    
    def create_transcript_file(self, transcription: str, original_filename: str) -> str:
        """Create a text file from transcription."""
        try:
            # Create temp directory
            temp_dir = "temp"
            os.makedirs(temp_dir, exist_ok=True)
            
            # Create filename
            base_name = os.path.splitext(original_filename)[0]
            transcript_path = os.path.join(temp_dir, f"{base_name}_transcript.txt")
            
            # Write transcription to file
            with open(transcript_path, 'w', encoding='utf-8') as f:
                f.write(transcription)
            
            logger.info(f"Transcript saved to: {transcript_path}")
            return transcript_path
            
        except Exception as e:
            logger.error(f"Error creating transcript file: {str(e)}")
            return None

    def _create_transcript_pdf(self, transcription: str, pdf_path: str, filename: str):
        """Create a PDF file from transcription using reportlab."""
        try:
            # Create the PDF document
            doc = SimpleDocTemplate(pdf_path, pagesize=letter)
            story = []
            
            # Get styles
            styles = getSampleStyleSheet()
            
            # Create custom styles
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=16,
                spaceAfter=30,
                alignment=TA_CENTER,
                textColor=blue
            )
            
            header_style = ParagraphStyle(
                'CustomHeader',
                parent=styles['Heading2'],
                fontSize=12,
                spaceAfter=20,
                textColor=black
            )
            
            body_style = ParagraphStyle(
                'CustomBody',
                parent=styles['Normal'],
                fontSize=11,
                spaceAfter=12,
                alignment=TA_LEFT,
                textColor=black
            )
            
            # Add title
            title = Paragraph(f"Audio Transcription: {filename}", title_style)
            story.append(title)
            story.append(Spacer(1, 20))
            
            # Add metadata
            metadata = f"""
            <b>File:</b> {filename}<br/>
            <b>Transcription Date:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br/>
            <b>Character Count:</b> {len(transcription):,}<br/>
            <b>Word Count:</b> {len(transcription.split()):,}
            """
            meta_para = Paragraph(metadata, header_style)
            story.append(meta_para)
            story.append(Spacer(1, 30))
            
            # Add transcription content
            # Split transcription into paragraphs for better formatting
            paragraphs = transcription.split('\n\n')
            
            for para in paragraphs:
                if para.strip():
                    # Clean up the paragraph
                    clean_para = para.strip().replace('\n', ' ')
                    if clean_para:
                        p = Paragraph(clean_para, body_style)
                        story.append(p)
                        story.append(Spacer(1, 12))
            
            # Build the PDF
            doc.build(story)
            logger.info(f"PDF transcript created: {pdf_path}")
            
        except Exception as e:
            logger.error(f"Error creating PDF transcript: {str(e)}")
            # If PDF creation fails, just continue with text file
            pass

    def _transcribe_long_audio_chunked(self, audio_path: str, duration: float) -> str:
        """Transcribe long audio files in parallel chunks."""
        try:
            import librosa
            import soundfile as sf
            from concurrent.futures import ThreadPoolExecutor, as_completed
            
            # Load audio
            audio, sr = librosa.load(audio_path, sr=16000)
            
            # Split into 5-minute chunks
            chunk_duration = 300  # 5 minutes in seconds
            chunk_samples = int(chunk_duration * sr)
            
            # Prepare chunks
            chunks = []
            for i in range(0, len(audio), chunk_samples):
                chunk = audio[i:i + chunk_samples]
                chunk_start = i / sr
                chunk_end = min((i + chunk_samples) / sr, duration)
                chunk_num = len(chunks) + 1
                
                # Save chunk to temporary file
                chunk_path = f"temp_chunk_{chunk_num}.wav"
                sf.write(chunk_path, chunk, sr, subtype='PCM_16')
                chunks.append((chunk_num, chunk_path, chunk_start, chunk_end))
            
            print(f" Processing {len(chunks)} chunks in parallel...")
            
            # Process chunks in parallel
            transcriptions = [None] * len(chunks)
            
            def transcribe_chunk(chunk_info):
                chunk_num, chunk_path, chunk_start, chunk_end = chunk_info
                try:
                    print(f" Processing chunk {chunk_num}: {chunk_start/60:.1f}m - {chunk_end/60:.1f}m")
                    
                    result = self.whisper_model.transcribe(
                        chunk_path,
                        language="en",
                        task="transcribe",
                        fp16=False
                    )
                    
                    chunk_text = result.get('text', '').strip()
                    
                    # Clean up temporary file
                    os.remove(chunk_path)
                    
                    if chunk_text:
                        print(f"✅ Chunk {chunk_num} completed: {len(chunk_text)} characters")
                        return chunk_num - 1, chunk_text
                    else:
                        print(f"⚠️ Chunk {chunk_num} returned no text")
                        return chunk_num - 1, ""
                    
                except Exception as e:
                    print(f"❌ Error transcribing chunk {chunk_num}: {str(e)}")
                    if os.path.exists(chunk_path):
                        os.remove(chunk_path)
                    return chunk_num - 1, ""
            
            # Use ThreadPoolExecutor for parallel processing
            with ThreadPoolExecutor(max_workers=4) as executor:
                future_to_chunk = {executor.submit(transcribe_chunk, chunk): chunk for chunk in chunks}
                
                for future in as_completed(future_to_chunk):
                    chunk_idx, text = future.result()
                    if text:
                        transcriptions[chunk_idx] = text
            
            # Combine transcriptions in order
            final_transcription = " ".join([t for t in transcriptions if t])
            
            if final_transcription:
                print(f" Full transcription completed! Total: {len(final_transcription)} characters")
                return final_transcription
            else:
                print("❌ No transcription completed")
                return "No speech detected in audio file."
            
        except Exception as e:
            logger.error(f"Error in parallel chunked transcription: {str(e)}")
            return f"Error transcribing long audio: {str(e)}"

    def transcribe_long_audio_parallel(self, audio_path: str, chunk_duration: int = 300) -> str:
        """Transcribe long audio files using parallel processing for speed."""
        try:
            import librosa
            import soundfile as sf
            from concurrent.futures import ThreadPoolExecutor, as_completed
            
            # Load audio
            audio, sr = librosa.load(audio_path, sr=16000)
            duration = len(audio) / sr
            
            logger.info(f"Processing {duration/60:.1f} minute audio in parallel chunks")
            
            # Split into chunks (5 minutes each)
            chunk_samples = int(chunk_duration * sr)
            
            # Prepare chunks
            chunks = []
            for i in range(0, len(audio), chunk_samples):
                chunk = audio[i:i + chunk_samples]
                chunk_start = i / sr
                chunk_end = min((i + chunk_samples) / sr, duration)
                chunk_num = len(chunks) + 1
                
                # Save chunk to temporary file
                chunk_path = f"temp_chunk_{chunk_num}.wav"
                sf.write(chunk_path, chunk, sr, subtype='PCM_16')
                chunks.append((chunk_num, chunk_path, chunk_start, chunk_end))
            
            print(f" Processing {len(chunks)} chunks in parallel...")
            
            # Process chunks in parallel
            transcriptions = [None] * len(chunks)
            
            def transcribe_chunk(chunk_info):
                chunk_num, chunk_path, chunk_start, chunk_end = chunk_info
                try:
                    print(f" Processing chunk {chunk_num}: {chunk_start/60:.1f}m - {chunk_end/60:.1f}m")
                    
                    # Use tiny model for speed
                    if self.whisper_model is None:
                        self.load_whisper_model("tiny")
                    
                    result = self.whisper_model.transcribe(
                        chunk_path,
                        language="en",
                        task="transcribe",
                        verbose=False,
                        fp16=False,
                        condition_on_previous_text=False,
                        temperature=0.0,
                        compression_ratio_threshold=1.0,
                        logprob_threshold=-2.0,
                        no_speech_threshold=0.8
                    )
                    
                    chunk_text = result.get('text', '').strip()
                    
                    # Clean up temporary file
                    os.remove(chunk_path)
                    
                    if chunk_text:
                        print(f"✅ Chunk {chunk_num} completed: {len(chunk_text)} characters")
                        return chunk_num - 1, chunk_text
                    else:
                        print(f"⚠️ Chunk {chunk_num} returned no text")
                        return chunk_num - 1, ""
                    
                except Exception as e:
                    print(f"❌ Error transcribing chunk {chunk_num}: {str(e)}")
                    if os.path.exists(chunk_path):
                        os.remove(chunk_path)
                    return chunk_num - 1, ""
            
            # Use ThreadPoolExecutor for parallel processing
            with ThreadPoolExecutor(max_workers=4) as executor:
                future_to_chunk = {executor.submit(transcribe_chunk, chunk): chunk for chunk in chunks}
                
                for future in as_completed(future_to_chunk):
                    chunk_idx, text = future.result()
                    if text:
                        transcriptions[chunk_idx] = text
            
            # Combine transcriptions in order
            final_transcription = " ".join([t for t in transcriptions if t])
            
            logger.info(f"Parallel transcription completed: {len(final_transcription)} characters")
            return final_transcription
            
        except Exception as e:
            logger.error(f"Error in parallel transcription: {str(e)}")
            return f"Error in parallel transcription: {str(e)}"

class WebFileHandler:
    def __init__(self):
        self.text_processor = TextProcessor()
        self.audio_processor = AudioProcessor()
        self.saved_pdf_paths = []
        self.processing_status = {}
        self.is_processing = False

    def _process_single_file(self, uploaded_file):
        # Prevent double processing
        if self.is_processing:
            logger.info(f"Already processing a file, skipping {uploaded_file.name}")
            return None
            
        self.is_processing = True
        
        try:
            # Check if this audio file has already been processed
            if uploaded_file.name in self.processing_status:
                logger.info(f"Audio file {uploaded_file.name} already processed, skipping")
                self.is_processing = False
                return None
                
            logger.info(f"Processing file: {uploaded_file.name}")
            ext = os.path.splitext(uploaded_file.name)[1].lower()
            temp_path = f"temp/{uploaded_file.name}"
            os.makedirs("temp", exist_ok=True)
            
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Handle different file types
            if ext in ['.pdf', '.docx', '.pptx']:
                # Existing document processing
                if ext == ".pdf":
                    text_content = self.text_processor.extract_text_from_pdf(temp_path, enable_image_processing=True)
                elif ext == ".docx":
                    text_content = self.text_processor.extract_text_from_docx(temp_path)
                elif ext in [".pptx"]:
                    text_content = self.text_processor.extract_text_from_pptx(temp_path)
                
                self.saved_pdf_paths.append(temp_path)
                
            elif ext in ['.mp3', '.wav', '.m4a', '.flac', '.ogg', '.aac']:
                # Audio file processing with speed optimization
                logger.info(f"Processing audio file: {uploaded_file.name}")
                
                # Check file size to decide on processing method
                file_size = os.path.getsize(temp_path)
                file_size_mb = file_size / (1024 * 1024)
                
                if file_size_mb > 50:  # Large files (>50MB)
                    logger.info("Large audio file detected, using parallel processing")
                    transcription = self.audio_processor.transcribe_long_audio_parallel(temp_path)
                else:
                    # Smaller files use regular processing
                    transcription = self.audio_processor.transcribe_audio(temp_path, method="whisper")
                
                logger.info(f"Transcription result: {len(transcription)} characters")
                
                # Check if transcription failed
                if transcription.startswith("Error"):
                    logger.error(f"Transcription failed: {transcription}")
                    self.is_processing = False
                    return None
                
                # Create transcript file
                transcript_path = self.audio_processor.create_transcript_file(transcription, uploaded_file.name)
                
                # Process transcription as text content
                text_content = transcription
                self.processing_status[uploaded_file.name] = {
                    'status': 'completed',
                    'transcript_path': transcript_path,
                    'characters': len(transcription)
                }

            else:
                logger.warning(f"Unsupported file type: {uploaded_file.name}")
                self.is_processing = False
                return None

            if text_content and text_content.strip():
                logger.info(f"Processing text content: {len(text_content)} characters")
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
                                'date': datetime.now().strftime('%Y-%m-%d'),
                                'file_type': 'audio' if ext in ['.mp3', '.wav', '.m4a', '.flac', '.ogg', '.aac'] else 'document'
                            }
                        })
                logger.info(f"Successfully processed {uploaded_file.name}: {len(documents)} chunks")
                self.is_processing = False
                return documents
            else:
                logger.warning(f"No text content extracted from {uploaded_file.name}")
                self.is_processing = False
                return None

        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            logger.error(f"Error processing file {uploaded_file.name}: {str(e)}")
            logger.error(f"Full traceback: {error_details}")
            self.is_processing = False
            return None

    def get_saved_pdf_paths(self):
        """Get the paths of saved PDF files for image processing."""
        return self.saved_pdf_paths

    def get_processing_status(self):
        """Get the processing status of uploaded files."""
        return self.processing_status

### =================== Vector Store =================== ###
class VectorStore:
    _model = None  # Class variable to store the model
    
    def __init__(self, dimension: int = 768):
        self.documents = []
        self.embeddings = None
        self.index = None
        self.chunk_size = 2000
        self.chunk_overlap = 100
        self.initialized = False
        
        # Initialize embedding model only once
        self._initialize_model()

    def _initialize_model(self):
        """Initialize the embedding model only once."""
        if VectorStore._model is None:  # Only load if not already loaded
            try:
                from sentence_transformers import SentenceTransformer
                import torch
                
                # Force CPU usage and handle meta tensor issue
                logger.info("Loading sentence transformers model...")
                
                # Set environment variables to force CPU usage
                os.environ['CUDA_VISIBLE_DEVICES'] = ''
                
                # Try different approaches to handle the device issue
                try:
                    # Method 1: Load with explicit CPU device and trust_remote_code
                    VectorStore._model = SentenceTransformer(
                        'all-MiniLM-L6-v2', 
                        device='cpu',
                        trust_remote_code=True
                    )
                    logger.info("Method 1 successful: Loaded with CPU device")
                    
                except Exception as e1:
                    logger.warning(f"Method 1 failed: {e1}")
                    try:
                        # Method 2: Load normally then move to CPU with to_empty
                        VectorStore._model = SentenceTransformer('all-MiniLM-L6-v2')
                        if hasattr(VectorStore._model, 'to_empty'):
                            VectorStore._model.to_empty(device='cpu')
                        else:
                            VectorStore._model.to('cpu')
                        logger.info("Method 2 successful: Used to_empty")
                        
                    except Exception as e2:
                        logger.warning(f"Method 2 failed: {e2}")
                        try:
                            # Method 3: Load with specific torch settings
                            torch.set_default_device('cpu')
                            VectorStore._model = SentenceTransformer('all-MiniLM-L6-v2')
                            VectorStore._model.to('cpu')
                            logger.info("Method 3 successful: Used torch default device")
                            
                        except Exception as e3:
                            logger.warning(f"Method 3 failed: {e3}")
                            # Method 4: Last resort - try with different model
                            try:
                                VectorStore._model = SentenceTransformer('paraphrase-MiniLM-L3-v2')
                                VectorStore._model.to('cpu')
                                logger.info("Method 4 successful: Used alternative model")
                            except Exception as e4:
                                logger.error(f"All methods failed. Last error: {e4}")
                                self.initialized = False
                                return
                
                # Verify the model is working
                try:
                    test_embedding = VectorStore._model.encode(['test'], convert_to_tensor=False)
                    logger.info(f"Model test successful. Embedding shape: {test_embedding.shape}")
                except Exception as e:
                    logger.error(f"Model test failed: {e}")
                    self.initialized = False
                    return
                
                logger.info("Hugging Face embedding model initialized successfully")
                
            except Exception as e:
                logger.error(f"Failed to initialize Hugging Face model: {str(e)}")
                self.initialized = False
                return
        
        self.model = VectorStore._model  # Use the shared model
        self.initialized = True

    def get_embeddings_batch(self, texts):
        """Get embeddings from Hugging Face model in batches."""
        try:
            if not self.initialized or self.model is None:
                logger.error("Model not initialized")
                return None

            # Get embeddings for all texts at once
            embeddings = self.model.encode(texts, convert_to_tensor=False)
            return embeddings.tolist() if hasattr(embeddings, 'tolist') else embeddings
            
        except Exception as e:
            logger.error(f"Error getting Hugging Face embeddings: {str(e)}")
            return None

    def add_documents(self, documents: List[Dict]):
        """Add documents to the vector store using Hugging Face embeddings."""
        if not documents:
            return
        
        if not self.initialized:
            logger.error("Model not initialized, cannot add documents")
            logger.error("This usually means the sentence transformers model failed to load")
            logger.error("Check the logs above for Hugging Face model initialization errors")
            return
        
        try:
            texts = [doc['text'] for doc in documents]
            
            # Get embeddings for all documents
            embeddings = self.get_embeddings_batch(texts)
            
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
            
            logger.info(f"Added {len(documents)} documents to vector store (Hugging Face embeddings)")
            
        except Exception as e:
            logger.error(f"Error adding documents to vector store: {str(e)}")

    def search(self, query: str, k: int = 3) -> List[Dict]:
        """Search for similar documents using Hugging Face embeddings."""
        if not self.documents or self.index is None or not self.initialized:
            return []
        
        try:
            # Get query embedding
            query_embedding = self.model.encode([query], convert_to_tensor=False)
            query_embedding = query_embedding.reshape(1, -1)
            
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
        # Enhanced prompt that prioritizes specific, detailed information
        self.system_prompt = (
            system_prompt or
            "You are an expert financial consultant and analyst with 20+ years of experience in investment banking, "
            "equity research, and financial modeling. You specialize in deep-dive financial analysis and comprehensive reporting. "
            
            "CRITICAL: When asked about financial data, revenue, projections, or specific figures, ALWAYS lead with the most "
            "specific and detailed numerical information available. Do NOT start with vague summaries or general statements. "
            "If the context contains specific numbers, dates, or financial figures, present those FIRST. Only provide general "
            "commentary AFTER presenting the specific data. "

            "You must provide comprehensive, detailed analysis in paragraph form. Never give one-sentence answers. "
            "Always provide thorough analysis with multiple data points, historical context, year-over-year comparisons, "
            "trend analysis, risk factors, market conditions, competitive analysis, and investment implications. "
            "When projections are available, explain the assumptions and growth drivers. Compare with industry benchmarks, "
            "competitors, or historical performance. Provide insights on what the data means for investors and stakeholders. "
            
            "Structure your responses with clear paragraphs that flow logically: start with an executive summary of key findings, "
            "present detailed financial metrics with exact numbers, provide historical context and trends, include comparative "
            "analysis, discuss implications and outlook, and address potential risks and opportunities. "
            
            "Always cite the document source (filename and page or chunk) for every fact or statement in your answer. "
            "If you use multiple sources, cite each one clearly. Include page numbers when available. "
            
            "Remember: You are a senior financial analyst. Your clients expect comprehensive, detailed analysis that provides "
            "actionable insights. Never provide superficial or vague responses."
        )
        self.response_cache = {}  # Simple cache for repeated queries

    def generate_answer(self, question: str, context: str, normalize_length: bool = True) -> str:
        try:
            # Check cache first
            cache_key = f"{question[:100]}_{hash(context[:500])}"
            if cache_key in self.response_cache:
                return self.response_cache[cache_key]
            
            # Set maximum tokens for both input and output
            max_input_tokens = 128000  # Maximum context window for GPT-4o
            max_output_tokens = 16384   # Maximum response tokens
            
            # Count and limit input tokens
            import tiktoken
            enc = tiktoken.get_encoding("cl100k_base")
            
            # Count input tokens
            input_text = f"Context: {context}\n\nQuestion: {question}"
            input_tokens = len(enc.encode(input_text))
            
            # Truncate context if too long
            if input_tokens > max_input_tokens:
                # Calculate how much context we can keep
                question_tokens = len(enc.encode(question))
                system_tokens = len(enc.encode(self.system_prompt))
                available_tokens = max_input_tokens - question_tokens - system_tokens - 100  # Buffer
                
                if available_tokens > 0:
                    # Truncate context to fit
                    context = enc.decode(enc.encode(context)[:available_tokens])
                else:
                    return "Error: Question too long for processing."
            
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
                "model": "gpt-4o",
                "messages": messages,
                "temperature": 0.3,
                "max_tokens": max_output_tokens  # Maximum output tokens
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
        """Determine appropriate response length based on question type."""
        if not normalize_length:
            return 16384
        
        question_lower = question.lower()
        
        # Short responses for simple questions
        if any(word in question_lower for word in ['what is', 'define', 'explain briefly', 'summarize']):
            return 16384
        
        # Medium responses for analysis questions
        if any(word in question_lower for word in ['analyze', 'compare', 'discuss', 'evaluate']):
            return 16384
        
        # Long responses for complex questions
        if any(word in question_lower for word in ['detailed', 'comprehensive', 'thorough']):
            return 16384
        
        # Default medium length
        return 16384

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
                
                # Enhanced question processing - always request comprehensive answers
                enhanced_question = f"Please provide a comprehensive, detailed analysis of: {question}. Include specific data points, historical context, trends, and implications. Do not provide brief answers - give the full analysis."
                
                answer = self.llm.generate_answer(enhanced_question, context, normalize_length=False)  # Force maximum length
                
                # Store conversation history with chunk metadata from top result
                top_chunk = results[0] if results else None
                self.conversation_history.append({
                    'question': question,
                    'answer': answer,
                    'context': context,
                    'timestamp': datetime.now().isoformat(),
                    'source': top_chunk['metadata'].get('source') if top_chunk else None,
                    'chunk_id': top_chunk['metadata'].get('chunk_id') if top_chunk else None,
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
        self.internet_mode = internet_mode
        self.image_processor = OnDemandImageProcessor()
        self.performance_metrics = {
            'total_queries': 0,
            'avg_response_time': 0,
            'error_count': 0
        }
        self.chart_extractor = PDFChartExtractor()  # Add this line

    def process_web_uploads(self, uploaded_files):
        """Process multiple uploaded files."""
        try:
            if not uploaded_files:
                return False
            
            all_documents = []
            
            for uploaded_file in uploaded_files:
                documents = self.file_handler._process_single_file(uploaded_file)
                if documents:
                    all_documents.extend(documents)
            
            if all_documents:
                # Add documents to vector store
                self.vector_store.add_documents(all_documents)
                return True
            else:
                return False
                
        except Exception as e:
            logger.error(f"Error processing web uploads: {str(e)}")
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
                system_prompt="You are a financial consultant with access to live web information. Answer questions based on the provided web search results. CRITICAL: When asked about financial data, revenue, projections, or specific figures, ALWAYS lead with the most specific and detailed numerical information available. Do NOT start with vague summaries or general statements. If the context contains specific numbers, dates, or financial figures, present those FIRST. Only provide general commentary AFTER presenting the specific data. Always cite your sources and provide accurate, up-to-date information from the web. No one sentence answers. Always provide a detailed answer."
            )
            
            # For now, return a simple response indicating internet mode
            return f"Internet mode is enabled. This would search the web for: {question}"
            
        except Exception as e:
            logger.error(f"Error generating internet answer: {str(e)}")
            return f"Error generating internet answer: {str(e)}"

    def process_question_with_mode(self, question: str, normalize_length: bool = True) -> str:
        """Process question using either document mode or internet mode."""
        # Always use the working function for internet mode
        if self.internet_mode:
            # Use internet mode
            logger.info("Processing question using internet mode")
            answer = generate_live_web_answer(question)  # ← USE THE WORKING FUNCTION
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
            answer = generate_live_web_answer(follow_up_question)  # ← USE THE WORKING FUNCTION
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

    def process_live_web_question(self, question: str) -> str:
        """Process question using live web search."""
        try:
            answer = generate_live_web_answer(question)
            self.add_to_conversation_history(question, answer, "live_web_search", "internet")
            return answer
        except Exception as e:
            logger.error(f"Error processing live web question: {str(e)}")
            return f"Error: {str(e)}"

    def process_image_request(self, question: str, pdf_path: str = None) -> str:
        """Process a request for images from a document, focusing on relevant pages."""
        try:
            logger.info(f"Processing image request: {question}")
            
            # Determine which PDF to search
            if pdf_path is None:
                # Use the most recent document from conversation history
                conversation_history = self.get_conversation_history()
                for conv in reversed(conversation_history):
                    if conv.get('source'):
                        pdf_path = conv['source']
                        break
                
                if not pdf_path:
                    return "No document specified. Please upload a document first."
            
            # Check if PDF exists
            if not os.path.exists(pdf_path):
                # Try temp directory
                temp_path = os.path.join("temp", os.path.basename(pdf_path))
                if os.path.exists(temp_path):
                    pdf_path = temp_path
                else:
                    return f"Document not found: {pdf_path}"
            
            # Get relevant pages from recent conversation or search results
            relevant_pages = self._get_relevant_pages_for_question(question, pdf_path)
            
            if not relevant_pages:
                # Fallback to processing all pages if no relevant pages found
                logger.info("No specific relevant pages found, processing entire document")
                all_images = self.image_processor.extract_images_from_pdf(pdf_path)
            else:
                # Process only relevant pages
                logger.info(f"Processing images from relevant pages: {relevant_pages}")
                all_images = self.image_processor.extract_images_from_relevant_pages(pdf_path, relevant_pages)
            
            if not all_images:
                return "No images found in the relevant pages of the document."
            
            logger.info(f"Found {len(all_images)} images, analyzing...")
            
            # ANALYZE EACH IMAGE
            results = []
            for i, img_info in enumerate(all_images):
                try:
                    logger.info(f"Analyzing image {i+1}/{len(all_images)}: {img_info['path']}")
                    analysis = self.image_processor.analyze_image_with_gpt4(img_info['path'], question)
                    
                    results.append({
                        **img_info,
                        'analysis': analysis
                    })
                    
                except Exception as e:
                    logger.error(f"Error analyzing image {img_info['path']}: {str(e)}")
                    results.append({
                        **img_info,
                        'analysis': f"Error analyzing this image: {str(e)}"
                    })
            
            # FORMAT THE RESPONSE
            response_parts = []
            response_parts.append(f"📊 **Found {len(results)} image(s) in relevant pages:**")
            response_parts.append("")
            
            for i, img_info in enumerate(results, 1):
                response_parts.append(f"**Image {i} (Page {img_info['page']}):**")
                response_parts.append(img_info['analysis'])
                response_parts.append("")
            
            return "\n".join(response_parts)
            
        except Exception as e:
            logger.error(f"Error processing image request: {str(e)}")
            return f"Error processing image request: {str(e)}"

    def process_chart_request(self, question: str, pdf_path: str = None) -> list:
        """Process requests for charts/graphs and return image files."""
        try:
            logger.info(f"Starting chart request processing for: {question}")
            
            # Determine which PDF to search
            if pdf_path is None:
                # First try to get PDF paths from the file handler
                if hasattr(self, 'file_handler') and self.file_handler:
                    saved_paths = self.file_handler.get_saved_pdf_paths()
                    if saved_paths:
                        pdf_path = saved_paths[0]
                        logger.info(f"Using PDF path from file handler: {pdf_path}")
                
                # If no path from file handler, try conversation history
                if not pdf_path:
                    conversation_history = self.get_conversation_history()
                    if conversation_history:
                        for conv in reversed(conversation_history):
                            if 'source' in conv:
                                pdf_path = conv['source']
                                logger.info(f"Using PDF path from conversation: {pdf_path}")
                                break
            
            if not pdf_path:
                logger.error("No PDF path found")
                return []
            
            if not os.path.exists(pdf_path):
                logger.error(f"PDF file does not exist: {pdf_path}")
                return []
            
            logger.info(f"Processing PDF: {pdf_path}")
            
            # Check if asking for a specific page
            import re
            page_match = re.search(r'page\s+(\d+)', question.lower())
            if page_match:
                page_number = int(page_match.group(1))
                logger.info(f"Converting specific page {page_number} to image")
                image_path = self.chart_extractor.convert_single_page_to_image(pdf_path, page_number)
                if image_path and os.path.exists(image_path):
                    logger.info(f"Successfully created image: {image_path}")
                    return [{
                        'type': 'chart_image',
                        'page': page_number,
                        'image_path': image_path,
                        'description': f"Chart from page {page_number}"
                    }]
                else:
                    logger.error(f"Failed to create image for page {page_number}")
                    return []
            
            # Get relevant pages for the question
            relevant_pages = self._get_relevant_pages_for_question(question, pdf_path)
            
            if not relevant_pages:
                # If no relevant pages found, try first few pages (but check PDF length first)
                import fitz
                doc = fitz.open(pdf_path)
                total_pages = len(doc)
                doc.close()
                
                # Only use pages that actually exist
                relevant_pages = [i for i in [1, 2, 3] if i <= total_pages]
                logger.info(f"No relevant pages found, using first {len(relevant_pages)} pages: {relevant_pages}")
            
            logger.info(f"Converting pages to images: {relevant_pages}")
            
            # Convert relevant pages to images
            chart_images = []
            for page_num in relevant_pages:
                logger.info(f"Converting page {page_num} to image")
                image_path = self.chart_extractor.convert_single_page_to_image(pdf_path, page_num)
                if image_path and os.path.exists(image_path):
                    logger.info(f"Successfully created image for page {page_num}: {image_path}")
                    chart_images.append({
                        'type': 'chart_image',
                        'page': page_num,
                        'image_path': image_path,
                        'description': f"Chart from page {page_num}"
                    })
                else:
                    logger.error(f"Failed to create image for page {page_num}")
            
            logger.info(f"Generated {len(chart_images)} chart images")
            return chart_images
            
        except Exception as e:
            logger.error(f"Error processing chart request: {str(e)}")
            return []

    def _get_relevant_pages_for_question(self, question: str, pdf_path: str) -> list:
        """Get relevant page numbers for a question."""
        try:
            relevant_pages = set()
            
            # Method 1: Get pages from recent search results
            search_results = self.vector_store.search(question, k=5)
            for chunk in search_results:
                page = chunk['metadata'].get('page')
                if page:
                    relevant_pages.add(int(page))
            
            # Method 2: Get pages from recent conversation history
            conversation_history = self.get_conversation_history()
            for conv in conversation_history[-5:]:  # Last 5 conversations
                page = conv.get('page')
                if page:
                    relevant_pages.add(int(page))
            
            # Method 3: Check if question mentions specific pages
            import re
            page_matches = re.findall(r'page\s+(\d+)', question.lower())
            for page_match in page_matches:
                relevant_pages.add(int(page_match))
            
            logger.info(f"Found relevant pages: {sorted(relevant_pages)}")
            return sorted(relevant_pages)
            
        except Exception as e:
            logger.error(f"Error getting relevant pages: {str(e)}")
            return []

    # Add this debug method to the RAGSystem class (add it after the existing methods)
    def debug_chart_processing(self, question: str):
        """Debug method to see what's happening with chart processing."""
        try:
            logger.info("=== DEBUGGING CHART PROCESSING ===")
            
            # Check if file handler exists
            if hasattr(self, 'file_handler') and self.file_handler:
                saved_paths = self.file_handler.get_saved_pdf_paths()
                logger.info(f"File handler saved paths: {saved_paths}")
            else:
                logger.info("No file handler found")
            
            # Check conversation history
            conv_history = self.get_conversation_history()
            logger.info(f"Conversation history length: {len(conv_history)}")
            
            # Try to get PDF path
            pdf_path = None
            if hasattr(self, 'file_handler') and self.file_handler:
                saved_paths = self.file_handler.get_saved_pdf_paths()
                if saved_paths:
                    pdf_path = saved_paths[0]
                    logger.info(f"Using PDF path from file handler: {pdf_path}")
            
            if not pdf_path:
                for conv in reversed(conv_history):
                    if conv.get('source'):
                        pdf_path = conv['source']
                        logger.info(f"Using PDF path from conversation: {pdf_path}")
                        break
            
            if not pdf_path:
                logger.error("No PDF path found!")
                return "No PDF path found"
            
            if not os.path.exists(pdf_path):
                logger.error(f"PDF file does not exist: {pdf_path}")
                return f"PDF file does not exist: {pdf_path}"
            
            # Check if chart_extractor exists
            if not hasattr(self, 'chart_extractor'):
                logger.error("No chart_extractor found!")
                return "No chart_extractor found"
            
            # Try to convert first page
            logger.info("Attempting to convert page 1 to image...")
            image_path = self.chart_extractor.convert_single_page_to_image(pdf_path, 1)
            
            if image_path:
                logger.info(f"Successfully created image: {image_path}")
                if os.path.exists(image_path):
                    logger.info("Image file exists on disk")
                    return f"Success! Image created: {image_path}"
                else:
                    logger.error("Image file does not exist on disk")
                    return "Image file does not exist on disk"
            else:
                logger.error("convert_single_page_to_image returned None")
                return "convert_single_page_to_image returned None"
                
        except Exception as e:
            logger.error(f"Debug error: {str(e)}")
            return f"Debug error: {str(e)}"

    # Add this method to your RAGSystem class
    def analyze_image_with_gpt4(self, image_path: str, question: str = None) -> str:
        """Analyze image using GPT-4 Vision API with enhanced prompts."""
        try:
            import base64
            
            # Read and encode image
            with open(image_path, "rb") as image_file:
                image_data = base64.b64encode(image_file.read()).decode('utf-8')
            
            # Determine analysis type based on question
            if "chart" in question.lower() or "graph" in question.lower() or "data" in question.lower():
                system_prompt = """You are an expert data analyst. Analyze this chart/graph and provide:
1. **Data Extraction**: Extract all numerical values, labels, and data points
2. **Trend Analysis**: Identify key trends, patterns, and relationships
3. **Key Insights**: Highlight the most important findings
4. **Context**: Provide context for what the data represents
5. **Recommendations**: If applicable, suggest actions based on the data

Present the information in a clear, structured format with specific numbers and details."""
            
            elif "text" in question.lower() or "ocr" in question.lower():
                system_prompt = """You are an expert OCR specialist. Extract and transcribe all text from this image:
1. **Text Content**: Transcribe all visible text accurately
2. **Formatting**: Preserve formatting, structure, and layout
3. **Tables**: If tables are present, format them clearly
4. **Handwriting**: If handwritten text is present, transcribe it
5. **Special Characters**: Include all symbols, numbers, and special characters

Present the extracted text in a clean, readable format."""
            
            else:
                system_prompt = """You are an expert image analyst. Analyze this image comprehensively:
1. **Visual Content**: Describe what you see in detail
2. **Objects/People**: Identify objects, people, or elements present
3. **Context**: Provide context about the image content
4. **Text**: Extract any visible text or labels
5. **Charts/Data**: If charts or data are present, analyze them
6. **Insights**: Provide relevant insights and observations

Be thorough and specific in your analysis."""
            
            # Prepare messages
            messages = [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": question or "Please analyze this image in detail."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_data}"
                            }
                        }
                    ]
                }
            ]
            
            # Call GPT-4 Vision API
            client = openai.OpenAI(api_key=OPENAI_API_KEY)
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                max_tokens=4096,
                temperature=0.3
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error in image analysis: {str(e)}")
            return f"Error analyzing image: {str(e)}"

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

def render_chunk_source_image(source_path, page_num, chunk_text):
    """Render a PDF page as an image, highlighting the chunk text if possible."""
    import fitz
    import os
    os.makedirs("temp", exist_ok=True)
    doc = fitz.open(source_path)
    page = doc.load_page(page_num - 1)  # 0-based index
    # Try to highlight all instances of the chunk text
    if chunk_text:
        text_instances = page.search_for(chunk_text)
        for inst in text_instances:
            page.add_highlight_annot(inst)
    pix = page.get_pixmap(dpi=200)
    img_path = f"temp/page_{page_num}_chunk_highlighted.png"
    pix.save(img_path)
    doc.close()
    return img_path

def batch_documents_by_token_limit(documents, max_tokens=16384):
    enc = tiktoken.get_encoding("cl100k_base")
    batches = []
    current_batch = []
    current_tokens = 0
    for doc in documents:
        tokens = len(enc.encode(doc['text']))
        if tokens > max_tokens:
            print(f"SKIPPING CHUNK: {tokens} tokens (too large for a single batch)")
            continue  # Skip this chunk
        if current_tokens + tokens > max_tokens and current_batch:
            print(f"Creating batch with {len(current_batch)} docs, {current_tokens} tokens")
            batches.append(current_batch)
            current_batch = []
            current_tokens = 0
        current_batch.append(doc)
        current_tokens += tokens
    if current_batch:
        print(f"Creating batch with {len(current_batch)} docs, {current_tokens} tokens")
        batches.append(current_batch)
    return batches

def search_web_live(query: str, num_results: int = 5) -> list:
    """Search the web using DuckDuckGo for live results."""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=num_results))
        
        web_results = []
        for result in results:
            web_results.append({
                "title": result.get("title", ""),
                "snippet": result.get("body", ""),
                "link": result.get("link", ""),
                "source": "DuckDuckGo Live Search"
            })
        
        return web_results
    except Exception as e:
        logger.error(f"DuckDuckGo search error: {str(e)}")
        return []

def generate_live_web_answer(question: str) -> str:
    """Generate answer using live web search results."""
    try:
        # Search the web live
        web_results = search_web_live(question, num_results=5)
        
        if not web_results:
            return "No live web results found for your question."
        
        # Format web results for context
        web_context = "\n\n".join([
            f"Source: {result['title']}\n{result['snippet']}\nURL: {result['link']}"
            for result in web_results
        ])
        
        # Use OpenAI to generate answer from live web results
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "You are a financial consultant with access to live web information. Answer questions based on the provided web search results. CRITICAL: When asked about financial data, revenue, projections, or specific figures, ALWAYS lead with the most specific and detailed numerical information available. Do NOT start with vague summaries or general statements. If the context contains specific numbers, dates, or financial figures, present those FIRST. Only provide general commentary AFTER presenting the specific data. Always cite your sources and provide accurate, up-to-date information from the web. No one sentence answers. Always provide a detailed answer."
                },
                {
                    "role": "user",
                    "content": f"Live Web Search Results:\n{web_context}\n\nQuestion: {question}\n\nPlease answer based on the live web results above. Include source citations."
                }
            ],
            max_tokens=16384,
            temperature=0.3
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        logger.error(f"Error generating live web answer: {str(e)}")
        return f"Error accessing live web: {str(e)}"

# Add this new class for on-demand image processing
class OnDemandImageProcessor:
    def __init__(self):
        self.processed_images = {}
        
    def extract_images_from_pdf(self, pdf_path: str) -> list:
        """Extract images from PDF using PyMuPDF."""
        try:
            if not os.path.exists(pdf_path):
                logger.error(f"PDF file not found: {pdf_path}")
                return []
            
            doc = fitz.open(pdf_path)
            images = []
            
            for page_idx in range(len(doc)):
                try:
                    page = doc.load_page(page_idx)
                    image_list = page.get_images()
                    
                    for img_index, img in enumerate(image_list):
                        try:
                            # Get image data
                            xref = img[0]
                            pix = fitz.Pixmap(doc, xref)
                            
                            # Convert to PNG format
                            if pix.n - pix.alpha < 4:  # GRAY or RGB
                                img_data = pix.tobytes("png")
                            else:  # CMYK: convert to RGB first
                                pix1 = fitz.Pixmap(fitz.csRGB, pix)
                                img_data = pix1.tobytes("png")
                                pix1 = None
                            
                            # Save image
                            img_filename = f"page_{page_idx + 1}_image_{img_index + 1}.png"
                            img_path = os.path.join("temp", img_filename)
                            os.makedirs("temp", exist_ok=True)
                            
                            with open(img_path, "wb") as img_file:
                                img_file.write(img_data)
                            
                            images.append({
                                'path': img_path,
                                'page': page_idx + 1,
                                'image_index': img_index + 1,
                                'description': f"Image from page {page_idx + 1}"
                            })
                            
                            pix = None
                            
                        except Exception as e:
                            logger.warning(f"Error extracting image {img_index} from page {page_idx + 1}: {str(e)}")
                            continue
                            
                except Exception as e:
                    logger.warning(f"Error processing page {page_idx + 1}: {str(e)}")
                    continue
            
            doc.close()
            logger.info(f"Extracted {len(images)} images from PDF")
            return images
        
        except Exception as e:
            logger.error(f"Error extracting images from PDF: {str(e)}")
            return []
    
    def analyze_image_with_gpt4(self, image_path: str, question: str = None) -> str:
        """Analyze an image using GPT-4 Vision API."""
        try:
            if not os.path.exists(image_path):
                return "Image file not found."
            
            # Check file size (OpenAI has limits)
            file_size = os.path.getsize(image_path)
            if file_size > 20 * 1024 * 1024:  # 20MB limit
                return "Image file too large for analysis."
            
            # Read and encode the image
            with open(image_path, "rb") as img_file:
                img_data = img_file.read()
                encoded_image = base64.b64encode(img_data).decode('ascii')
            
            # Prepare the prompt
            if question:
                system_prompt = f"Analyze this image and answer the specific question: {question}. If you see charts, graphs, or tables, extract the data and present it in a clear format."
            else:
                system_prompt = "Analyze this image and extract all text and data from any charts, graphs, or tables. Present chart data in tabular format for easy understanding."
            
            # API call to GPT-4 Vision
            headers = {
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": "gpt-4o",
                "messages": [
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{encoded_image}"
                                }
                            }
                        ]
                    }
                ],
                "max_tokens": 4096,
                "temperature": 0.3
            }
            
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"]
            else:
                logger.error(f"GPT-4 API error: {response.status_code} - {response.text}")
                return f"Error analyzing image: API returned {response.status_code}"
                
        except Exception as e:
            logger.error(f"Error analyzing image with GPT-4: {str(e)}")
            return f"Error analyzing image: {str(e)}"

    # Add this new method to the OnDemandImageProcessor class
    def extract_images_from_relevant_pages(self, pdf_path: str, relevant_pages: list, buffer_pages: int = 2) -> list:
        """Extract images only from pages near the relevant chunks."""
        try:
            if not os.path.exists(pdf_path):
                logger.error(f"PDF file not found: {pdf_path}")
                return []
            
            doc = fitz.open(pdf_path)
            total_pages = len(doc)
            images = []
            
            # Expand relevant pages with buffer
            pages_to_process = set()
            for page in relevant_pages:
                # Add the page itself and buffer pages around it
                for i in range(max(1, page - buffer_pages), min(total_pages + 1, page + buffer_pages + 1)):
                    pages_to_process.add(i)
            
            logger.info(f"Processing images from pages: {sorted(pages_to_process)} (out of {total_pages} total pages)")
            
            for page_num in sorted(pages_to_process):
                try:
                    page_idx = page_num - 1  # Convert to 0-based index
                    page = doc.load_page(page_idx)
                    image_list = page.get_images()
                    
                    for img_index, img in enumerate(image_list):
                        try:
                            # Get image data
                            xref = img[0]
                            pix = fitz.Pixmap(doc, xref)
                            
                            # Convert to PNG format
                            if pix.n - pix.alpha < 4:  # GRAY or RGB
                                img_data = pix.tobytes("png")
                            else:  # CMYK: convert to RGB first
                                pix1 = fitz.Pixmap(fitz.csRGB, pix)
                                img_data = pix1.tobytes("png")
                                pix1 = None
                            
                            # Save image
                            img_filename = f"page_{page_num}_image_{img_index + 1}.png"
                            img_path = os.path.join("temp", img_filename)
                            os.makedirs("temp", exist_ok=True)
                            
                            with open(img_path, "wb") as img_file:
                                img_file.write(img_data)
                            
                            images.append({
                                'path': img_path,
                                'page': page_num,
                                'image_index': img_index + 1,
                                'description': f"Image from page {page_num}"
                            })
                            
                            pix = None
                            
                        except Exception as e:
                            logger.warning(f"Error extracting image {img_index} from page {page_num}: {str(e)}")
                            continue
                            
                except Exception as e:
                    logger.warning(f"Error processing page {page_num}: {str(e)}")
                    continue
            
            doc.close()
            logger.info(f"Extracted {len(images)} images from {len(pages_to_process)} relevant pages")
            return images
            
        except Exception as e:
            logger.error(f"Error extracting images from relevant pages: {str(e)}")
            return []

# Add this new class for PDF to image conversion and chart extraction
class PDFChartExtractor:
    def __init__(self):
        self.output_dir = "chart_extractions"
        os.makedirs(self.output_dir, exist_ok=True)
    
    def convert_single_page_to_image(self, pdf_path, page_number):
        """Convert a single PDF page to an image."""
        try:
            import fitz  # PyMuPDF
            
            # Open the PDF
            doc = fitz.open(pdf_path)
            
            if page_number > len(doc) or page_number < 1:
                logger.error(f"Page {page_number} not found. PDF has {len(doc)} pages.")
                return None
            
            # Get the specific page
            page = doc.load_page(page_number - 1)  # 0-based index
            
            # Render page as image
            pix = page.get_pixmap(dpi=200)
            
            # Save the page image
            image_path = os.path.join(self.output_dir, f"page_{page_number}.png")
            pix.save(image_path)
            
            doc.close()
            
            logger.info(f"Successfully converted page {page_number} to {image_path}")
            return image_path
            
        except Exception as e:
            logger.error(f"Error converting page {page_number} to image: {str(e)}")
            return None

    def convert_pdf_to_images(self, pdf_path):
        """Convert PDF pages to images using PyMuPDF instead of pdf2image."""
        try:
            import fitz  # PyMuPDF
            
            # Open the PDF
            doc = fitz.open(pdf_path)
            image_paths = []
            
            # Convert each page to image
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                
                # Render page as image
                pix = page.get_pixmap(dpi=200)
                
                # Save image
                image_path = os.path.join(self.output_dir, f"page_{page_num + 1}.png")
                pix.save(image_path)
                image_paths.append(image_path)
            
            doc.close()
            return image_paths
            
        except Exception as e:
            logger.error(f"Error converting PDF to images with PyMuPDF: {str(e)}")
            return []

    def get_chart_data_by_page(self, pdf_path, page_number):
        """Extract chart data from a specific page using PyMuPDF."""
        try:
            import fitz  # PyMuPDF
            
            # Open the PDF
            doc = fitz.open(pdf_path)
            
            if page_number > len(doc) or page_number < 1:
                return f"Page {page_number} not found. PDF has {len(doc)} pages."
            
            # Get the specific page
            page = doc.load_page(page_number - 1)  # 0-based index
            
            # Render page as image
            pix = page.get_pixmap(dpi=200)
            
            # Save the page image
            image_path = os.path.join(self.output_dir, f"page_{page_number}.png")
            pix.save(image_path)
            
            doc.close()
            
            # Process the image
            extracted_text = self.process_image_and_extract_text(image_path)
            
            return extracted_text if extracted_text else f"No chart data found on page {page_number}"
            
        except Exception as e:
            logger.error(f"Error extracting chart data from page {page_number}: {str(e)}")
            return f"Error processing page {page_number}: {str(e)}"

    def process_image_and_extract_text(self, image_path, output_dir=None):
        """Process a single image and extract text/chart data using GPT-4 Vision."""
        try:
            if output_dir is None:
                output_dir = self.output_dir
            
            # Open and encode image
            with open(image_path, "rb") as image_file:
                image_data = base64.b64encode(image_file.read()).decode('utf-8')
            
            # Use OpenAI Vision API to extract text and chart data
            client = openai.OpenAI(api_key=OPENAI_API_KEY)
            
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Extract all text and chart data from this image. If there are charts, graphs, or tables, convert them to a tabular format. Include all text content and organize chart data in a clear, structured way."
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{image_data}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=4096,
                temperature=0.1
            )
            
            extracted_text = response.choices[0].message.content
            
            # Save extracted text to file
            page_num = os.path.basename(image_path).split('_')[1].split('.')[0]
            output_file = os.path.join(output_dir, f"page_{page_num}_extracted.txt")
            
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(extracted_text)
            
            return extracted_text
            
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {str(e)}")
            return f"Error processing image: {str(e)}"

    def process_pdf_for_charts(self, pdf_path):
        """Process entire PDF and extract chart data from all pages."""
        try:
            # Convert PDF to images
            image_paths = self.convert_pdf_to_images(pdf_path)
            
            if not image_paths:
                return {}
            
            extracted_data = {}
            
            # Process each page
            for i, image_path in enumerate(image_paths):
                page_num = i + 1
                logger.info(f"Processing page {page_num} for chart data")
                
                extracted_text = self.process_image_and_extract_text(image_path)
                if extracted_text and not extracted_text.startswith("Error"):
                    extracted_data[page_num] = extracted_text
            
            return extracted_data
            
        except Exception as e:
            logger.error(f"Error processing PDF for charts: {str(e)}")
            return {}

if __name__ == "__main__": 
    rag_system = RAGSystem()
    rag_system.run() 
