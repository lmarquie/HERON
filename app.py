import streamlit as st
import os
from local_draft import RAGSystem, WebFileHandler, render_chunk_source_image
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from datetime import datetime
import time
import logging
import json
import concurrent.futures
import re
import csv
import io
import subprocess
import sys
import importlib

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set page config
st.set_page_config(
    page_title="HERON",
    page_icon="",
    layout="wide"
)

# Force reload to clear cache
if 'local_draft' in sys.modules:
    importlib.reload(sys.modules['local_draft'])

# Add this right after your imports at the very top of app.py
def export_transcription_to_pdf(transcription_text: str, filename: str = "transcription"):
    """Export transcription to PDF for download."""
    try:
        from reportlab.lib.pagesizes import letter, A4
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.lib.colors import black, blue
        from reportlab.lib.enums import TA_LEFT, TA_CENTER
        
        # Create temp directory
        temp_dir = "temp"
        os.makedirs(temp_dir, exist_ok=True)
        
        # Create PDF filename
        pdf_filename = f"{filename}_transcription.pdf"
        pdf_path = os.path.join(temp_dir, pdf_filename)
        
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
        title = Paragraph(f"Audio Transcription", title_style)
        story.append(title)
        story.append(Spacer(1, 20))
        
        # Add metadata
        metadata = f"""
        <b>Original File:</b> {filename}<br/>
        <b>Export Date:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br/>
        <b>Character Count:</b> {len(transcription_text):,}<br/>
        <b>Word Count:</b> {len(transcription_text.split()):,}
        """
        meta_para = Paragraph(metadata, header_style)
        story.append(meta_para)
        story.append(Spacer(1, 30))
        
        # Remove excessive repeated phrases (e.g., 'be able to be able to ...')
        def remove_repetitions(text, min_repeats=3):
            # This regex finds 3 or more consecutive repeats of a short phrase (1-6 words)
            pattern = re.compile(r'((?:\b\w+\b(?:\s+|\s*\n\s*)){1,6})\1{' + str(min_repeats-1) + ',}', re.IGNORECASE)
            def repl(match):
                return match.group(1)
            # Apply repeatedly until no more matches
            prev = None
            while prev != text:
                prev = text
                text = pattern.sub(repl, text)
            return text
        
        cleaned_transcription = remove_repetitions(transcription_text)
        # Split transcription into sentences for better formatting
        sentences = re.split(r'(?<=[.!?]) +', cleaned_transcription)
        
        for sentence in sentences:
            if sentence.strip():
                clean_sentence = sentence.strip().replace('\n', ' ')
                if clean_sentence:
                    p = Paragraph(clean_sentence, body_style)
                    story.append(p)
                    # Reduce extra space for readability
                    story.append(Spacer(1, 18))  # Reduced spacing between sentences
        
        # Build the PDF
        doc.build(story)
        
        return pdf_path
        
    except Exception as e:
        st.error(f"Error creating PDF: {str(e)}")
        return None

# Initialize RAG system with improved session management
def initialize_rag_system():
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = RAGSystem(is_web=True, use_vision_api=False)
        st.session_state.documents_loaded = False
        st.session_state.answer_given = False
        st.session_state.processing_status = {}
        st.session_state.last_upload_time = None
        st.session_state.error_count = 0
        st.session_state.performance_metrics = {}
        st.session_state.internet_mode = False  # Initialize internet mode

# Simple answer generation with improved error handling
def generate_answer(question):
    try:
        start_time = time.time()
        
        # Check if question is in French
        is_french = _is_french_question(question)
        
        # Use the new mode-aware question processing
        answer = st.session_state.rag_system.process_question_with_mode(question, normalize_length=True)
        
        # Update performance metrics
        response_time = time.time() - start_time
        st.session_state.performance_metrics['last_response_time'] = response_time
        
        # Translate to French if question was in French
        if is_french:
            answer = _translate_to_french(answer)
        
        return answer
        
    except Exception as e:
        st.session_state.error_count += 1
        logger.error(f"Error generating answer: {str(e)}")
        return f"Error: {str(e)}"

# Follow-up question generation with improved error handling
def generate_follow_up(follow_up_question):
    return st.session_state.rag_system.handle_follow_up(follow_up_question)

# PDF Export function
def export_conversation_to_pdf():
    """Export conversation history to PDF."""
    try:
        conversation_history = st.session_state.rag_system.get_conversation_history()
        
        if not conversation_history:
            return None
        
        # Create PDF file
        pdf_filename = f"heron_conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        pdf_path = os.path.join("exports", pdf_filename)
        os.makedirs("exports", exist_ok=True)
        
        doc = SimpleDocTemplate(pdf_path, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []
        
        # Title
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=16,
            spaceAfter=30
        )
        story.append(Paragraph("HERON Conversation Export", title_style))
        story.append(Spacer(1, 12))
        
        # Add each conversation exchange
        question_counter = 1
        for i, conv in enumerate(conversation_history):
            question_type = conv.get('question_type', '')
            mode = conv.get('mode', 'Unknown')
            
            # Determine if this is a follow-up
            is_followup = 'followup' in question_type.lower()
            
            # Question label
            if is_followup:
                question_label = f"Follow-up {question_counter}:"
            else:
                question_label = f"Q{question_counter}:"
            
            # Question
            question_style = ParagraphStyle(
                'Question',
                parent=styles['Normal'],
                fontSize=12,
                spaceAfter=6,
                leftIndent=20
            )
            story.append(Paragraph(f"<b>{question_label}</b> {conv['question']}", question_style))
            
            # Answer
            if isinstance(conv['answer'], list):
                answer_text = "Found 1 image" if conv['answer'] else "No images found"
                story.append(Paragraph(f"<b>A{question_counter}:</b> {answer_text}", question_style))
            else:
                story.append(Paragraph(f"<b>A{question_counter}:</b> {conv['answer']}", question_style))
            
            # Add metadata if available
            source = conv.get('source', '')
            page = conv.get('page', '')
            if source or page:
                meta_text = f"<i>Mode: {mode}"
                if source:
                    meta_text += f" | Source: {source}"
                if page:
                    meta_text += f" | Page: {page}"
                meta_text += "</i>"
                story.append(Paragraph(meta_text, question_style))
            
            story.append(Spacer(1, 12))
            question_counter += 1
        
        # Build PDF
        doc.build(story)
        
        return pdf_path
        
    except Exception as e:
        logger.error(f"Error creating PDF: {str(e)}")
        return None

def export_conversation_to_markdown():
    """Export conversation history to Markdown."""
    try:
        conversation_history = st.session_state.rag_system.get_conversation_history()
        
        if not conversation_history:
            return None
        
        # Create markdown content
        markdown_content = []
        markdown_content.append("# HERON Conversation Export")
        markdown_content.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        markdown_content.append("")
        
        # Add each conversation exchange
        question_counter = 1
        for i, conv in enumerate(conversation_history):
            question_type = conv.get('question_type', '')
            mode = conv.get('mode', 'Unknown')
            
            # Determine if this is a follow-up
            is_followup = 'followup' in question_type.lower()
            
            # Question label
            if is_followup:
                question_label = f"Follow-up {question_counter}:"
            else:
                question_label = f"Q{question_counter}:"
            
            markdown_content.append(f"## {question_label} {conv['question']}")
            markdown_content.append("")
            
            # Answer
            if isinstance(conv['answer'], list):
                answer_text = "Found 1 image" if conv['answer'] else "No images found"
                markdown_content.append(f"**A{question_counter}:** {answer_text}")
            else:
                markdown_content.append(f"**A{question_counter}:** {conv['answer']}")
            
            # Add metadata if available
            source = conv.get('source', '')
            page = conv.get('page', '')
            if source or page:
                meta_text = f"*Mode: {mode}"
                if source:
                    meta_text += f" | Source: {source}"
                if page:
                    meta_text += f" | Page: {page}"
                meta_text += "*"
                markdown_content.append(meta_text)
            
            markdown_content.append("")
            markdown_content.append("---")
            markdown_content.append("")
            question_counter += 1
        
        return "\n".join(markdown_content)
        
    except Exception as e:
        logger.error(f"Error creating Markdown: {str(e)}")
        return None

def export_conversation_to_csv():
    """Export conversation history to CSV."""
    try:
        conversation_history = st.session_state.rag_system.get_conversation_history()
        
        if not conversation_history:
            return None
        
        # Create CSV content
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Write header
        writer.writerow(['Question Number', 'Question Type', 'Question', 'Answer', 'Mode', 'Source', 'Page', 'Timestamp'])
        
        # Write data
        question_counter = 1
        for i, conv in enumerate(conversation_history):
            question = conv['question']
            question_type = conv.get('question_type', '')
            
            # Determine if this is a follow-up
            is_followup = 'followup' in question_type.lower()
            question_type_display = "Follow-up" if is_followup else "Initial"
            
            # Handle answer (could be text or list)
            if isinstance(conv['answer'], list):
                answer = "Found 1 image" if conv['answer'] else "No images found"
            else:
                answer = conv['answer']
            
            mode = conv.get('mode', 'Unknown')
            source = conv.get('source', '')
            page = conv.get('page', '')
            timestamp = conv.get('timestamp', '')
            
            writer.writerow([f"Q{question_counter}", question_type_display, question, answer, mode, source, page, timestamp])
            question_counter += 1
        
        return output.getvalue()
        
    except Exception as e:
        logger.error(f"Error creating CSV: {str(e)}")
        return None

# Handle Enter key submission
def handle_enter_key():
    """Handle Enter key press for question submission."""
    if st.session_state.get('question_input', '').strip():
        st.session_state.submit_question = True

def handle_followup_enter_key():
    """Handle Enter key press for follow-up question submission."""
    if st.session_state.get('followup_input', '').strip():
        st.session_state.submit_followup = True

# Main app
initialize_rag_system()

# At the top of your app, initialize these
if 'chat_input_key' not in st.session_state:
    st.session_state.chat_input_key = 0
if 'last_processed_question' not in st.session_state:
    st.session_state.last_processed_question = ""
if 'question_input' not in st.session_state:
    st.session_state['question_input'] = ""

# Main content - Modern chat interface
# (Do not display st.title("HERON") here)

# Display conversation history using logic layer - Modern chat style
conversation_history = st.session_state.rag_system.get_conversation_history()
if conversation_history:
    # Modern chat container
    chat_container = st.container()
    with chat_container:
        for i, conv in enumerate(conversation_history):
            # Show mode indicator
            mode_icon = "Internet" if conv.get('mode') == 'internet' else "Document"
            mode_text = "Internet" if conv.get('mode') == 'internet' else "Document"
            
            # Add audio indicator
            file_type = conv.get('metadata', {}).get('file_type', 'document')
            if file_type == 'audio':
                mode_icon = "Audio"
                mode_text = "Audio"
            
            # Question bubble (user)
            with st.chat_message("user"):
                st.write(f"{conv['question']}")
                st.caption(f"{mode_text} Mode")
            
            # Answer bubble (assistant)
            with st.chat_message("assistant"):
                # If the answer is a list (old image info from OpenCV), show disabled message
                if isinstance(conv['answer'], list):
                    st.info("Image processing has been disabled. Use 'Show Source' button to see highlighted chunks instead.")
                else:
                    # Display answer as plain text
                    answer = conv['answer']
                    st.text(answer)
                    
                    # Show source attribution if available
                    doc_name = conv.get('source_document')
                    page_num = conv.get('source_page')
                    if doc_name or page_num:
                        attribution = "**Source:** "
                        if doc_name:
                            attribution += doc_name
                        if page_num:
                            attribution += f" (Page {page_num})"
                        st.caption(attribution)

                # Show 'Show Source' button if chunk metadata is present, or auto-show for source requests
                source = conv.get('source')
                page = conv.get('page')
                chunk_text = conv.get('chunk_text')
                question_type = conv.get('question_type', '')
                
                if source and page and chunk_text:
                    if question_type == 'source_request':
                        # Automatically show source for source requests
                        pdf_path = source
                        if not os.path.exists(pdf_path):
                            # Try temp directory
                            temp_path = os.path.join("temp", source)
                            if os.path.exists(temp_path):
                                pdf_path = temp_path
                            else:
                                # Try with just the filename
                                filename = os.path.basename(source)
                                temp_path = os.path.join("temp", filename)
                                if os.path.exists(temp_path):
                                    pdf_path = temp_path
                        
                        if os.path.exists(pdf_path):
                            img_path = render_chunk_source_image(pdf_path, page, chunk_text)
                            if os.path.exists(img_path):
                                st.image(img_path, caption=f"Page {page} (highlighted chunk)", use_container_width=True)
                            else:
                                st.warning(f"Could not render source image. Expected path: {img_path}")
                        else:
                            st.warning(f"Source PDF not found: {source}")
                            st.info("Available files in temp directory:")
                            if os.path.exists("temp"):
                                for file in os.listdir("temp"):
                                    st.text(f"  - {file}")
                            else:
                                st.text("  - temp directory doesn't exist")
                    else:
                        # Show button for other cases
                        show_source = st.button(f"Show Source for Q{i+1}", key=f"show_source_{i}")
                        if show_source:
                            # Try to find the actual PDF path (uploaded file may be in temp/)
                            pdf_path = source
                            if not os.path.exists(pdf_path):
                                # Try temp directory
                                temp_path = os.path.join("temp", source)
                                if os.path.exists(temp_path):
                                    pdf_path = temp_path
                            if os.path.exists(pdf_path):
                                img_path = render_chunk_source_image(pdf_path, page, chunk_text)
                                if os.path.exists(img_path):
                                    st.image(img_path, caption=f"Page {page} (highlighted chunk)", use_container_width=True)
                                else:
                                    st.warning("Could not render source image.")
                            else:
                                st.warning("Source PDF not found.")

# Always show chat input (permanent chat interface)
# Initialize input key counter
if 'chat_input_key_counter' not in st.session_state:
    st.session_state.chat_input_key_counter = 0

# Add this function to detect image requests
def is_image_request(question: str) -> bool:
    """Detect if the question is asking for images."""
    question_lower = question.lower()
    image_keywords = [
        'show me the image', 'show me the chart', 'show me the graph', 'show me the figure',
        'show me the table', 'show me the diagram', 'show me the picture',
        'find the image', 'find the chart', 'find the graph', 'find the figure',
        'find the table', 'find the diagram', 'find the picture',
        'extract image', 'extract chart', 'extract graph', 'extract figure',
        'extract table', 'extract diagram', 'extract picture',
        'image in', 'chart in', 'graph in', 'figure in', 'table in',
        'what does the image show', 'what does the chart show', 'what does the graph show',
        'analyze the image', 'analyze the chart', 'analyze the graph',
        'data in the image', 'data in the chart', 'data in the graph'
    ]
    
    return any(keyword in question_lower for keyword in image_keywords)

# Add chart request detection
def is_chart_request(question: str) -> bool:
    """Detect if the question is asking for charts/graphs."""
    question_lower = question.lower()
    chart_keywords = [
        # Original keywords
        'show me the chart', 'show me the graph', 'show me the figure',
        'display the chart', 'display the graph', 'display the figure',
        'find the chart', 'find the graph', 'find the figure',
        'get the chart', 'get the graph', 'get the figure',
        'chart data', 'graph data', 'figure data',
        'show chart', 'show graph', 'show figure',
        'what does the chart show', 'what does the graph show',
        'chart information', 'graph information',
        'extract chart', 'extract graph', 'extract figure',
        'convert chart', 'convert graph', 'convert figure',
        
        # Additional common phrases
        'give me a graph', 'give me a chart', 'give me a figure',
        'create a graph', 'create a chart', 'create a figure',
        'make a graph', 'make a chart', 'make a figure',
        'generate a graph', 'generate a chart', 'generate a figure',
        'plot a graph', 'plot a chart', 'plot a figure',
        'draw a graph', 'draw a chart', 'draw a figure',
        'visualize the data', 'visualize data',
        'chart the data', 'graph the data',
        'show me a visualization', 'create a visualization',
        'data visualization', 'financial chart', 'financial graph',
        'revenue chart', 'revenue graph', 'earnings chart', 'earnings graph',
        'projection chart', 'projection graph', 'forecast chart', 'forecast graph'
    ]
    return any(keyword in question_lower for keyword in chart_keywords)



def is_audio_question(question: str) -> bool:
    """Detect if the question is about audio content."""
    question_lower = question.lower()
    audio_keywords = [
        'audio', 'recording', 'transcript', 'transcription', 'voice', 'speech',
        'said', 'mentioned', 'talked about', 'discussed', 'conversation',
        'interview', 'podcast', 'meeting', 'call', 'recording'
    ]
    return any(keyword in question_lower for keyword in audio_keywords)

def _is_french_question(text: str) -> bool:
    """Detect if the question is in French."""
    text_lower = text.lower()
    
    # More specific French words and phrases
    french_words = [
        'comment', 'pourquoi', 'quand', 'où', 'qui', 'quoi', 'quoie', 'combien', 'quel', 'quelle', 'quels', 'quelles',
        'comment', 'pourquoi', 'quand', 'où', 'qui', 'quoi', 'quoie', 'combien', 'quel', 'quelle', 'quels', 'quelles',
        'est-ce', 'sont-ce', 'avez-vous', 'avez-vous', 'pouvez-vous', 'voulez-vous', 'allez-vous',
        'comment allez-vous', 'comment ça va', 'ça va', 'bonjour', 'salut', 'au revoir', 'merci',
        's\'il vous plaît', 's\'il te plaît', 'excusez-moi', 'désolé', 'pardon',
        'et', 'le', 'la', 'les', 'un', 'une', 'des', 'ou', 'avec', 'sur', 'dans', 'par', 'de', 'du'
    ]
    
    # French characters
    french_chars = ['é', 'è', 'ê', 'ë', 'à', 'â', 'ô', 'ù', 'û', 'ç', 'î', 'ï']
    
    # Check for French words (exact matches to avoid false positives)
    french_word_count = sum(1 for word in french_words if f' {word} ' in f' {text_lower} ' or text_lower.startswith(word) or text_lower.endswith(word))
    
    # Check for French characters
    french_char_count = sum(1 for char in french_chars if char in text)
    
    # Check for common French question patterns
    french_patterns = ['est-ce que', 'qu\'est-ce que', 'comment', 'pourquoi', 'quand', 'où']
    french_pattern_count = sum(1 for pattern in french_patterns if pattern in text_lower)
    
    total_score = french_word_count + french_char_count + french_pattern_count
    
    # Require at least 2 indicators to be more confident
    return total_score >= 2

def _translate_text(text: str, source_lang: str, target_lang: str) -> str:
    """Translate text between languages using Deep Translator."""
    try:
        from deep_translator import GoogleTranslator
        
        # For very long texts, use much smaller chunks
        if len(text) > 4000:  # More conservative limit
            # Split into much smaller chunks by sentences first, then words
            import re
            
            # Split by sentences first
            sentences = re.split(r'[.!?]+', text)
            chunks = []
            current_chunk = ""
            max_chunk_size = 2000  # Much smaller chunks for safety
            
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                    
                # If adding this sentence would exceed the limit, save current chunk and start new one
                if len(current_chunk + " " + sentence) > max_chunk_size:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = sentence
                else:
                    current_chunk += " " + sentence if current_chunk else sentence
            
            # Add the last chunk
            if current_chunk:
                chunks.append(current_chunk.strip())
            
            # If any chunk is still too long, split it further by words
            final_chunks = []
            for chunk in chunks:
                if len(chunk) > max_chunk_size:
                    # Split by words
                    words = chunk.split()
                    word_chunk = ""
                    for word in words:
                        if len(word_chunk + " " + word) > max_chunk_size:
                            if word_chunk:
                                final_chunks.append(word_chunk.strip())
                            word_chunk = word
                        else:
                            word_chunk += " " + word if word_chunk else word
                    if word_chunk:
                        final_chunks.append(word_chunk.strip())
                else:
                    final_chunks.append(chunk)
            
            # Translate each chunk
            translator = GoogleTranslator(source=source_lang, target=target_lang)
            translated_chunks = []
            
            for i, chunk in enumerate(final_chunks):
                try:
                    # Add a delay between chunks to avoid rate limiting
                    if i > 0:
                        import time
                        time.sleep(0.2)  # Longer delay for safety
                    
                    # Skip empty chunks
                    if not chunk.strip():
                        continue
                        
                    translated_chunk = translator.translate(chunk)
                    translated_chunks.append(translated_chunk)
                except Exception as chunk_error:
                    logger.error(f"Error translating chunk {i} (length: {len(chunk)}): {str(chunk_error)}")
                    # If chunk is still too long, try to split it even further
                    if "Text length need to be between 0 and 5000 characters" in str(chunk_error):
                        # Split into even smaller pieces
                        sub_chunks = [chunk[j:j+1500] for j in range(0, len(chunk), 1500)]
                        for sub_chunk in sub_chunks:
                            try:
                                translated_sub_chunk = translator.translate(sub_chunk)
                                translated_chunks.append(translated_sub_chunk)
                                time.sleep(0.2)
                            except Exception as sub_error:
                                logger.error(f"Error translating sub-chunk: {str(sub_error)}")
                                translated_chunks.append(sub_chunk)  # Keep original
                    else:
                        translated_chunks.append(chunk)  # Keep original if translation fails
            
            return ' '.join(translated_chunks)
        else:
            # Text is short enough, translate normally
            translator = GoogleTranslator(source=source_lang, target=target_lang)
            result = translator.translate(text)
            return result

    except Exception as e:
        logger.error(f"Error translating from {source_lang} to {target_lang}: {str(e)}")
        return text  # Return original text if translation fails

def _translate_to_french(text: str) -> str:
    """Translate English text to French (for backward compatibility)."""
    return _translate_text(text, 'en', 'fr')

def _translate_to_english(text: str) -> str:
    """Translate French text to English."""
    return _translate_text(text, 'fr', 'en')



# Update the submit_chat_message function
def submit_chat_message():
    chat_input_key = f"chat_input_{st.session_state.chat_input_key}"
    chat_question = st.session_state.get(chat_input_key, st.session_state['question_input'])
    analysis_mode = st.session_state.get('analysis_mode', 'General')
    
    # Check if this question has already been processed
    if chat_question.strip() == st.session_state.get('last_processed_question', ""):
        return  # Skip if already processed
    
    if chat_question.strip():
        # Store the question to prevent duplicate processing
        st.session_state.last_processed_question = chat_question.strip()
        
        # Check if this is a follow-up to an image analysis
        conversation_history = st.session_state.rag_system.get_conversation_history()
        is_image_followup = False
        if conversation_history:
            # Look for any recent image analysis in the last few messages
            # Check the last 3 messages to see if any were image analysis
            recent_messages = conversation_history[-3:] if len(conversation_history) >= 3 else conversation_history
            for msg in recent_messages:
                if msg.get('question_type') == 'image_analysis':
                    is_image_followup = True
                    break
        
        # If this is a follow-up to image analysis, use the follow-up handler
        if is_image_followup:
            with st.spinner("Processing follow-up question..."):
                answer = st.session_state.rag_system.handle_follow_up(chat_question, analysis_mode=analysis_mode)
            st.rerun()
            return
        
        # Check if this is a chart request
        elif is_chart_request(chat_question):
            # Process chart request with progress indicator
            with st.spinner("Converting PDF pages to images..."):
                chart_results = st.session_state.rag_system.process_chart_request(chat_question, analysis_mode=analysis_mode)
            
            if not chart_results:
                st.warning("No charts found or error processing charts")
            else:
                st.success(f"Found {len(chart_results)} chart(s)")
                
                # Display each chart image
                for i, chart_info in enumerate(chart_results):
                    st.subheader(f"Chart {i+1} - Page {chart_info['page']}")
                    
                    # Get the image path
                    image_path = chart_info.get('image_path')
                    
                    # Debug info
                    st.write(f"Debug: Looking for image at: {image_path}")
                    st.write(f"Debug: File exists: {os.path.exists(image_path) if image_path else False}")
                    
                    if image_path and os.path.exists(image_path):
                        try:
                            # Display the image
                            with open(image_path, "rb") as img_file:
                                st.image(img_file, caption=chart_info.get('description', f"Page {chart_info['page']}"))
                            st.write(f"Successfully displayed image from {image_path}")
                        except Exception as e:
                            st.error(f"Error displaying image: {str(e)}")
                    else:
                        st.error(f"Image file not found: {image_path}")
                        # List files in the directory to debug
                        if image_path:
                            dir_path = os.path.dirname(image_path)
                            if os.path.exists(dir_path):
                                files = os.listdir(dir_path)
                                st.write(f"Files in directory {dir_path}: {files}")
            
            # Add to conversation history for chart requests
            st.session_state.rag_system.add_to_conversation_history(chat_question, f"Displayed {len(chart_results)} charts", "chart_request", "document")
            st.rerun()
            
        # Check if this is an image request
        elif is_image_request(chat_question):
            # Process image request with visual feedback
            progress_placeholder = st.empty()
            progress_placeholder.info("Searching for images in document...")
            
            try:
                answer = st.session_state.rag_system.process_image_request(chat_question, analysis_mode=analysis_mode)
                progress_placeholder.empty()
                
                if "No images found" in answer:
                    st.warning("No images found in the document")
                elif "Error" in answer:
                    st.error("Error processing images")
                else:
                    st.success("Image analysis completed!")
                
            except Exception as e:
                progress_placeholder.error(f"Error: {str(e)}")
                answer = f"Error processing image request: {str(e)}"
            
            # Add to conversation history for image requests
            st.session_state.rag_system.add_to_conversation_history(chat_question, answer, "image_request", "document")
            st.rerun()
            
        # Check if documents are loaded OR internet mode is enabled
        elif not st.session_state.get('documents_loaded', False) and not st.session_state.get('internet_mode', False):
            # Quick message for no documents and no internet mode
            answer = "Please upload a document first or enable Live Web Search."
            st.session_state.rag_system.add_to_conversation_history(chat_question, answer, "error", "document")
            st.rerun()
            
        else:
            # Add loading indicator for all question processing
            with st.spinner("Thinking..."):
                # Check if question is in French
                is_french = _is_french_question(chat_question)
                
                # Process based on mode - simplified logic
                if st.session_state.get('internet_mode', False):
                    # Use live web search - this method already adds to conversation history
                    answer = st.session_state.rag_system.process_live_web_question(chat_question)
                else:
                    # Use document search - this method already adds to conversation history
                    answer = st.session_state.rag_system.process_question_with_mode(chat_question, normalize_length=True, analysis_mode=analysis_mode)
                
                # REMOVE THIS TRANSLATION - The QuestionHandler already generates French responses
                # if is_french:
                #     answer = _translate_to_french(answer)
                #     # Update the conversation history with the translated answer
                #     conversation_history = st.session_state.rag_system.get_conversation_history()
                #     if conversation_history:
                #         conversation_history[-1]['answer'] = answer
            
            # Don't add to conversation history here since the RAG methods already do it
            st.rerun()
    
    # Only increment after rerun, so the key stays in sync
    st.session_state.chat_input_key += 1

# Show current mode in the placeholder
current_mode = "Internet Search" if st.session_state.get('internet_mode', False) else "Document Search"
placeholder_text = f"Ask a question (using {current_mode})..."

# Add this function definition before the sidebar code
def process_image_analysis(uploaded_image, analysis_type, custom_question=""):
    """Process uploaded image for analysis."""
    try:
        # Save uploaded image temporarily
        temp_image_path = f"temp_uploaded_image_{int(time.time())}.png"
        with open(temp_image_path, "wb") as f:
            f.write(uploaded_image.getbuffer())
        
        # Process based on analysis type
        if analysis_type == "General":
            question = "Please analyze this image and describe what you see in detail."
        elif analysis_type == "OCR":
            question = "Please extract and transcribe all text visible in this image."
        elif analysis_type == "Charts":
            question = "Please analyze this chart or graph and extract the data, trends, and key insights."
        elif analysis_type == "Custom":
            question = custom_question if custom_question else "Please analyze this image."
        
        # Process with GPT-4 Vision
        with st.spinner("Analyzing image..."):
            answer = st.session_state.rag_system.analyze_image_with_gpt4(temp_image_path, question)
            
            # Add to conversation history with just the answer (no question prefix)
            st.session_state.rag_system.add_to_conversation_history(
                f"Image Analysis ({analysis_type})",
                answer,
                "image_analysis",
                "image"
            )
        
        # Clean up temp file
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)
        
        st.success("Image analysis completed!")
        st.rerun()
        
    except Exception as e:
        st.error(f"Error analyzing image: {str(e)}")
        logger.error(f"Image analysis error: {str(e)}")

# Sidebar - Clean, organized controls
with st.sidebar:
    # Document Management Section
    st.subheader("Documents")
    
    # File uploader for individual files
    uploaded_files = st.file_uploader(
        "Upload files",
        type=['pdf', 'doc', 'docx', 'xls', 'xlsx', 'ppt', 'pptx', 'mp3', 'wav', 'm4a', 'flac', 'ogg', 'aac'],
        accept_multiple_files=True,
        key="pdf_uploader"
    )
    
    # Folder uploader (ZIP files)
    uploaded_zip = st.file_uploader(
        "Upload folder (ZIP file)",
        type=['zip'],
        accept_multiple_files=False,
        key="zip_uploader",
        help="Upload a ZIP file containing multiple documents. The ZIP will be extracted and all supported files will be processed."
    )
    
    # Process uploaded files and ZIP
    all_files_to_process = []
    
    # Add individual files
    if uploaded_files:
        all_files_to_process.extend(uploaded_files)
    
    # Process ZIP file if uploaded
    if uploaded_zip:
        import zipfile
        import tempfile
        import io
        
        try:
            with st.spinner("Extracting ZIP file..."):
                # Check ZIP file size to prevent memory issues
                zip_size_mb = uploaded_zip.size / (1024 * 1024)
                if zip_size_mb > 500:  # 500MB warning for 1GB Streamlit memory
                    st.warning(f"Large ZIP file detected ({zip_size_mb:.1f}MB). For optimal performance with 1GB memory limit, keep ZIP files under 500MB.")
                if zip_size_mb > 800:  # Hard limit
                    st.error(f"ZIP file too large ({zip_size_mb:.1f}MB). Maximum allowed size is 800MB to prevent memory issues.")
                    st.stop()
                
                # Create a temporary directory to extract files
                with tempfile.TemporaryDirectory() as temp_dir:
                    # Extract ZIP contents
                    with zipfile.ZipFile(uploaded_zip, 'r') as zip_ref:
                        zip_ref.extractall(temp_dir)
                    
                    # Walk through extracted files and find supported types
                    supported_extensions = {'.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx', 
                                          '.mp3', '.wav', '.m4a', '.flac', '.ogg', '.aac'}
                    
                    extracted_files = []
                    for root, dirs, files in os.walk(temp_dir):
                        for file in files:
                            file_path = os.path.join(root, file)
                            file_ext = os.path.splitext(file)[1].lower()
                            
                            if file_ext in supported_extensions:
                                # Create a memory-efficient file-like object
                                # Don't load entire file into memory - use file path instead
                                class FileWrapper:
                                    def __init__(self, file_path, filename):
                                        self.file_path = file_path
                                        self.name = filename
                                        self._file = None
                                    
                                    def getbuffer(self):
                                        # Only read when needed, not during extraction
                                        if self._file is None:
                                            with open(self.file_path, 'rb') as f:
                                                return f.read()
                                        return self._file.read()
                                    
                                    def seek(self, pos):
                                        if self._file is None:
                                            self._file = open(self.file_path, 'rb')
                                        self._file.seek(pos)
                                    
                                    def close(self):
                                        if self._file:
                                            self._file.close()
                                            self._file = None
                                
                                file_obj = FileWrapper(file_path, file)
                                extracted_files.append(file_obj)
                    
                    if extracted_files:
                        # Limit number of files to prevent memory issues
                        max_files = 50  # Reasonable limit
                        if len(extracted_files) > max_files:
                            st.warning(f"Large ZIP detected with {len(extracted_files)} files. Processing first {max_files} files to prevent memory issues.")
                            extracted_files = extracted_files[:max_files]
                        
                        st.success(f"Extracted {len(extracted_files)} supported files from ZIP")
                        all_files_to_process.extend(extracted_files)
                    else:
                        st.warning("No supported files found in ZIP")
                        
        except Exception as e:
            st.error(f"Error processing ZIP file: {str(e)}")
            logger.error(f"ZIP processing error: {str(e)}")
    
    # Process all files (individual + extracted from ZIP)
    if all_files_to_process:
        current_files = [f.name for f in all_files_to_process]
        last_files = st.session_state.get('last_uploaded_files', [])
        
        # Check if files are actually new and not currently processing
        new_files = [f for f in current_files if f not in last_files]
        is_processing = st.session_state.get('is_processing_files', False)
        
        if (new_files or not st.session_state.get('documents_loaded')) and not is_processing:
            st.session_state.is_processing_files = True
            st.session_state.last_uploaded_files = current_files
            st.session_state.last_upload_time = time.time()
            
            # Check for audio files
            audio_files = [f for f in all_files_to_process if f.name.lower().endswith(('.mp3', '.wav', '.m4a', '.flac', '.ogg', '.aac'))]
            document_files = [f for f in all_files_to_process if f.name.lower().endswith(('.pdf', '.docx', '.pptx', '.doc', '.xls', '.xlsx', '.ppt'))]
            
            if audio_files:
                st.info(f"Found {len(audio_files)} audio file(s) - will transcribe to text")
            
            # Process files in smaller batches to avoid memory issues
            batch_size = 3  # Process 3 files at a time
            success_count = 0
            
            with st.spinner(f"Processing {len(all_files_to_process)} files..."):
                for i in range(0, len(all_files_to_process), batch_size):
                    batch = all_files_to_process[i:i+batch_size]
                    batch_names = [f.name for f in batch]
                    
                    st.info(f"Processing batch {i//batch_size + 1}: {', '.join(batch_names)}")
                    
                    try:
                        if st.session_state.rag_system.process_web_uploads(batch):
                            success_count += len(batch)
                            st.success(f"Processed {len(batch)} files successfully")
                        else:
                            st.error(f"Failed to process batch: {', '.join(batch_names)}")
                    except Exception as e:
                        st.error(f"Error processing batch: {str(e)}")
                        logger.error(f"Error processing batch: {str(e)}")
            
            if success_count > 0:
                st.success(f"Successfully processed {success_count}/{len(all_files_to_process)} files")
                st.session_state.documents_loaded = True
                st.session_state.processing_status = st.session_state.rag_system.file_handler.get_processing_status()
            else:
                st.error("Failed to process any files")
            
            st.session_state.is_processing_files = False
        else:
            # Files haven't changed or currently processing, just show status
            if st.session_state.get('documents_loaded'):
                st.info(f"{len(current_files)} file(s) already loaded")

    # Search Mode Section
    st.markdown("---")
    st.subheader("Search Mode")
    
    # Add Analysis Perspective selector
    st.subheader("Analysis Perspective")
    new_analysis_mode = st.selectbox(
        "Choose analysis perspective:",
        ["General", "Financial Document", "Company Evaluation", "Legal Document", "Financial Excel Document"],
        index=0 if 'analysis_mode' not in st.session_state else ["General", "Financial Document", "Company Evaluation", "Legal Document", "Financial Excel Document"].index(st.session_state["analysis_mode"]),
        help="Choose the lens through which the document will be analyzed."
    )
    if st.button("Save Analysis Mode", key="save_analysis_mode"):
        st.session_state["analysis_mode"] = new_analysis_mode
        st.success(f"Analysis mode set to: {new_analysis_mode}")
    elif "analysis_mode" not in st.session_state:
        st.session_state["analysis_mode"] = new_analysis_mode

    search_mode = st.selectbox(
        "Choose search mode:",
        ["Documents", "Live Web Search", "Both"],
        index=0,
        help="Live Web Search uses DuckDuckGo for real-time information"
    )

    # Track previous search mode and force rerun on change
    if 'previous_search_mode' not in st.session_state:
        st.session_state.previous_search_mode = search_mode

    # If search mode changed, update RAG system FIRST, then force rerun
    if st.session_state.previous_search_mode != search_mode:
        # Update the RAG system's internet mode BEFORE rerun
        if search_mode == "Live Web Search":
            st.session_state.rag_system.set_internet_mode(True)
            st.session_state.internet_mode = True  # Also update session state
        elif search_mode == "Documents":
            st.session_state.rag_system.set_internet_mode(False)
            st.session_state.internet_mode = False  # Also update session state
        elif search_mode == "Both":
            st.session_state.rag_system.set_internet_mode(True)
            st.session_state.internet_mode = True  # Also update session state
            st.session_state.use_both_modes = True
        
        st.session_state.previous_search_mode = search_mode
        st.rerun()

    # Now set the internet mode based on current selection
    if search_mode == "Live Web Search":
        st.session_state.internet_mode = True
        st.success("Live Web Search Enabled")
    elif search_mode == "Documents":
        st.session_state.internet_mode = False
        if st.session_state.get('documents_loaded'):
            st.info(f"Document Mode ({len(st.session_state.get('last_uploaded_files', []))} docs)")
        else:
            st.warning("No documents loaded")
    elif search_mode == "Both":
        st.session_state.internet_mode = True
        st.session_state.use_both_modes = True
        st.success("Both Modes Enabled")

    # Export Section (only show if there's conversation history)
    if conversation_history:
        st.markdown("---")
        st.subheader("Export")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("PDF", use_container_width=True):
                pdf_path = export_conversation_to_pdf()
                if pdf_path and os.path.exists(pdf_path):
                    with open(pdf_path, "rb") as pdf_file:
                        pdf_bytes = pdf_file.read()
                    st.download_button(
                        label="Download PDF",
                        data=pdf_bytes,
                        file_name="conversation.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
        
        with col2:
            if st.button("Markdown", use_container_width=True):
                md_path = export_conversation_to_markdown()
                if md_path and os.path.exists(md_path):
                    with open(md_path, "r", encoding='utf-8') as md_file:
                        md_content = md_file.read()
                    st.download_button(
                        label="Download Markdown",
                        data=md_content,
                        file_name="conversation.md",
                        mime="text/markdown",
                        use_container_width=True
                    )
        
        with col3:
            if st.button("CSV", use_container_width=True):
                csv_path = export_conversation_to_csv()
                if csv_path and os.path.exists(csv_path):
                    with open(csv_path, "r", encoding='utf-8') as csv_file:
                        csv_content = csv_file.read()
                    st.download_button(
                        label="Download CSV",
                        data=csv_content,
                        file_name="conversation.csv",
                        mime="text/csv",
                        use_container_width=True
                    )

    # Add transcription export section
    st.markdown("---")
    st.subheader("Audio Transcriptions")
    
    # Check if there are any audio transcriptions available
    if hasattr(st.session_state.rag_system, 'file_handler') and st.session_state.rag_system.file_handler:
        saved_paths = st.session_state.rag_system.file_handler.get_saved_pdf_paths()
        audio_transcripts = [path for path in saved_paths if '_transcript.txt' in path]
        
        if audio_transcripts:
            st.info(f"Found {len(audio_transcripts)} audio transcription(s)")
            
            for transcript_path in audio_transcripts:
                try:
                    # Read the transcription
                    with open(transcript_path, 'r', encoding='utf-8') as f:
                        transcription_text = f.read()
                    
                    # Get original filename
                    base_name = os.path.basename(transcript_path).replace('_transcript.txt', '')
                    
                    # Create export button
                    if st.button(f"Export {base_name} PDF", key=f"export_{base_name}"):
                        pdf_path = export_transcription_to_pdf(transcription_text, base_name)
                        if pdf_path and os.path.exists(pdf_path):
                            with open(pdf_path, "rb") as pdf_file:
                                pdf_bytes = pdf_file.read()
                            st.download_button(
                                label=f"Download {base_name} PDF",
                                data=pdf_bytes,
                                file_name=f"{base_name}_transcription.pdf",
                                mime="application/pdf",
                                use_container_width=True,
                                key=f"download_{base_name}"
                            )
                        else:
                            st.error("Failed to create PDF")
                            
                except Exception as e:
                    st.error(f"Error reading transcription {transcript_path}: {str(e)}")
        else:
            st.info("No audio transcriptions found. Upload an audio file to create transcriptions.")
    else:
        st.info("No file handler available. Upload an audio file first.")

    # Session Management Section
    st.markdown("---")
    st.subheader("Session")
    
    # Only show the Reset button, full width
    if st.button("Reset", type="secondary", use_container_width=True):
        st.session_state.rag_system.clear_conversation_history()
        if 'answer_given' in st.session_state:
            del st.session_state.answer_given
        # Clean up files
        if hasattr(st.session_state.rag_system.file_handler, 'get_saved_pdf_paths'):
            saved_paths = st.session_state.rag_system.file_handler.get_saved_pdf_paths()
            for pdf_path in saved_paths:
                if os.path.exists(pdf_path):
                    try:
                        os.remove(pdf_path)
                    except Exception as e:
                        logger.error(f"Could not remove {pdf_path}: {e}")
        if os.path.exists("images"):
            import shutil
            try:
                shutil.rmtree("images")
            except Exception as e:
                logger.error(f"Could not clean up images directory: {e}")
        st.session_state.rag_system.reset_performance_metrics()
        st.session_state.error_count = 0
        st.session_state.performance_metrics = {}
        st.session_state.documents_loaded = False
        st.session_state.last_uploaded_files = []
        st.session_state.processing_status = {}
        st.success("Session reset!")

    # Translation Section
    if conversation_history:
        st.markdown("---")
        st.subheader("Translation")
        
        # Get the last answer for translation
        last_answer = conversation_history[-1]['answer'] if conversation_history else ""
        
        if last_answer:
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("🇫🇷 To French", use_container_width=True, key="translate_to_french"):
                    with st.spinner("Translating to French..."):
                        translated = _translate_to_french(last_answer)
                        # Add translation to conversation history
                        st.session_state.rag_system.add_to_conversation_history(
                            "Translate to French",
                            f"**Translation to French:**\n\n{translated}",
                            "translation_request",
                            "document"
                        )
                    st.success("Translated to French!")
                    st.rerun()
            
            with col2:
                if st.button("🇬🇧 To English", use_container_width=True, key="translate_to_english"):
                    with st.spinner("Translating to English..."):
                        translated = _translate_to_english(last_answer)
                        # Add translation to conversation history
                        st.session_state.rag_system.add_to_conversation_history(
                            "Translate to English",
                            f"**Translation to English:**\n\n{translated}",
                            "translation_request",
                            "document"
                        )
                    st.success("Translated to English!")
                    st.rerun()
        else:
            st.info("No answer to translate. Ask a question first.")

    # Compact Performance Section
    if conversation_history or st.session_state.get('performance_metrics'):
        st.markdown("---")
        st.subheader("Stats")
        
        try:
            metrics = st.session_state.rag_system.get_performance_metrics()
            stats = st.session_state.rag_system.question_handler.get_conversation_stats()
            
            # Display key metrics in a compact format
            col1, col2 = st.columns(2)
            with col1:
                if metrics:
                    st.metric("Queries", metrics.get('total_queries', 0))
                if conversation_history:
                    st.metric("Messages", len(conversation_history))
            
            with col2:
                if metrics:
                    st.metric("Errors", metrics.get('error_count', 0))
                if st.session_state.get('documents_loaded'):
                    st.metric("Docs", len(st.session_state.get('last_uploaded_files', [])))
            
            # Show response time if available
            if st.session_state.performance_metrics.get('last_response_time'):
                st.caption(f"Last response: {st.session_state.performance_metrics['last_response_time']:.2f}s")
                
        except Exception as e:
            st.error(f"Metrics error: {e}")

    # Add this test function to your sidebar
    def test_image_extraction():
        """Test if image extraction actually works."""
        try:
            uploaded_files = st.session_state.get('last_uploaded_files', [])
            if not uploaded_files:
                return "No documents uploaded"
            
            pdf_path = os.path.join("temp", uploaded_files[0])
            if not os.path.exists(pdf_path):
                return "Document not found in temp directory"
            
            # Actually extract images
            processor = OnDemandImageProcessor()
            images = processor.extract_images_from_pdf(pdf_path)
            
            if images:
                # Show the actual images found
                st.write(f"Found {len(images)} images:")
                for i, img_info in enumerate(images, 1):
                    st.write(f"  - Image {i}: Page {img_info['page']} ({img_info['path']})")
                
                # Show first image as preview
                if os.path.exists(images[0]['path']):
                    st.image(images[0]['path'], caption=f"Preview: {images[0]['description']}", width=300)
                
                return f"Successfully extracted {len(images)} images"
            else:
                return "No images found in document (PDF may only contain text)"
            
        except Exception as e:
            return f"Error: {str(e)}"

    # Add this to your sidebar
    if st.button("Test Image Extraction", use_container_width=True):
        result = test_image_extraction()
        st.info(result)

    # Image Analysis Section - moved to sidebar
    st.markdown("---")
    st.subheader("Image Analysis")
    
    uploaded_image = st.file_uploader(
        "Upload image",
        type=['png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'],
        key="image_uploader",
        help="Upload an image to analyze"
    )
    
    analysis_type = st.selectbox(
        "Analysis type",
        ["General", "OCR", "Charts", "Custom"],
        help="Choose analysis type"
    )
    
    # Show image preview and custom question only when image is uploaded
    if uploaded_image is not None:
        # Display image with compact styling for sidebar
        st.image(uploaded_image, caption="", width=200)
        
        # Custom question only if needed
        custom_question = ""
        if analysis_type == "Custom":
            custom_question = st.text_input(
                "Ask about this image:",
                placeholder="e.g., What data does this chart show?",
                help="Enter your specific question"
            )
        
        # Clean analyze button
        if st.button("Analyze", use_container_width=True, type="primary"):
            if uploaded_image is not None:
                process_image_analysis(uploaded_image, analysis_type, custom_question)
            else:
                st.warning("Please upload an image first")

# Minimal error display in main area
if st.session_state.error_count > 0:
    st.error(f"{st.session_state.error_count} error(s) - check sidebar for details.")

# Modern chat input - always visible
chat_input_key = f"chat_input_{st.session_state.chat_input_key}"
chat_question = st.chat_input(
    placeholder_text,
    key=chat_input_key
)

if chat_question:
    submit_chat_message()



def install_system_dependencies():
    """Install system dependencies if needed."""
    try:
        # Check if ffmpeg is available
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        print("FFmpeg is available")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("FFmpeg not found. Audio processing may not work properly.")
        print("To install FFmpeg:")
        print("  Ubuntu/Debian: sudo apt-get install ffmpeg")
        print("  macOS: brew install ffmpeg")
        print("  Windows: Download from https://ffmpeg.org/download.html")

# Call this function when the app starts
if __name__ == "__main__":
    install_system_dependencies() 
