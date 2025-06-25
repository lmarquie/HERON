import streamlit as st
import os
from local_draft import RAGSystem, WebFileHandler, render_chunk_source_image, get_page_image_simple
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
import fitz  # PyMuPDF for fast page extraction

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set page config
st.set_page_config(
    page_title="HERON",
    page_icon="🦅",
    layout="wide"
)

def clean_answer_text(answer):
    # Add a space between a number and a letter if they are stuck together
    answer = re.sub(r'(\d)([a-zA-Z])', r'\1 \2', answer)
    answer = re.sub(r'([a-zA-Z])(\d)', r'\1 \2', answer)
    # Replace multiple spaces with a single space
    answer = re.sub(r'\s+', ' ', answer)
    # Optionally, remove stray markdown italics/asterisks
    answer = answer.replace('*', '')
    return answer

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
        
        # Use the new mode-aware question processing
        answer = st.session_state.rag_system.process_question_with_mode(question, normalize_length=True)
        
        # Update performance metrics
        response_time = time.time() - start_time
        st.session_state.performance_metrics['last_response_time'] = response_time
        
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
        for i, conv in enumerate(conversation_history):
            # Question
            question_style = ParagraphStyle(
                'Question',
                parent=styles['Normal'],
                fontSize=12,
                spaceAfter=6,
                leftIndent=20
            )
            story.append(Paragraph(f"<b>Q{i+1}:</b> {conv['question']}", question_style))
            
            # Answer
            if isinstance(conv['answer'], list):
                # Handle image answers
                answer_text = "Found 1 image" if conv['answer'] else "No images found"
                story.append(Paragraph(f"<b>A{i+1}:</b> {answer_text}", question_style))
            else:
                # Handle text answers
                story.append(Paragraph(f"<b>A{i+1}:** {conv['answer']}", question_style))
            
            story.append(Spacer(1, 12))
        
        # Build PDF
        doc.build(story)
        
        return pdf_path
        
    except Exception as e:
        logger.error(f"Error creating PDF: {str(e)}")
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
            mode_icon = "🌐" if conv.get('mode') == 'internet' else "📄"
            mode_text = "Internet" if conv.get('mode') == 'internet' else "Document"
            
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
                    # Highlight relevant text if available
                    answer = conv['answer']
                    highlight = conv.get('highlight')
                    if highlight and highlight in answer:
                        # Use HTML <mark> for highlighting
                        answer = answer.replace(highlight, f'<mark>{highlight}</mark>')
                        answer = clean_answer_text(answer)
                        st.markdown(answer, unsafe_allow_html=True)
                    else:
                        answer = clean_answer_text(answer)
                        st.write(answer)
                    
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
                chunk_id = conv.get('chunk_id')
                total_chunks = conv.get('total_chunks')
                
                # Debug: Show what metadata we have
                if st.session_state.get('debug_mode', False):
                    st.write(f"**Debug Info:** Source={source}, Page={page}, Chunk ID={chunk_id}, Total Chunks={total_chunks}")
                
                if source and page and chunk_text:
                    if question_type == 'source_request':
                        # Automatically show source for source requests
                        # Use fast on-demand extraction
                        img_path = get_page_image_fast(source, page)
                        
                        if img_path and os.path.exists(img_path):
                            caption = f"Page {page}"
                            if chunk_id is not None:
                                caption += f" (Chunk {chunk_id}"
                                if total_chunks is not None:
                                    caption += f" of {total_chunks}"
                                caption += ")"
                            caption += " - Source Page"
                            st.image(img_path, caption=caption, use_container_width=True)
                        else:
                            st.warning(f"Could not extract page {page} from {source}")
                            # Show available files for debugging
                            if st.session_state.get('debug_mode', False):
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
                            # Use fast on-demand extraction
                            img_path = get_page_image_fast(source, page)
                            
                            if img_path and os.path.exists(img_path):
                                caption = f"Page {page}"
                                if chunk_id is not None:
                                    caption += f" (Chunk {chunk_id}"
                                    if total_chunks is not None:
                                        caption += f" of {total_chunks}"
                                    caption += ")"
                                caption += " - Source Page"
                                st.image(img_path, caption=caption, use_container_width=True)
                            else:
                                st.warning(f"Could not extract page {page} from {source}")

# Always show chat input (permanent chat interface)
# Initialize input key counter
if 'chat_input_key_counter' not in st.session_state:
    st.session_state.chat_input_key_counter = 0

def submit_chat_message():
    chat_input_key = f"chat_input_{st.session_state.chat_input_key_counter}"
    chat_question = st.session_state.get(chat_input_key, "")
    if chat_question.strip():
        # Check if documents are loaded first
        if not st.session_state.get('documents_loaded', False):
            # Quick message for no documents
            answer = "Please upload a document first."
            st.session_state.rag_system.add_to_conversation_history(chat_question, answer, "error", "document")
            st.rerun()
        else:
            # Check if this is a source/graph request
            question_lower = chat_question.lower()
            trigger_phrases = [
                'show me the source', 'show me the graph', 'show me the chart', 'show me the figure',
                'show me the table', 'show me the image', 'show the source', 'show the graph',
                'show the chart', 'show the figure', 'show the table', 'show the image',
                'source for this', 'graph for this', 'chart for this', 'figure for this',
                'table for this', 'image for this'
            ]
            
            is_source_request = any(phrase in question_lower for phrase in trigger_phrases)
            
            if is_source_request:
                # For source requests, always use the most recent chunk from QuestionHandler
                question_handler_history = st.session_state.rag_system.question_handler.conversation_history
                
                if question_handler_history:
                    # Get the most recent conversation entry with chunk metadata
                    most_recent = question_handler_history[-1]
                    if most_recent.get('source') and most_recent.get('chunk_text'):
                        # Use the page number directly from metadata
                        page_num = most_recent.get('page')
                        chunk_text = most_recent.get('chunk_text', '')
                        
                        # If no page number in metadata, default to 1 (but this shouldn't happen now)
                        if not page_num:
                            page_num = 1
                        
                        answer = f"Showing source from: {most_recent.get('source')}, Page {page_num}, Chunk {most_recent.get('chunk_id', '?')}"
                        # Create the entry manually to include chunk metadata
                        entry = {
                            'question': chat_question,
                            'answer': answer,
                            'question_type': 'source_request',
                            'mode': 'document',
                            'timestamp': datetime.now().isoformat(),
                            'source': most_recent.get('source'),
                            'page': page_num,
                            'chunk_text': chunk_text,
                            'chunk_id': most_recent.get('chunk_id'),
                            'total_chunks': most_recent.get('total_chunks')
                        }
                        st.session_state.rag_system.conversation_history.append(entry)
                    else:
                        answer = "No chunk metadata found in the most recent answer."
                        st.session_state.rag_system.add_to_conversation_history(chat_question, answer, "error", "document")
                else:
                    answer = "No previous conversation found. Please ask a question first."
                    st.session_state.rag_system.add_to_conversation_history(chat_question, answer, "error", "document")
            else:
                # Check if this is actually a follow-up question (has previous conversation)
                # Get the current conversation history to check if there are real Q&A pairs
                current_history = st.session_state.rag_system.get_conversation_history()
                has_real_conversation = any(
                    conv.get('question_type') not in ['error'] 
                    for conv in current_history
                )
                
                if has_real_conversation:
                    # Use follow-up processing for actual follow-up questions
                    answer = st.session_state.rag_system.process_follow_up_with_mode(chat_question, normalize_length=True)
                else:
                    # Use regular question processing for new questions
                    answer = st.session_state.rag_system.process_question_with_mode(chat_question, normalize_length=True)
            st.rerun()
    # Only increment after rerun, so the key stays in sync
    st.session_state.chat_input_key_counter += 1

# Show current mode in the placeholder
current_mode = "Internet Search" if st.session_state.get('internet_mode', False) else "Document Search"
placeholder_text = f"Ask a question (using {current_mode})..."

# Modern chat input - always visible
chat_input_key = f"chat_input_{st.session_state.chat_input_key_counter}"
chat_question = st.chat_input(
    placeholder_text,
    key=chat_input_key
)

if chat_question:
    submit_chat_message()

# Sidebar - Clean, organized controls
with st.sidebar:
    
    # Document Management Section
    st.subheader("Documents")
    
    # File uploader
    uploaded_files = st.file_uploader(
        "Upload files",
        type=['pdf', 'doc', 'docx', 'xls', 'xlsx', 'ppt', 'pptx'],
        accept_multiple_files=True,
        key="pdf_uploader"
    )
    
    # Process uploaded files
    if uploaded_files:
        current_files = [f.name for f in uploaded_files]
        last_files = st.session_state.get('last_uploaded_files', [])
        if current_files != last_files or not st.session_state.get('documents_loaded'):
            st.session_state.last_uploaded_files = current_files
            st.session_state.last_upload_time = time.time()
            with st.spinner("Processing..."):
                if st.session_state.rag_system.process_web_uploads(uploaded_files):
                    st.success(f"{len(uploaded_files)} file(s) loaded")
                    st.session_state.documents_loaded = True
                    st.session_state.processing_status = st.session_state.rag_system.file_handler.get_processing_status()
                else:
                    st.error("Processing failed")
                    st.session_state.documents_loaded = False

    # Search Mode Section
    st.markdown("---")
    st.subheader("Search Mode")
    
    internet_mode = st.toggle(
        "Internet Search",
        value=st.session_state.get('internet_mode', False),
        help="Search internet instead of documents"
    )

    # Update RAG system mode
    if 'rag_system' in st.session_state:
        st.session_state.rag_system.set_internet_mode(internet_mode)
        st.session_state.internet_mode = internet_mode

    # Show current mode
    if internet_mode:
        st.success("Internet Mode")
    else:
        if st.session_state.get('documents_loaded'):
            st.info(f"Document Mode ({len(st.session_state.get('last_uploaded_files', []))} docs)")
        else:
            st.warning("No documents loaded")

    # Session Management Section
    st.markdown("---")
    st.subheader("Session")
    
    # Debug toggle
    debug_mode = st.toggle(
        "Debug Mode",
        value=st.session_state.get('debug_mode', False),
        help="Show debug information to troubleshoot issues"
    )
    st.session_state.debug_mode = debug_mode
    
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

# Minimal error display in main area
if st.session_state.error_count > 0:
    st.error(f"{st.session_state.error_count} error(s) - check sidebar for details.")

# Fast on-demand page extraction
def get_page_image_fast(source_path, page_num):
    """Extract a single page image on demand - much faster than processing all pages.
    
    Args:
        source_path: Path to the PDF file
        page_num: Page number (1-based)
    
    Returns:
        Path to the page image file
    """
    try:
        # Try to find the PDF file
        pdf_path = source_path
        if not os.path.exists(pdf_path):
            # Try temp directory
            temp_path = os.path.join("temp", source_path)
            if os.path.exists(temp_path):
                pdf_path = temp_path
            else:
                # Try with just the filename
                filename = os.path.basename(source_path)
                temp_path = os.path.join("temp", filename)
                if os.path.exists(temp_path):
                    pdf_path = temp_path
                else:
                    return None
        
        # Extract only this specific page
        doc = fitz.open(pdf_path)
        page = doc.load_page(page_num - 1)  # 0-based index
        
        # Fast rendering with lower DPI
        pix = page.get_pixmap(dpi=120)  # Lower DPI for speed
        os.makedirs("temp", exist_ok=True)
        img_path = f"temp/page_{page_num}_fast.png"
        pix.save(img_path)
        doc.close()
        
        return img_path
        
    except Exception as e:
        logging.error(f"Error extracting page {page_num} from {source_path}: {str(e)}")
        return None 
