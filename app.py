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

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set page config
st.set_page_config(
    page_title="HERON",
    page_icon="🦅",
    layout="wide"
)

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

# Handle image display requests with improved error handling
def handle_image_request(question):
    """Handle requests to show images."""
    try:
        all_images = st.session_state.rag_system.get_all_images()
        
        if not all_images:
            return []
        
        # Extract page number if mentioned
        import re
        page_match = re.search(r'page\s+(\d+)', question.lower())
        
        if page_match:
            page_num = int(page_match.group(1))
            # Show images from specific page
            page_images = [img_info for img_info in all_images.values() if img_info['page'] == page_num]
            return page_images
        else:
            # Show all images
            return list(all_images.values())
    except Exception as e:
        st.session_state.error_count += 1
        logger.error(f"Error handling image request: {str(e)}")
        return []

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
                # If the answer is a list (image info), display images
                if isinstance(conv['answer'], list):
                    if conv['answer']:
                        img_info = conv['answer'][0]  # Only show the most relevant image
                        if os.path.exists(img_info['path']):
                            # Display image with enhanced caption
                            caption = f"Page {img_info['page']}, Image {img_info['image_num']}"
                            if 'description' in img_info:
                                caption += f" - {img_info['description']}"
                            if 'similarity_score' in img_info:
                                caption += f" (Similarity: {img_info['similarity_score']:.2f})"
                            st.image(img_info['path'], caption=caption, use_container_width=True)
                    else:
                        st.write("No images were found in the uploaded documents.")
                else:
                    # Highlight relevant text if available
                    answer = conv['answer']
                    highlight = conv.get('highlight')
                    if highlight and highlight in answer:
                        # Use HTML <mark> for highlighting
                        answer = answer.replace(highlight, f'<mark>{highlight}</mark>')
                        st.markdown(answer, unsafe_allow_html=True)
                    else:
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

                # Show source image automatically if user's question matches a trigger phrase and chunk metadata is present
                source = conv.get('source')
                page = conv.get('page')
                chunk_text = conv.get('chunk_text')
                user_question = conv.get('question', '').lower()
                trigger_phrases = [
                    'show me the source',
                    'show me the graph',
                    'show me the chart',
                    'show me the figure',
                    'show me the table',
                    'show me the image',
                    'show the source',
                    'show the graph',
                    'show the chart',
                    'show the figure',
                    'show the table',
                    'show the image',
                    'source for this info',
                    'graph for this info',
                    'chart for this info',
                    'figure for this info',
                    'table for this info',
                    'image for this info',
                ]
                if source and page and chunk_text and any(phrase in user_question for phrase in trigger_phrases):
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
    # Center the logo using columns
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image("heron.png", width=120)
    
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
