import streamlit as st
import os
from local_draft import RAGSystem, WebFileHandler
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from datetime import datetime
import time
import logging

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
    try:
        start_time = time.time()
        
        # Use the new mode-aware follow-up processing
        answer = st.session_state.rag_system.process_follow_up_with_mode(follow_up_question, normalize_length=True)
        
        # Update performance metrics
        response_time = time.time() - start_time
        st.session_state.performance_metrics['last_response_time'] = response_time
        
        return answer
        
    except Exception as e:
        st.session_state.error_count += 1
        logger.error(f"Error generating follow-up: {str(e)}")
        return f"Error: {str(e)}"

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
                story.append(Paragraph(f"<b>A{i+1}:</b> {conv['answer']}", question_style))
            
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

# Simple sidebar with improved file handling
with st.sidebar:
    st.header("Upload Documents")
    
    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=['pdf'],
        accept_multiple_files=True,
        key="pdf_uploader"
    )
    
    if uploaded_files:
        # Check if files are new (different from last upload)
        current_files = [f.name for f in uploaded_files]
        last_files = st.session_state.get('last_uploaded_files', [])
        
        if current_files != last_files or not st.session_state.get('documents_loaded'):
            st.session_state.last_uploaded_files = current_files
            st.session_state.last_upload_time = time.time()
            
            with st.spinner("Processing documents..."):
                if st.session_state.rag_system.process_web_uploads(uploaded_files):
                    st.success(f"Processed {len(uploaded_files)} file(s)")
                    st.session_state.documents_loaded = True
                    
                    # Get processing status
                    processing_status = st.session_state.rag_system.file_handler.get_processing_status()
                    st.session_state.processing_status = processing_status
                    
                    # Show processing details
                    if processing_status:
                        st.subheader("Processing Details:")
                        for filename, status in processing_status.items():
                            if status == "success":
                                st.success(f"✅ {filename}")
                            else:
                                st.error(f"❌ {filename}")
                else:
                    st.error("Failed to process files")
                    st.session_state.documents_loaded = False
        else:
            st.info("Documents already loaded. Upload new files to replace them.")

# Internet mode toggle
st.sidebar.markdown("---")
st.sidebar.header("🌐 Search Mode")
internet_mode = st.sidebar.toggle(
    "Internet Search Mode",
    value=st.session_state.get('internet_mode', False),
    help="Enable to search the internet instead of uploaded documents"
)

# Update RAG system internet mode
if 'rag_system' in st.session_state:
    st.session_state.rag_system.set_internet_mode(internet_mode)
    st.session_state.internet_mode = internet_mode

# Show current mode status
if 'rag_system' in st.session_state:
    mode_status = st.session_state.rag_system.get_mode_status()
    if internet_mode:
        st.sidebar.success("🌐 Internet Search Mode Active")
        st.sidebar.info("Searching the web for answers")
    else:
        st.sidebar.info("📄 Document Search Mode Active")
        if mode_status['documents_loaded']:
            st.sidebar.success(f"📚 {mode_status['total_documents']} documents loaded")
        else:
            st.sidebar.warning("No documents loaded")

# Performance metrics in sidebar
if st.session_state.performance_metrics:
    st.sidebar.markdown("---")
    st.sidebar.subheader("Performance")
    if 'last_response_time' in st.session_state.performance_metrics:
        st.sidebar.metric("Last Response Time", f"{st.session_state.performance_metrics['last_response_time']:.2f}s")
    if st.session_state.error_count > 0:
        st.sidebar.metric("Errors", st.session_state.error_count)

# Main content
st.title("HERON")

# Display conversation history using logic layer
conversation_history = st.session_state.rag_system.get_conversation_history()
if conversation_history:
    st.subheader("Conversation History")
    for i, conv in enumerate(conversation_history):
        # Show mode indicator
        mode_icon = "🌐" if conv.get('mode') == 'internet' else "📄"
        mode_text = "Internet" if conv.get('mode') == 'internet' else "Document"
        
        with st.expander(f"{mode_icon} Q{i+1}: {conv['question'][:50]}... ({mode_text} Mode)"):
            st.write(f"**Question:** {conv['question']}")
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
                st.write(f"**Answer:** {conv['answer']}")

# Show main question input only if there is no conversation history
if not conversation_history:
    # Initialize submit flag
    if 'submit_question' not in st.session_state:
        st.session_state.submit_question = False
    
    # Show current mode in the input
    current_mode = "Internet Search" if st.session_state.get('internet_mode', False) else "Document Search"
    placeholder_text = f"Ask a question (using {current_mode})..."
    
    question = st.text_input(
        "Ask a question:",
        key="question_input",
        placeholder=placeholder_text,
        on_change=handle_enter_key,
        help=f"Press Enter to submit. Currently using {current_mode} mode."
    )
    
    # Handle submission via button or Enter key
    if st.button("Get Answer", type="primary") or st.session_state.submit_question:
        with st.spinner("Processing..."):
            answer = generate_answer(question)
            # If answer is a list (image info), display images
            if isinstance(answer, list):
                if answer:
                    img_info = answer[0]  # Only show the most relevant image
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
                st.write(answer)
        st.session_state.answer_given = True
        st.session_state.submit_question = False  # Reset flag

# Always show follow-up input if there is any conversation history
if conversation_history:
    st.markdown("---")
    
    # Initialize input key counter
    if 'followup_input_key_counter' not in st.session_state:
        st.session_state.followup_input_key_counter = 0
    
    def submit_followup():
        follow_up_input_key = f"followup_input_{st.session_state.followup_input_key_counter}"
        follow_up_question = st.session_state.get(follow_up_input_key, "")
        if follow_up_question.strip():
            generate_follow_up(follow_up_question)
        st.session_state.followup_input_key_counter += 1
    
    # Use a container to keep input and button together
    followup_container = st.container()
    with followup_container:
        col_input, col_button = st.columns([4, 1])
        with col_input:
            follow_up_input_key = f"followup_input_{st.session_state.followup_input_key_counter}"
            st.text_input(
                "Ask a follow-up question:",
                key=follow_up_input_key,
                value="",
                on_change=submit_followup,
                help="Press Enter to submit"
            )
        with col_button:
            if st.button("Submit Follow-up", key="submit_followup_btn"):
                submit_followup()

# Control buttons
st.markdown("---")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("Reset Session", type="secondary"):
        # Clear session state and conversation history
        st.session_state.rag_system.clear_conversation_history()
        
        # Clear the answer given flag
        if 'answer_given' in st.session_state:
            del st.session_state.answer_given
        
        # Clean up temporary PDF files
        if hasattr(st.session_state.rag_system.file_handler, 'get_saved_pdf_paths'):
            saved_paths = st.session_state.rag_system.file_handler.get_saved_pdf_paths()
            for pdf_path in saved_paths:
                if os.path.exists(pdf_path):
                    try:
                        os.remove(pdf_path)
                        logger.info(f"Cleaned up: {pdf_path}")
                    except Exception as e:
                        logger.error(f"Could not remove {pdf_path}: {e}")
        
        # Clean up image files
        if os.path.exists("images"):
            import shutil
            try:
                shutil.rmtree("images")
                logger.info("Cleaned up images directory")
            except Exception as e:
                logger.error(f"Could not clean up images directory: {e}")
        
        # Reset performance metrics
        st.session_state.rag_system.reset_performance_metrics()
        st.session_state.error_count = 0
        st.session_state.performance_metrics = {}
        
        # Clear upload state
        st.session_state.documents_loaded = False
        st.session_state.last_uploaded_files = []
        st.session_state.processing_status = {}

with col2:
    if st.button("Export PDF"):
        if conversation_history:
            pdf_path = export_conversation_to_pdf()
            if pdf_path:
                # Read the PDF file and create download button
                with open(pdf_path, "rb") as pdf_file:
                    pdf_bytes = pdf_file.read()
                
                st.download_button(
                    label="Download PDF",
                    data=pdf_bytes,
                    file_name=f"heron_conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf"
                )
        else:
            st.warning("No conversation to export")

with col3:
    if st.button("Performance Stats"):
        metrics = st.session_state.rag_system.get_performance_metrics()
        stats = st.session_state.rag_system.question_handler.get_conversation_stats()
        
        st.subheader("System Performance")
        st.metric("Total Queries", metrics.get('total_queries', 0))
        st.metric("Errors", metrics.get('error_count', 0))
        if 'last_response_time' in st.session_state.performance_metrics:
            st.metric("Last Response Time", f"{st.session_state.performance_metrics['last_response_time']:.2f}s")
        
        st.subheader("Conversation Stats")
        st.metric("Total Questions", stats.get('total_questions', 0))
        st.metric("Error Count", stats.get('error_count', 0))

# Error display
if st.session_state.error_count > 0:
    st.error(f"⚠️ {st.session_state.error_count} error(s) encountered. Check the logs for details.")

# Session info
if st.session_state.documents_loaded:
    st.sidebar.markdown("---")
    st.sidebar.subheader("Session Info")
    st.sidebar.info(f"Documents loaded: {len(st.session_state.get('last_uploaded_files', []))}")
    if st.session_state.last_upload_time:
        st.sidebar.info(f"Last upload: {datetime.fromtimestamp(st.session_state.last_upload_time).strftime('%H:%M:%S')}") 
