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

# Main content - Clean, modern chat interface
st.title("HERON")

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

# Sidebar - All controls moved here for clean main area
with st.sidebar:
    st.header("HERON Controls")
    
    # Document Management Section
    st.subheader("Documents")
    
    # Show uploaded documents and allow removal
    if 'last_uploaded_files' in st.session_state and st.session_state['last_uploaded_files']:
        st.write("**Uploaded Documents:**")
        docs_to_remove = []
        for doc_name in st.session_state['last_uploaded_files']:
            col1, col2 = st.columns([4, 1])
            with col1:
                st.write(doc_name)
            with col2:
                if st.button("Remove", key=f"remove_{doc_name}", help="Remove document"):
                    docs_to_remove.append(doc_name)
        # Remove selected docs
        if docs_to_remove:
            st.session_state['last_uploaded_files'] = [d for d in st.session_state['last_uploaded_files'] if d not in docs_to_remove]
            st.session_state.documents_loaded = False
            st.session_state.processing_status = {}
            st.experimental_rerun()

    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=['pdf'],
        accept_multiple_files=True,
        key="pdf_uploader"
    )
    
    # Progress indicator and status
    if uploaded_files:
        current_files = [f.name for f in uploaded_files]
        last_files = st.session_state.get('last_uploaded_files', [])
        if current_files != last_files or not st.session_state.get('documents_loaded'):
            st.session_state.last_uploaded_files = current_files
            st.session_state.last_upload_time = time.time()
            with st.spinner("Processing documents..."):
                # Show progress bar for processing
                progress_bar = st.progress(0)
                total_steps = 3
                progress_bar.progress(1/total_steps, text="Uploading files...")
                time.sleep(0.5)
                if st.session_state.rag_system.process_web_uploads(uploaded_files):
                    progress_bar.progress(2/total_steps, text="Embedding documents...")
                    time.sleep(0.5)
                    st.success(f"Processed {len(uploaded_files)} file(s)")
                    st.session_state.documents_loaded = True
                    processing_status = st.session_state.rag_system.file_handler.get_processing_status()
                    st.session_state.processing_status = processing_status
                    progress_bar.progress(1.0, text="Ready!")
                else:
                    st.error("Failed to process files")
                    st.session_state.documents_loaded = False
                    progress_bar.progress(1.0, text="Error")
        else:
            st.info("Documents already loaded")
    
    # Show processing details
    if st.session_state.get('processing_status'):
        st.write("**Processing Details:**")
        for filename, status in st.session_state['processing_status'].items():
            if status == "success":
                st.success(f"✅ {filename}")
            else:
                st.error(f"❌ {filename}")

    # Search Mode Section
    st.markdown("---")
    st.subheader("Search Mode")
    
    internet_mode = st.toggle(
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
            st.success("Internet Search Active")
        else:
            st.info("Document Search Active")
            if mode_status['documents_loaded']:
                st.success(f"{mode_status['total_documents']} documents loaded")
            else:
                st.warning("No documents loaded")

    # Session Management Section
    st.markdown("---")
    st.subheader("Session")
    
    # Reset Session
    if st.button("Reset Session", type="secondary", use_container_width=True):
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
        st.success("Session reset!")

    # Save/Load Session
    col_save, col_load = st.columns(2)
    with col_save:
        if st.button("Save Session", use_container_width=True):
            # Prepare session data
            session_data = {
                'conversation_history': st.session_state.rag_system.get_conversation_history(),
                'last_uploaded_files': st.session_state.get('last_uploaded_files', []),
                'documents_loaded': st.session_state.get('documents_loaded', False),
                'performance_metrics': st.session_state.get('performance_metrics', {}),
                'error_count': st.session_state.get('error_count', 0),
                'last_upload_time': st.session_state.get('last_upload_time', None),
                'processing_status': st.session_state.get('processing_status', {}),
                'internet_mode': st.session_state.get('internet_mode', False)
            }
            session_json = json.dumps(session_data, indent=2)
            st.download_button(
                label="Download Session",
                data=session_json,
                file_name=f"heron_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True
            )

    with col_load:
        uploaded_session = st.file_uploader("Load Session", type=["json"], key="session_loader")
        if uploaded_session is not None:
            try:
                session_data = json.load(uploaded_session)
                # Restore session state
                st.session_state['last_uploaded_files'] = session_data.get('last_uploaded_files', [])
                st.session_state['documents_loaded'] = session_data.get('documents_loaded', False)
                st.session_state['performance_metrics'] = session_data.get('performance_metrics', {})
                st.session_state['error_count'] = session_data.get('error_count', 0)
                st.session_state['last_upload_time'] = session_data.get('last_upload_time', None)
                st.session_state['processing_status'] = session_data.get('processing_status', {})
                st.session_state['internet_mode'] = session_data.get('internet_mode', False)
                # Restore conversation history in RAG system
                st.session_state.rag_system.set_conversation_history(session_data.get('conversation_history', []))
                st.success("Session loaded!")
            except Exception as e:
                st.error(f"Failed to load: {e}")

    # Export Section (only show if there's conversation)
    if conversation_history:
        st.markdown("---")
        st.subheader("Export")
        
        export_format = st.selectbox(
            "Export format:",
            ["PDF", "Markdown", "CSV"],
            help="Choose export format"
        )
        
        if st.button(f"Export {export_format}", use_container_width=True):
            if export_format == "PDF":
                pdf_path = export_conversation_to_pdf()
                if pdf_path:
                    with open(pdf_path, "rb") as pdf_file:
                        pdf_bytes = pdf_file.read()
                    st.download_button(
                        label="Download PDF",
                        data=pdf_bytes,
                        file_name=f"heron_conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
            elif export_format == "Markdown":
                md_lines = ["# HERON Conversation Export\n"]
                for i, conv in enumerate(conversation_history):
                    md_lines.append(f"**Q{i+1}:** {conv['question']}")
                    if isinstance(conv['answer'], list):
                        md_lines.append(f"**A{i+1}:** [Image(s) attached]")
                    else:
                        md_lines.append(f"**A{i+1}:** {conv['answer']}")
                    doc_name = conv.get('source_document')
                    page_num = conv.get('source_page')
                    if doc_name or page_num:
                        attr = "Source: "
                        if doc_name:
                            attr += doc_name
                        if page_num:
                            attr += f" (Page {page_num})"
                        md_lines.append(f"<sub>{attr}</sub>")
                    md_lines.append("")
                md_content = "\n".join(md_lines)
                st.download_button(
                    label="Download Markdown",
                    data=md_content,
                    file_name=f"heron_conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                    mime="text/markdown",
                    use_container_width=True
                )
            elif export_format == "CSV":
                import csv
                import io
                output = io.StringIO()
                writer = csv.writer(output)
                writer.writerow(["Question", "Answer", "Source Document", "Page"])
                for conv in conversation_history:
                    answer = conv['answer'] if not isinstance(conv['answer'], list) else '[Image(s) attached]'
                    writer.writerow([
                        conv['question'],
                        answer,
                        conv.get('source_document', ''),
                        conv.get('source_page', '')
                    ])
                st.download_button(
                    label="Download CSV",
                    data=output.getvalue(),
                    file_name=f"heron_conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )

    # Performance Section
    st.markdown("---")
    st.subheader("Performance")
    
    # Get actual performance metrics
    try:
        metrics = st.session_state.rag_system.get_performance_metrics()
        stats = st.session_state.rag_system.question_handler.get_conversation_stats()
        
        # Display key metrics
        if metrics:
            st.metric("Total Queries", metrics.get('total_queries', 0))
            st.metric("Total Errors", metrics.get('error_count', 0))
        
        if stats:
            st.metric("Total Questions", stats.get('total_questions', 0))
            st.metric("Conversation Errors", stats.get('error_count', 0))
        
        if st.session_state.performance_metrics:
            if 'last_response_time' in st.session_state.performance_metrics:
                st.metric("Last Response Time", f"{st.session_state.performance_metrics['last_response_time']:.2f}s")
        
        if conversation_history:
            st.metric("Messages", len(conversation_history))
        
        if st.session_state.get('documents_loaded'):
            st.metric("Documents", len(st.session_state.get('last_uploaded_files', [])))
            
    except Exception as e:
        st.error(f"Error loading performance metrics: {e}")

    # Session info
    if st.session_state.documents_loaded:
        st.markdown("---")
        st.subheader("Info")
        st.info(f"{len(st.session_state.get('last_uploaded_files', []))} documents loaded")
        if st.session_state.last_upload_time:
            st.info(f"Last upload: {datetime.fromtimestamp(st.session_state.last_upload_time).strftime('%H:%M:%S')}")

# Error display (minimal, in main area)
if st.session_state.error_count > 0:
    st.error(f"{st.session_state.error_count} error(s) encountered. Check the sidebar for details.") 
