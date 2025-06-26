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

def submit_chat_message():
    chat_input_key = f"chat_input_{st.session_state.chat_input_key}"
    chat_question = st.session_state.get(chat_input_key, "")
    if chat_question.strip():
        # Check if documents are loaded OR internet mode is enabled
        if not st.session_state.get('documents_loaded', False) and not st.session_state.get('internet_mode', False):
            # Quick message for no documents and no internet mode
            answer = "Please upload a document first or enable Live Web Search."
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
                # Source request logic (your existing code)
                pass  # Add your existing source request code here
            else:
                # Check if both modes are enabled
                if st.session_state.get('use_both_modes', False) and st.session_state.get('documents_loaded', False):
                    # Try document search first, then web search if no good results
                    doc_answer = st.session_state.rag_system.process_question_with_mode(chat_question, normalize_length=True)
                    
                    # If document answer is generic, try web search
                    if "No relevant information found" in doc_answer or "No documents loaded" in doc_answer:
                        web_answer = st.session_state.rag_system.process_live_web_question(chat_question)
                        answer = f"Document Search: {doc_answer}\n\nWeb Search: {web_answer}"
                    else:
                        answer = f"Document Search: {doc_answer}"
                        
                elif st.session_state.get('internet_mode', False):
                    # Use live web search only
                    answer = st.session_state.rag_system.process_live_web_question(chat_question)
                else:
                    # Use document search only
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
    st.session_state.chat_input_key += 1

# Show current mode in the placeholder
current_mode = "Internet Search" if st.session_state.get('internet_mode', False) else "Document Search"
placeholder_text = f"Ask a question (using {current_mode})..."

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
    
    search_mode = st.selectbox(
        "Choose search mode:",
        ["Documents", "Live Web Search", "Both"],
        index=0,
        help="Live Web Search uses DuckDuckGo for real-time information"
    )

    # Track previous search mode and force rerun on change
    if 'previous_search_mode' not in st.session_state:
        st.session_state.previous_search_mode = search_mode

    # If search mode changed, force rerun to update session state
    if st.session_state.previous_search_mode != search_mode:
        st.session_state.previous_search_mode = search_mode
        st.rerun()

    # Now set the internet mode based on current selection
    if search_mode == "Live Web Search":
        st.session_state.internet_mode = True
        st.success("🌐 Live Web Search Enabled")
    elif search_mode == "Documents":
        st.session_state.internet_mode = False
        if st.session_state.get('documents_loaded'):
            st.info(f"📄 Document Mode ({len(st.session_state.get('last_uploaded_files', []))} docs)")
        else:
            st.warning("No documents loaded")
    elif search_mode == "Both":
        st.session_state.internet_mode = True
        st.session_state.use_both_modes = True
        st.success("🌐 Both Modes Enabled")

    # Export Section (only show if there's conversation history)
    if conversation_history:
        st.markdown("---")
        st.subheader("Export")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📄 PDF", use_container_width=True):
                pdf_path = export_conversation_to_pdf()
                if pdf_path and os.path.exists(pdf_path):
                    with open(pdf_path, "rb") as pdf_file:
                        pdf_bytes = pdf_file.read()
                    st.download_button(
                        label="Download PDF",
                        data=pdf_bytes,
                        file_name=f"heron_conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
                else:
                    st.error("Failed to create PDF")
        
        with col2:
            if st.button("📝 Markdown", use_container_width=True):
                markdown_content = export_conversation_to_markdown()
                if markdown_content:
                    st.download_button(
                        label="Download MD",
                        data=markdown_content,
                        file_name=f"heron_conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                        mime="text/markdown",
                        use_container_width=True
                    )
                else:
                    st.error("Failed to create Markdown")
        
        with col3:
            if st.button("📊 CSV", use_container_width=True):
                csv_content = export_conversation_to_csv()
                if csv_content:
                    st.download_button(
                        label="Download CSV",
                        data=csv_content,
                        file_name=f"heron_conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                else:
                    st.error("Failed to create CSV")

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

# Modern chat input - always visible
chat_input_key = f"chat_input_{st.session_state.chat_input_key}"
chat_question = st.chat_input(
    placeholder_text,
    key=chat_input_key
)

if chat_question:
    submit_chat_message() 
