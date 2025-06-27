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
            
            # Add audio indicator
            file_type = conv.get('metadata', {}).get('file_type', 'document')
            if file_type == 'audio':
                mode_icon = "🎵"
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

# Update the submit_chat_message function
def submit_chat_message():
    chat_input_key = f"chat_input_{st.session_state.chat_input_key}"
    chat_question = st.session_state.get(chat_input_key, "")
    
    if chat_question.strip():
        # Check if this is a chart request
        if is_chart_request(chat_question):
            # Process chart request with progress indicator
            with st.spinner("🔄 Converting PDF to images and extracting chart data..."):
                answer = st.session_state.rag_system.process_chart_request(chat_question)
            st.session_state.rag_system.add_to_conversation_history(chat_question, answer, "chart_request", "document")
            st.rerun()
        else:
            # Check if this is an image request FIRST
            if is_image_request(chat_question):
                # Process image request with visual feedback
                progress_placeholder = st.empty()
                progress_placeholder.info("🔍 Searching for images in document...")
                
                try:
                    answer = st.session_state.rag_system.process_image_request(chat_question)
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
                
                st.session_state.rag_system.add_to_conversation_history(chat_question, answer, "image_request", "document")
                st.rerun()
            else:
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
                        'show me the table', 'show the source', 'show the graph',
                        'show the chart', 'show the figure', 'show the table',
                        'source for this', 'graph for this', 'chart for this', 'figure for this',
                        'table for this'
                    ]
                    
                    is_source_request = any(phrase in question_lower for phrase in trigger_phrases)
                    
                    if is_source_request:
                        # Source request logic (your existing code)
                        pass  # Add your existing source request code here
                    else:
                        # Add loading indicator for all question processing
                        with st.spinner("🤔 Thinking..."):
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
        type=['pdf', 'doc', 'docx', 'xls', 'xlsx', 'ppt', 'pptx', 'mp3', 'wav', 'm4a', 'flac', 'ogg', 'aac'],
        accept_multiple_files=True,
        key="pdf_uploader"
    )
    
    # Process uploaded files
    if uploaded_files:
        current_files = [f.name for f in uploaded_files]
        last_files = st.session_state.get('last_uploaded_files', [])
        
        # Check if files are actually new and not currently processing
        new_files = [f for f in current_files if f not in last_files]
        is_processing = st.session_state.get('is_processing_files', False)
        
        if (new_files or not st.session_state.get('documents_loaded')) and not is_processing:
            st.session_state.is_processing_files = True
            st.session_state.last_uploaded_files = current_files
            st.session_state.last_upload_time = time.time()
            
            # Check for audio files
            audio_files = [f for f in uploaded_files if f.name.lower().endswith(('.mp3', '.wav', '.m4a', '.flac', '.ogg', '.aac'))]
            document_files = [f for f in uploaded_files if f.name.lower().endswith(('.pdf', '.docx', '.pptx', '.doc', '.xls', '.xlsx', '.ppt'))]
            
            if audio_files:
                st.info(f"🎵 Found {len(audio_files)} audio file(s) - will transcribe to text")
            
            with st.spinner("Processing..."):
                if st.session_state.rag_system.process_web_uploads(uploaded_files):
                    st.success(f"{len(uploaded_files)} file(s) loaded")
                    if audio_files:
                        st.success("🎵 Audio files transcribed successfully")
                    st.session_state.documents_loaded = True
                    st.session_state.processing_status = st.session_state.rag_system.file_handler.get_processing_status()
                else:
                    st.error("Processing failed")
                    st.session_state.documents_loaded = False
        
            st.session_state.is_processing_files = False
        else:
            # Files haven't changed or currently processing, just show status
            if st.session_state.get('documents_loaded'):
                st.info(f"{len(current_files)} file(s) already loaded")

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
                        file_name="conversation.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
        
        with col2:
            if st.button("📝 Markdown", use_container_width=True):
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
            if st.button("📊 CSV", use_container_width=True):
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
                return "❌ No documents uploaded"
            
            pdf_path = os.path.join("temp", uploaded_files[0])
            if not os.path.exists(pdf_path):
                return "❌ Document not found in temp directory"
            
            # Actually extract images
            processor = OnDemandImageProcessor()
            images = processor.extract_images_from_pdf(pdf_path)
            
            if images:
                # Show the actual images found
                st.write(f"✅ Found {len(images)} images:")
                for i, img_info in enumerate(images, 1):
                    st.write(f"  - Image {i}: Page {img_info['page']} ({img_info['path']})")
                
                # Show first image as preview
                if os.path.exists(images[0]['path']):
                    st.image(images[0]['path'], caption=f"Preview: {images[0]['description']}", width=300)
                
                return f"✅ Successfully extracted {len(images)} images"
            else:
                return "⚠️ No images found in document (PDF may only contain text)"
            
        except Exception as e:
            return f"❌ Error: {str(e)}"

    # Add this to your sidebar
    if st.button("🔍 Test Image Extraction", use_container_width=True):
        result = test_image_extraction()
        st.info(result)

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

# Add this function to handle audio-specific questions
def is_audio_question(question: str) -> bool:
    """Detect if the question is about audio content."""
    question_lower = question.lower()
    audio_keywords = [
        'audio', 'recording', 'transcript', 'transcription', 'voice', 'speech',
        'said', 'mentioned', 'talked about', 'discussed', 'conversation',
        'interview', 'podcast', 'meeting', 'call', 'recording'
    ]
    return any(keyword in question_lower for keyword in audio_keywords)

def install_system_dependencies():
    """Install system dependencies if needed."""
    try:
        # Check if ffmpeg is available
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        print("✅ FFmpeg is available")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("⚠️ FFmpeg not found. Audio processing may not work properly.")
        print("To install FFmpeg:")
        print("  Ubuntu/Debian: sudo apt-get install ffmpeg")
        print("  macOS: brew install ffmpeg")
        print("  Windows: Download from https://ffmpeg.org/download.html")

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
        
        # Add transcription content
        # Split transcription into paragraphs for better formatting
        paragraphs = transcription_text.split('\n\n')
        
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
        
        return pdf_path
        
    except Exception as e:
        st.error(f"Error creating PDF: {str(e)}")
        return None

# Call this function when the app starts
if __name__ == "__main__":
    install_system_dependencies() 
