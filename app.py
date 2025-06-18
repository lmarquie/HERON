import streamlit as st
import os
import json
import time
import gc
import shutil
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from local_draft import RAGSystem, WebFileHandler
from config import OPENAI_API_KEY
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

# Try to import optional dependencies
try:
    import psutil
except ImportError:
    psutil = None

try:
    import torch
except ImportError:
    torch = None

# Create app data directory if it doesn't exist
DATA_DIR = 'app_data'
os.makedirs(DATA_DIR, exist_ok=True)

def get_current_settings():
    """Get the current settings from session state or defaults"""
    if not hasattr(st.session_state, 'settings'):
        st.session_state.settings = {
            'chunk_size': 500,
            'chunk_overlap': 50,
            'max_chunks': 10,
            'search_depth': 3,
            'relevance_threshold': 0.7,
            'temperature': 0.3,
            'max_tokens': 1000,
            'top_p': 0.9,
            'speed': 0.5,
            'accuracy': 0.5
        }
    return st.session_state.settings

# Set page config
st.set_page_config(
    page_title="HERON - Herbert Advisory",
    page_icon="🦅",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar
with st.sidebar:
    st.markdown("""
        <div style='text-align: center; padding: 1rem 0; margin-bottom: 1rem;'>
            <h2 style='color: #1E3A8A; font-size: 1.2rem; font-weight: 600; margin: 0 0 0.5rem 0;'>Herbert Advisory</h2>
            <p style='color: #4B5563; font-size: 0.9rem; line-height: 1.4;'>
                HERON (Herbert Embedded Retrieval and Oracle Network) is an advanced document analysis platform that leverages artificial intelligence to facilitate comprehensive document interrogation through natural language processing capabilities.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # File uploader
    st.header("Upload Documents")
    try:
        uploaded_files = st.file_uploader(
            label="Upload your documents to analyze",
            type=['pdf', 'txt'],
            accept_multiple_files=True,
            help="Upload PDF or text files to analyze",
            label_visibility="collapsed",
            key="sidebar_file_uploader"
        )
        
        if uploaded_files:
            if st.session_state.rag_system.process_web_uploads(uploaded_files):
                st.success(f"Successfully processed {len(uploaded_files)} file(s)")
                st.session_state.documents_loaded = True
            else:
                st.error("Failed to process uploaded files")
    except Exception as e:
        st.error(f"Error with file uploader: {str(e)}")
        st.info("If you're running this in a cloud environment, please ensure you have proper permissions.")
    
    st.divider()
    
    # Display current model settings
    st.header("Model Settings")
    
    settings = get_current_settings()
    
    # Document Processing
    st.subheader("Document Processing")
    st.write("Chunk Size")
    st.progress(settings['chunk_size'] / 1000)  # Normalize to 0-1 range
    st.write("Chunk Overlap")
    st.progress(settings['chunk_overlap'] / 100)  # Normalize to 0-1 range
    st.write("Max Chunks")
    st.progress(settings['max_chunks'] / 20)  # Normalize to 0-1 range
    
    # Search Settings
    st.subheader("Search Settings")
    st.write("Search Depth")
    st.progress(settings['search_depth'] / 5)  # Normalize to 0-1 range
    st.write("Relevance Threshold")
    st.progress(settings['relevance_threshold'])  # Already 0-1 range
    
    # Model Parameters
    st.subheader("Model Parameters")
    st.write("Temperature")
    st.progress(settings['temperature'])  # Already 0-1 range
    st.write("Max Tokens")
    st.progress(settings['max_tokens'] / 2000)  # Normalize to 0-1 range
    st.write("Top P")
    st.progress(settings['top_p'])  # Already 0-1 range

# Initialize session state
if 'current_page' not in st.session_state:
    st.session_state.current_page = "main"
if 'question' not in st.session_state:
    st.session_state.question = ""
if 'answer' not in st.session_state:
    st.session_state.answer = None
if 'sources' not in st.session_state:
    st.session_state.sources = []
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []
if 'documents_loaded' not in st.session_state:
    st.session_state.documents_loaded = False
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'rag_system' not in st.session_state:
    st.session_state.rag_system = None
if 'processing' not in st.session_state:
    st.session_state.processing = False
if 'current_theme' not in st.session_state:
    st.session_state.current_theme = "System Default"
if 'main_answer' not in st.session_state:
    st.session_state.main_answer = None
if 'main_results' not in st.session_state:
    st.session_state.main_results = []
if 'follow_up_answer' not in st.session_state:
    st.session_state.follow_up_answer = None

# Create data directory if it doesn't exist
DATA_DIR = "app_data"
os.makedirs(DATA_DIR, exist_ok=True)

def save_conversation_history():
    """Save conversation history to a file"""
    if 'main_answer' in st.session_state:
        data = {
            'main_answer': st.session_state.main_answer,
            'main_results': st.session_state.main_results if 'main_results' in st.session_state else [],
            'question': st.session_state.question if 'question' in st.session_state else "",
            'timestamp': datetime.now().isoformat()
        }
        
        # Save to file
        with open(os.path.join(DATA_DIR, 'conversation_history.json'), 'w') as f:
            json.dump(data, f)

def load_conversation_history():
    """Load conversation history from file"""
    history_file = os.path.join(DATA_DIR, 'conversation_history.json')
    if os.path.exists(history_file):
        try:
            with open(history_file, 'r') as f:
                data = json.load(f)
                
            if data['main_answer']:
                st.session_state.main_answer = data['main_answer']
            if 'main_results' in data:
                st.session_state.main_results = data['main_results']
            if 'question' in data:
                st.session_state.question = data['question']
        except Exception as e:
            st.error(f"Error loading conversation history: {str(e)}")

def save_settings(settings):
    """Save settings to a file"""
    try:
        # Ensure app_data directory exists with proper permissions
        app_data_dir = os.path.abspath('app_data')
        if not os.path.exists(app_data_dir):
            os.makedirs(app_data_dir, mode=0o755)
        
        # Ensure all required settings are present
        default_settings = {
            'num_results': 3,
            'chunk_size': 500,
            'chunk_overlap': 50,
            'model_temperature': 0.3,
            'sequence_length': 256,
            'batch_size': 128,
            'use_half_precision': True,
            'doc_percentage': 15,
            'speed_accuracy': 50
        }
        
        # Update defaults with current settings
        default_settings.update(settings)
        
        # Save to file with proper formatting
        settings_path = os.path.join(app_data_dir, 'settings.json')
        
        # First, try to write to a temporary file
        temp_path = settings_path + '.tmp'
        try:
            with open(temp_path, 'w') as f:
                json.dump(default_settings, f, indent=4)
            
            # If successful, rename the temporary file to the actual file
            if os.path.exists(settings_path):
                os.remove(settings_path)
            os.rename(temp_path, settings_path)
            
        except Exception as e:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            return False
        
        # Update the RAG system with new settings
        if hasattr(st.session_state, 'rag_system') and st.session_state.rag_system is not None:
            st.session_state.rag_system.apply_settings(default_settings)
            
            # Save the updated vector store
            if hasattr(st.session_state.rag_system, 'vector_store') and st.session_state.rag_system.vector_store is not None:
                vector_store_path = os.path.join(app_data_dir, 'vector_store')
                st.session_state.rag_system.vector_store.save_local(vector_store_path)
        
        # Verify the file was created
        if os.path.exists(settings_path):
            st.success("Settings saved successfully!")
            return True
        else:
            st.error("Settings file was not created successfully")
            return False
            
    except Exception as e:
        st.error(f"Error saving settings: {str(e)}")
        return False

def load_settings():
    """Load settings from file"""
    try:
        app_data_dir = os.path.abspath('app_data')
        settings_path = os.path.join(app_data_dir, 'settings.json')
        
        if os.path.exists(settings_path):
            try:
                with open(settings_path, 'r') as f:
                    saved_settings = json.load(f)
                
                # Ensure all required settings are present
                default_settings = {
                    'num_results': 3,
                    'chunk_size': 500,
                    'chunk_overlap': 50,
                    'model_temperature': 0.3,
                    'sequence_length': 256,
                    'batch_size': 128,
                    'use_half_precision': True,
                    'doc_percentage': 15,
                    'speed_accuracy': 50
                }
                
                # Update defaults with saved settings
                default_settings.update(saved_settings)
                
                # Apply settings to RAG system if it exists
                if hasattr(st.session_state, 'rag_system') and st.session_state.rag_system is not None:
                    st.session_state.rag_system.apply_settings(default_settings)
                
                return default_settings
            except Exception as e:
                st.error(f"Error reading settings file: {str(e)}")
    except Exception as e:
        st.error(f"Error loading settings: {str(e)}")
    
    # Return default settings if no file exists or error occurs
    return {
        'num_results': 3,
        'chunk_size': 500,
        'chunk_overlap': 50,
        'model_temperature': 0.3,
        'sequence_length': 256,
        'batch_size': 128,
        'use_half_precision': True,
        'doc_percentage': 15,
        'speed_accuracy': 50
    }

def save_document_state():
    """Save document state to a file"""
    if not os.path.exists('app_data'):
        os.makedirs('app_data')
    
    if st.session_state.rag_system and st.session_state.rag_system.vector_store:
        # Save the vector store
        st.session_state.rag_system.vector_store.save_local('app_data/vector_store')
        # Save document state
        with open('app_data/document_state.json', 'w') as f:
            json.dump({
                'documents_loaded': st.session_state.documents_loaded,
                'last_updated': datetime.now().isoformat()
            }, f)

def load_document_state():
    """Load document state from file"""
    try:
        if os.path.exists('app_data/document_state.json'):
            with open('app_data/document_state.json', 'r') as f:
                state = json.load(f)
                st.session_state.documents_loaded = state['documents_loaded']
                
                # Load the vector store if it exists
                if os.path.exists('app_data/vector_store'):
                    st.session_state.rag_system = RAGSystem()
                    st.session_state.rag_system.vector_store.load_local('app_data/vector_store')
                    return True
    except Exception as e:
        st.error(f"Error loading document state: {str(e)}")
    return False

# Load saved data
load_conversation_history()
settings = load_settings()
documents_restored = load_document_state()

# Initialize RAG system with settings if it doesn't exist
try:
    if not hasattr(st.session_state, 'rag_system') or st.session_state.rag_system is None:
        with st.spinner("Initializing RAG system..."):
            try:
                # Initialize with explicit device and model settings
                st.session_state.rag_system = RAGSystem(
                    settings={
                        'device': 'cpu',  # Force CPU for web version
                        'use_half_precision': False,  # Disable half precision for stability
                        'model_temperature': 0.3,
                        'chunk_size': 500,
                        'chunk_overlap': 50,
                        'num_results': 3
                    },
                    is_web=True
                )
                st.success("RAG system initialized successfully!")
            except Exception as e:
                st.error(f"Error initializing RAG system: {str(e)}")
                st.info("Please refresh the page and try again.")
except Exception as e:
    st.error(f"Critical error initializing RAG system: {str(e)}")
    st.info("Please refresh the page and try again.")

def update_settings_from_main_slider():
    """Update all settings based on the main speed/accuracy slider"""
    try:
        slider_value = st.session_state.speed_accuracy_slider
        settings = get_current_settings()
        
        # Update all settings based on the main slider
        settings.update({
            'doc_percentage': int(5 + (slider_value / 100) * 25),  # Reduced range
            'chunk_size': int(200 + (slider_value / 100) * 400),   # Reduced range
            'chunk_overlap': int(20 + (slider_value / 100) * 40),  # Reduced range
            'model_temperature': float(0.1 + (slider_value / 100) * 0.2),
            'sequence_length': int(128 + (slider_value / 100) * 256),  # Reduced range
            'batch_size': int(32 + (slider_value / 100) * 128),    # Reduced range
            'use_half_precision': slider_value < 50,
            'speed_accuracy': int(slider_value)
        })
        
        # Apply settings to RAG system
        if hasattr(st.session_state, 'rag_system') and st.session_state.rag_system is not None:
            st.session_state.rag_system.apply_settings(settings)
            save_settings(settings)
            st.session_state.update_trigger = True
    except Exception as e:
        st.error(f"Error updating settings: {str(e)}")

def update_settings_from_individual_sliders():
    """Update settings based on individual slider changes"""
    try:
        settings = get_current_settings()
        
        # Update settings from individual sliders with explicit type conversion
        settings.update({
            'doc_percentage': int(st.session_state.doc_percentage_slider),
            'chunk_size': int(st.session_state.chunk_size_slider),
            'chunk_overlap': int(st.session_state.chunk_overlap_slider),
            'model_temperature': float(st.session_state.model_temp_slider),
            'sequence_length': int(st.session_state.seq_length_slider),
            'batch_size': int(st.session_state.batch_size_slider),
            'use_half_precision': bool(st.session_state.half_precision_toggle)
        })
        
        # Calculate new speed_accuracy based on individual settings
        new_speed_accuracy = int(calculate_speed_accuracy_from_settings(settings))
        settings['speed_accuracy'] = new_speed_accuracy
        
        # Apply settings to RAG system
        if hasattr(st.session_state, 'rag_system') and st.session_state.rag_system is not None:
            st.session_state.rag_system.apply_settings(settings)
            save_settings(settings)
            st.session_state.update_trigger = True
            st.rerun()
    except Exception as e:
        st.error(f"Error updating settings: {str(e)}")

def calculate_speed_accuracy_from_settings(settings):
    """Calculate speed-accuracy value from settings (0-100)"""
    # Normalize values to 0-100 range
    doc_percentage = settings['doc_percentage']  # Already 0-100
    chunk_size = (settings['chunk_size'] - 100) / 9  # 100-1000 -> 0-100
    chunk_overlap = settings['chunk_overlap']  # Already 0-100
    temperature = settings['model_temperature'] * 100  # 0-1 -> 0-100
    sequence_length = (settings['sequence_length'] - 64) / 4.48  # 64-512 -> 0-100
    batch_size = (settings['batch_size'] - 32) / 2.24  # 32-256 -> 0-100
    
    # Weights for each setting (sum to 1.0)
    weights = {
        'doc_percentage': 0.2,  # Reduced weight
        'chunk_size': 0.15,    # Reduced weight
        'chunk_overlap': 0.15, # Reduced weight
        'temperature': 0.1,    # Reduced weight
        'sequence_length': 0.2, # Increased weight
        'batch_size': 0.2      # Increased weight
    }
    
    # Calculate weighted average
    speed_accuracy = (
        doc_percentage * weights['doc_percentage'] +
        chunk_size * weights['chunk_size'] +
        chunk_overlap * weights['chunk_overlap'] +
        temperature * weights['temperature'] +
        sequence_length * weights['sequence_length'] +
        batch_size * weights['batch_size']
    )
    
    # Adjust for half precision
    if settings['use_half_precision']:
        speed_accuracy *= 0.85  # Reduce by 15% when using half precision
    
    return speed_accuracy

def show_settings_page():
    # Ensure RAG system exists
    if not hasattr(st.session_state, 'rag_system') or st.session_state.rag_system is None:
        settings = load_settings()
        st.session_state.rag_system = RAGSystem(settings)
    rag = st.session_state.rag_system

    # Get current settings from our central settings store
    settings = get_current_settings()

    col1, col2 = st.columns([4, 1])

    with col1:
        back = st.button("Back to Main", use_container_width=True)
        if back:
            st.session_state.current_page = "main"
            st.rerun()

    with col1:
        # Speed vs Accuracy Slider with automatic application
        speed_accuracy = st.slider(
            "Speed vs Accuracy",
            0, 100,
            int(get_current_settings()['speed_accuracy']),
            step=1,
            help="Adjust the balance between processing speed and accuracy",
            key="speed_accuracy_slider",
            on_change=update_settings_from_main_slider
        )
        
        # Advanced Settings Expander
        with st.expander("Advanced Settings", expanded=False):
            with st.form("sidebar_advanced_settings_form"):
                st.markdown("### Document Processing")
                doc_percentage = st.slider(
                    "Document Coverage (%)",
                    5, 100,
                    settings['doc_percentage'],
                    help="Percentage of documents to analyze",
                    key="doc_percentage_slider"
                )
                chunk_size = st.slider(
                    "Chunk Size (words)",
                    100, 1000,
                    settings['chunk_size'],
                    help="Number of words per text chunk",
                    key="chunk_size_slider"
                )
                chunk_overlap = st.slider(
                    "Chunk Overlap (words)",
                    10, 200,
                    settings['chunk_overlap'],
                    help="Number of overlapping words between chunks",
                    key="chunk_overlap_slider"
                )
                
                st.markdown("### Model Configuration")
                model_temp = st.slider(
                    "Model Temperature",
                    0.0, 0.3,
                    settings['model_temperature'],
                    step=0.01,
                    help="Lower values make responses more focused",
                    key="model_temp_slider"
                )
                seq_length = st.slider(
                    "Sequence Length",
                    64, 512,
                    settings['sequence_length'],
                    help="Maximum number of tokens to process",
                    key="seq_length_slider"
                )
                batch_size = st.slider(
                    "Batch Size",
                    32, 256,
                    settings['batch_size'],
                    help="Number of items to process at once",
                    key="batch_size_slider"
                )
                half_precision = st.toggle(
                    "Half Precision",
                    settings['use_half_precision'],
                    help="Enable for faster processing",
                    key="half_precision_toggle"
                )
                
                # Add submit button
                submitted = st.form_submit_button("Apply Settings")
                if submitted:
                    update_settings_from_individual_sliders()

    with col2:
        pass

def generate_pdf_summary():
    """Generate a PDF summary of the current conversation"""
    try:
        # Create a temporary PDF file with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pdf_path = os.path.join(DATA_DIR, f'conversation_summary_{timestamp}.pdf')
        
        # Create the PDF document
        doc = SimpleDocTemplate(
            pdf_path,
            pagesize=letter,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=72
        )
        
        # Create custom styles
        styles = getSampleStyleSheet()
        styles.add(ParagraphStyle(
            name='BrandTitle',
            parent=styles['Title'],
            fontSize=24,
            spaceAfter=30,
            textColor=colors.HexColor('#1E3A8A')  # Dark blue
        ))
        styles.add(ParagraphStyle(
            name='SubTitle',
            parent=styles['Title'],
            fontSize=16,
            spaceAfter=20,
            textColor=colors.HexColor('#4B5563')  # Gray
        ))
        styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=styles['Heading2'],
            fontSize=14,
            spaceAfter=12,
            textColor=colors.HexColor('#1E3A8A')  # Dark blue
        ))
        styles.add(ParagraphStyle(
            name='Question',
            parent=styles['Normal'],
            fontSize=12,
            spaceAfter=6,
            textColor=colors.HexColor('#1E3A8A')  # Dark blue
        ))
        styles.add(ParagraphStyle(
            name='Answer',
            parent=styles['Normal'],
            fontSize=11,
            spaceAfter=12,
            leftIndent=20
        ))
        styles.add(ParagraphStyle(
            name='Source',
            parent=styles['Normal'],
            fontSize=10,
            spaceAfter=6,
            textColor=colors.HexColor('#6B7280')  # Light gray
        ))
        
        # Build the PDF content
        story = []
        
        # Add header with logo and branding
        story.append(Paragraph("HERON", styles['BrandTitle']))
        story.append(Paragraph("Herbert Advisory", styles['SubTitle']))
        story.append(Paragraph(f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
        story.append(Spacer(1, 20))
        
        # Add main question and answer
        story.append(Paragraph("Main Question", styles['SectionHeader']))
        story.append(Paragraph(st.session_state.question, styles['Question']))
        story.append(Spacer(1, 10))
        
        story.append(Paragraph("Answer", styles['SectionHeader']))
        story.append(Paragraph(st.session_state.main_answer, styles['Answer']))
        story.append(Spacer(1, 20))
        
        # Add sources if available
        if st.session_state.main_results:
            story.append(Paragraph("Sources", styles['SectionHeader']))
            for result in st.session_state.main_results:
                source = result.get('metadata', {}).get('source', 'Unknown source')
                story.append(Paragraph(f"• {source}", styles['Source']))
            story.append(Spacer(1, 20))
        
        # Add follow-up questions and answers
        if hasattr(st.session_state, 'follow_up_questions') and st.session_state.follow_up_questions:
            story.append(Paragraph("Follow-up Questions", styles['SectionHeader']))
            for i, (question, answer) in enumerate(st.session_state.follow_up_questions, 1):
                story.append(Paragraph(f"Follow-up {i}", styles['Question']))
                story.append(Paragraph(question, styles['Question']))
                story.append(Paragraph(answer, styles['Answer']))
                story.append(Spacer(1, 10))
        
        # Add footer
        story.append(Spacer(1, 30))
        story.append(Paragraph("© 2024 Herbert Advisory. All rights reserved.", styles['Source']))
        story.append(Paragraph("This document was generated by HERON (Herbert Embedded Retrieval and Oracle Network)", styles['Source']))
        
        # Build the PDF
        doc.build(story)
        
        return pdf_path
    except Exception as e:
        st.error(f"Error generating PDF: {str(e)}")
        return None

def get_memory_usage():
    """Get current memory usage in MB."""
    if psutil is None:
        return 0  # Return 0 if psutil is not available
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # Convert to MB

def should_cleanup():
    """Check if cleanup is needed based on memory usage."""
    try:
        memory_usage = get_memory_usage()
        # Increase threshold for cloud environments
        threshold = 2000 if os.environ.get('CLOUD_ENVIRONMENT') else 1000
        return memory_usage > threshold  # Cleanup if memory usage exceeds threshold
    except Exception as e:
        st.warning(f"Error checking memory usage: {str(e)}")
        return False

def cleanup_resources():
    """Clean up all resources when the session ends."""
    try:
        # Clear Streamlit's cache
        st.cache_data.clear()
        st.cache_resource.clear()
        
        # Clear PyTorch's cache only if CUDA is available and torch is imported
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Clear temporary files
        temp_dir = "temp"
        if os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
                os.makedirs(temp_dir)
            except Exception as e:
                st.warning(f"Could not clean temp directory: {str(e)}")
        
        # Clear session state selectively
        keys_to_keep = ['rag_system', 'documents_loaded', 'vector_store']
        for key in list(st.session_state.keys()):
            if key not in keys_to_keep:
                del st.session_state[key]
            
        # Force garbage collection
        gc.collect()
            
    except Exception as e:
        st.warning(f"Error during cleanup: {str(e)}")

def type_text(text, container, speed=0.01):
    """Type out text with a typing effect"""
    full_text = ""
    for char in text:
        full_text += char
        container.markdown(full_text)
        time.sleep(speed)

def generate_answer(question, is_follow_up=False):
    try:
        st.session_state.processing = True
        start_time = time.time()
        
        # Get the global internet setting
        use_internet = st.session_state.get('use_internet', False)
        
        # Check if we can process the question
        if not st.session_state.documents_loaded and not use_internet:
            answer = "No documents have been uploaded. Please either upload documents using the file uploader in the sidebar or enable internet search to get answers."
            st.session_state.main_answer = answer
            st.session_state.processing = False
            return
        
        # Check if RAG system is properly initialized
        if not hasattr(st.session_state, 'rag_system') or st.session_state.rag_system is None:
            answer = "RAG system not initialized. Please refresh the page."
            st.session_state.main_answer = answer
            st.session_state.processing = False
            return
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        answer_container = st.empty()  # Container for the typing effect
        
        # Combine document search and processing
        process_start = time.time()
        progress_bar.progress(25)
        status_text.text("🤔 Processing your question...")
        
        # Get document-based answer with reduced context
        if is_follow_up and 'main_answer' in st.session_state:
            # Limit the previous context to a shorter version
            prev_context = f"Previous Q: {st.session_state.question[:100]}...\nA: {st.session_state.main_answer[:200]}...\nFollow-up: {question}"
        else:
            prev_context = question
        
        # Only search documents if they are loaded
        results = []
        if st.session_state.documents_loaded:
            results = st.session_state.rag_system.vector_store.search(question, k=3)
        
        # Process and filter results
        processed_results = []
        seen_sources = set()
        seen_texts = set()  # Track unique text content
        
        for result in results:
            source = result.get('metadata', {}).get('source', 'Unknown source')
            text = result.get('text', '')
            
            # Skip if we've seen this source or this exact text before
            if source in seen_sources or text in seen_texts:
                continue
        
            seen_sources.add(source)
            seen_texts.add(text)
            
            # Calculate relevance score
            raw_score = result.get('score', 0)
            
            # Normalize score to 0-1 range
            normalized_score = (raw_score + 1) / 2  # Convert to 0-1 range
            
            # Boost score for exact matches in source name
            question_terms = set(question.lower().split())
            source_terms = set(source.lower().split())
            
            # Check for exact matches in source name
            if any(term in source_terms for term in question_terms):
                normalized_score = max(normalized_score, 0.9)  # Higher boost for exact matches
            
            # Additional boost for financial reports and official documents
            if any(keyword in source.lower() for keyword in ['financial', 'report', 'filing', 'sec', 'annual', 'quarterly']):
                normalized_score = max(normalized_score, 0.85)
            
            # Additional boost for recent documents
            if 'date' in result.get('metadata', {}):
                doc_date = result['metadata']['date']
                if doc_date:
                    try:
                        doc_date = datetime.strptime(doc_date, '%Y-%m-%d')
                        days_old = (datetime.now() - doc_date).days
                        if days_old < 365:  # Boost for documents less than a year old
                            normalized_score = max(normalized_score, 0.8)
                    except:
                        pass
            
            processed_results.append({
                'metadata': result.get('metadata', {}),
                'score': normalized_score,
                'text': text
            })
        
        # Sort by score and take top 5
        processed_results.sort(key=lambda x: x['score'], reverse=True)
        results = processed_results[:5]  # Increased to 5 sources
        
        # Generate answer with optimized parameters
        if use_internet:
            internet_context = f"""You are an AI assistant with access to the internet.
            Provide a comprehensive answer using your knowledge and internet access.
            Make sure to:
            1. Cite sources for all factual information
            2. Mention if you can't find a source for specific claims
            3. Focus on accurate, up-to-date information
            4. Be clear about what information comes from your training vs internet sources
            
            Question: {question}"""
            
            try:
                # Use direct LLM call for internet search
                answer = st.session_state.rag_system.question_handler.llm.generate_answer(question, internet_context)
            except Exception as e:
                st.error(f"Internet search failed: {str(e)}")
                answer = "I'm sorry, I encountered an error while trying to search the internet. Please try again or disable internet search."
        else:
            if not st.session_state.documents_loaded:
                answer = "I cannot answer this question as there are no documents loaded. Please either upload documents or enable internet search."
            else:
                # Build context from document results
                doc_context = "You are an investment banking analyst. Use ONLY the following document information to answer the question. If the documents don't contain relevant information, say so clearly.\n\n"
                doc_context += f"Question: {question}\n\n"
                doc_context += "Document Information:\n"
                
                if results:
                    for i, result in enumerate(results, 1):
                        source = result.get('metadata', {}).get('source', 'Unknown source')
                        text = result.get('text', '')
                        doc_context += f"Source {i}: {source}\n"
                        doc_context += f"Content: {text}\n\n"
                else:
                    doc_context += "No relevant documents found.\n\n"
                
                doc_context += "Please provide a comprehensive answer based on the document information above. If you cannot answer the question with the provided documents, clearly state this limitation."
                
                answer = st.session_state.rag_system.question_handler.process_question(doc_context)
        
        # Store the answer
        if is_follow_up:
            st.session_state.follow_up_answer = answer
        else:
            st.session_state.main_answer = answer
            st.session_state.main_results = results
        
        progress_bar.progress(100)
        status_text.text("✅ Done!")
        
        # Clear status after a shorter delay
        time.sleep(0.5)
        status_text.empty()
        progress_bar.empty()

    except Exception as e:
        st.error(f"Error generating answer: {str(e)}")
        # Provide a fallback answer
        if is_follow_up:
            st.session_state.follow_up_answer = "I'm sorry, I encountered an error while processing your question. Please try again."
        else:
            st.session_state.main_answer = "I'm sorry, I encountered an error while processing your question. Please try again."
    finally:
        st.session_state.processing = False

def show_main_page():
    """Show the main page with file upload and question input."""
    try:
        # Initialize session state variables if they don't exist
        if 'question' not in st.session_state:
            st.session_state.question = ""
        if 'follow_up_question' not in st.session_state:
            st.session_state.follow_up_question = ""
        if 'current_follow_up_input' not in st.session_state:
            st.session_state.current_follow_up_input = ""
        if 'main_answer' not in st.session_state:
            st.session_state.main_answer = None
        if 'follow_up_answer' not in st.session_state:
            st.session_state.follow_up_answer = None
        if 'processing' not in st.session_state:
            st.session_state.processing = False
        
        # Check if RAG system is properly initialized
        if not hasattr(st.session_state, 'rag_system') or st.session_state.rag_system is None:
            st.error("RAG system not initialized. Please refresh the page.")
            return
        
        # Main content
        st.markdown("""
            <div style='text-align: center; padding: 2rem 0; display: flex; flex-direction: column; align-items: center;'>
                <h1 style='color: #1E3A8A; font-size: 3.5rem; font-weight: 700; margin: 0; text-shadow: 0 0 10px rgba(30, 58, 138, 0.3);'>HERON</h1>
            </div>
        """, unsafe_allow_html=True)
        
        # Question input
        question = st.text_input(
            label="Ask a question",
            value=st.session_state.question,
            label_visibility="collapsed",
            placeholder="Ask a question...",
            key="question_input"
        )

        # Internet toggle (global setting) - positioned right after question input
        st.session_state.use_internet = st.toggle("Internet Search", help="Enable internet search for all questions in this session")

        # Process question if enter is pressed
        if question and question != st.session_state.question and not st.session_state.processing:
            try:
                st.session_state.question = question
                st.session_state.processing = True
                
                # Initialize variables
                answer_container = st.empty()  # Container for the typing effect
                is_follow_up = False  # Flag for follow-up questions
                
                # Show processing status
                with st.spinner("Processing your question..."):
                    generate_answer(question, is_follow_up)
                        
                # Display the answer
                if st.session_state.main_answer:
                    st.markdown(st.session_state.main_answer)

            except Exception as e:
                st.error(f"Error processing question: {str(e)}")
                st.info("Please try again or rephrase your question.")
            finally:
                st.session_state.processing = False

        # Display sources if they exist (but not the answer since it's already shown)
        if hasattr(st.session_state, 'main_results') and st.session_state.main_results:
            st.markdown("### Sources")
            for i, result in enumerate(st.session_state.main_results, 1):
                source = result.get('metadata', {}).get('source', 'Unknown source')
                score = result.get('score', 0)
                relevance_percentage = round(score * 100, 2)
                st.markdown(f"{i}. **{source}** (Relevance: {relevance_percentage}%)")

        # Add follow-up question section if we have a main answer
        if hasattr(st.session_state, 'main_answer') and st.session_state.main_answer:
            st.markdown("---")
            st.markdown("### Follow-up Question")
            
            # Initialize follow-up questions list if it doesn't exist
            if 'follow_up_questions' not in st.session_state:
                st.session_state.follow_up_questions = []
            
            # Display all previous follow-up questions and answers
            for i, (q, a) in enumerate(st.session_state.follow_up_questions):
                st.markdown(f"**Follow-up {i+1}:** {q}")
                st.markdown(a)
                st.markdown("---")
            
            # Create the follow-up question widget
            follow_up_question = st.text_input(
                label="Ask a follow-up question",
                value=st.session_state.get('current_follow_up_input', ''),
                label_visibility="collapsed",
                placeholder="Ask a follow-up question...",
                key="current_follow_up_input"
            )

            # Process follow-up question if entered
            if follow_up_question and follow_up_question != st.session_state.get('current_follow_up_input', '') and not st.session_state.processing:
                try:
                    st.session_state.processing = True
                    
                    # Show processing status
                    with st.spinner("Processing your follow-up question..."):
                        generate_answer(follow_up_question, is_follow_up=True)
                    
                    # Add to follow-up questions list
                    if st.session_state.follow_up_answer:
                        st.session_state.follow_up_questions.append((follow_up_question, st.session_state.follow_up_answer))
                    
                    # Clear the input and rerun to show the new follow-up
                    st.session_state.current_follow_up_input = ""
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Error processing follow-up question: {str(e)}")
                    st.info("Please try again or rephrase your question.")
                finally:
                    st.session_state.processing = False

        # Only show reset and export buttons if there's an answer
        if hasattr(st.session_state, 'main_answer') and st.session_state.main_answer:
            st.markdown("---")
            # Center the buttons with equal width
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                # Create a container for the buttons
                button_container = st.container()
                with button_container:
                    # Create two equal columns for the buttons
                    button_col1, button_col2 = st.columns(2)
                    
                    # Define a common button style
                    button_style = """
                    <style>
                    div[data-testid="stButton"] button {
                        width: 100%;
                        min-width: 200px;
                    }
                    </style>
                    """
                    st.markdown(button_style, unsafe_allow_html=True)
                    
                    with button_col1:
                        if st.button("Reset Conversation", type="secondary", use_container_width=True):
                            # Clear all conversation-related session state
                            if 'main_answer' in st.session_state:
                                del st.session_state.main_answer
                            if 'main_results' in st.session_state:
                                del st.session_state.main_results
                            if 'question' in st.session_state:
                                st.session_state.question = ""
                            if 'follow_up_question' in st.session_state:
                                st.session_state.follow_up_question = ""
                            if 'follow_up_answer' in st.session_state:
                                del st.session_state.follow_up_answer
                            if 'follow_up_questions' in st.session_state:
                                del st.session_state.follow_up_questions
                            st.rerun()
                    
                    with button_col2:
                        if st.button("Export PDF", type="secondary", use_container_width=True):
                            try:
                                pdf_path = generate_pdf_summary()
                                if pdf_path:
                                    with open(pdf_path, "rb") as file:
                                        st.download_button(
                                            label="Download PDF",
                                            data=file,
                                            file_name=os.path.basename(pdf_path),
                                            mime="application/pdf",
                                            use_container_width=True
                                        )
                                    # Clean up the temporary file
                                    os.remove(pdf_path)
                            except Exception as e:
                                st.error(f"Error generating PDF: {str(e)}")
        
        st.markdown("</div>", unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
        st.info("Please refresh the page and try again.")

# Main app logic
if st.session_state.current_page == "settings":
    show_settings_page()
else:
    show_main_page()

# streamlit run app.py --server.maxUploadSize=200 --server.maxMessageSize=200 --server.enableCORS=false --server.enableXsrfProtection=false  --server.enableWebsocketCompression=false
