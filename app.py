import streamlit as st

# Set page config must be the first Streamlit command
st.set_page_config(
    page_title="Herbert Insight AI",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': None
    }
)

import os
from local_draft import RAGSystem, LocalFileHandler, VectorStore, QuestionHandler, TextProcessor
import time
import fitz
import json
from datetime import datetime
import torch
import shutil
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
import psutil
import gc
import tempfile
from concurrent.futures import ThreadPoolExecutor

# Create necessary directories with proper error handling
def ensure_directories():
    """Create necessary directories with proper error handling."""
    directories = ['app_data', 'temp', 'local_documents']
    for directory in directories:
        try:
            os.makedirs(directory, exist_ok=True)
            # Test write permissions
            test_file = os.path.join(directory, '.test')
            with open(test_file, 'w') as f:
                f.write('test')
            os.remove(test_file)
        except Exception as e:
            st.error(f"Error creating/accessing directory {directory}: {str(e)}")
            st.error("Please ensure the application has write permissions.")
            return False
    return True

# Initialize directories
if not ensure_directories():
    st.error("Failed to initialize required directories. Some features may not work properly.")

def clear_caches():
    """Clear all caches and temporary files."""
    try:
        # Clear Streamlit's cache
        st.cache_data.clear()
        st.cache_resource.clear()
        
        # Clear PyTorch's cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Clear temporary files
        temp_dir = "temp"
        if os.path.exists(temp_dir):
            try:
                # First try to remove all files in the directory
                for file in os.listdir(temp_dir):
                    file_path = os.path.join(temp_dir, file)
                    try:
                        if os.path.isfile(file_path):
                            os.unlink(file_path)
                        elif os.path.isdir(file_path):
                            shutil.rmtree(file_path)
                    except Exception as e:
                        print(f"Error removing {file_path}: {e}")
                
                # Then try to remove the directory itself
                try:
                    os.rmdir(temp_dir)
                except OSError:
                    pass  # Ignore if directory is not empty
                
                # Create fresh temp directory
                os.makedirs(temp_dir, exist_ok=True)
            except Exception as e:
                print(f"Error clearing temp directory: {e}")
                # Create temp directory if it doesn't exist
                os.makedirs(temp_dir, exist_ok=True)
    except Exception as e:
        print(f"Error in clear_caches: {e}")
        # Ensure temp directory exists
        os.makedirs("temp", exist_ok=True)

# Clear caches on startup
try:
    clear_caches()
except Exception as e:
    print(f"Error during startup cache clearing: {e}")
    # Ensure temp directory exists
    os.makedirs("temp", exist_ok=True)

# Theme configuration
st.markdown("""
    <style>
        /* Main background */
        .stApp {
            background-color: #faebd7;
        }
        
        /* Sidebar background */
        [data-testid="stSidebar"] {
            background-color: #20293d;
        }
        
        /* Button styling */
        .stButton > button {
            background-color: #ff7f50;
            color: #ffffff;
        }
        
        .stButton > button:hover {
            background-color: #ff6b3d;
            color: #ffffff;
        }
        
        /* Text color in sidebar */
        [data-testid="stSidebar"] * {
            color: #a4a6b5;
        }

        /* Toggle switch label color */
        [data-testid="stToggle"] > label {
            color: #20293d !important;
        }

        /* App title */
        .app-title {
            color: #20293d !important;
        }

        /* Text input styling */
        .stTextInput > div > div > input {
            background-color: #ffffff;
            border: none !important;
            box-shadow: none !important;
            color: #20293d !important;
        }

        /* Remove focus outline */
        .stTextInput > div > div > input:focus {
            outline: none !important;
            border: none !important;
            box-shadow: none !important;
        }

        /* Markdown text color */
        .stMarkdown p, .stMarkdown li {
            color: #20293d !important;
        }

        /* Headers color */
        .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
            color: #20293d !important;
        }

        /* File uploader styling */
        .stFileUploader > div {
            background-color: #ffffff;
            border: 2px solid #ff7f50;
            color: #20293d;
        }

        /* Progress bar color */
        .stProgress > div > div {
            background-color: #ff7f50;
        }

        /* Main content area */
        .main-content {
            background-color: #faebd7;
        }

        /* Toggle background */
        [data-testid="stToggle"] {
            background-color: #ffffff;
        }

        /* Secondary button styling */
        .stButton > button[data-baseweb="button"][kind="secondary"] {
            background-color: #20293d;
            color: #a4a6b5;
        }

        .stButton > button[data-baseweb="button"][kind="secondary"]:hover {
            background-color: #2c3a52;
            color: #ffffff;
        }

        /* Company name styling */
        .company-name {
            font-size: 1.2rem;
            font-weight: 600;
            color: #a4a6b5;
            font-family: 'Helvetica Neue', sans-serif;
            margin-bottom: 2rem;
            text-align: center;
        }
    </style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown('<div class="company-name">Herbert Advisory</div>', unsafe_allow_html=True)
    
    # File uploader at the top of sidebar for better visibility
    st.markdown("### Upload Documents")
    try:
        uploaded_files = st.file_uploader(
            label="Upload your documents to analyze",
            type=['pdf', 'txt'],
            accept_multiple_files=True,
            help="Upload PDF or text files to analyze",
            label_visibility="visible",
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

def get_current_settings():
    """Get the current settings from session state or defaults"""
    if not hasattr(st.session_state, 'settings'):
        st.session_state.settings = {
            'doc_percentage': 10,  # Reduced from 15
            'chunk_size': 300,     # Reduced from 500
            'chunk_overlap': 30,   # Reduced from 50
            'model_temperature': 0.3,
            'sequence_length': 256, # Reduced from 512
            'batch_size': 64,      # Reduced from 128
            'use_half_precision': True,
            'speed_accuracy': 50,
            'num_results': 2       # Reduced from 3
        }
    return st.session_state.settings

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
    """Generate a PDF summary of the current conversation."""
    if not hasattr(st.session_state, 'main_answer') or not st.session_state.main_answer:
        return None
    
    # Create a temporary file for the PDF
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pdf_path = f"conversation_summary_{timestamp}.pdf"
    
    # Create the PDF document
    doc = SimpleDocTemplate(pdf_path, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    
    # Add title
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30
    )
    story.append(Paragraph("Conversation Summary", title_style))
    story.append(Spacer(1, 20))
    
    # Add timestamp
    story.append(Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
    story.append(Spacer(1, 20))
    
    # Add main question and answer
    story.append(Paragraph("Main Question:", styles['Heading2']))
    story.append(Paragraph(st.session_state.question, styles['Normal']))
    story.append(Spacer(1, 10))
    
    story.append(Paragraph("Answer:", styles['Heading2']))
    story.append(Paragraph(st.session_state.main_answer, styles['Normal']))
    story.append(Spacer(1, 20))
    
    # Add sources if available
    if hasattr(st.session_state, 'main_results') and st.session_state.main_results:
        story.append(Paragraph("Sources:", styles['Heading2']))
        for i, result in enumerate(st.session_state.main_results, 1):
            story.append(Paragraph(f"Source {i}: {result['metadata']['source']}", styles['Normal']))
            story.append(Paragraph(f"Relevance: {result['score']:.2f}", styles['Normal']))
            story.append(Spacer(1, 10))
    
    # Add follow-up if available
    if hasattr(st.session_state, 'follow_up_question') and st.session_state.follow_up_question:
        story.append(Paragraph("Follow-up Question:", styles['Heading2']))
        story.append(Paragraph(st.session_state.follow_up_question, styles['Normal']))
        story.append(Spacer(1, 20))
    
    # Build the PDF
    doc.build(story)
    
    return pdf_path

def get_memory_usage():
    """Get current memory usage in MB."""
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
        
        # Clear PyTorch's cache only if CUDA is available
        if torch.cuda.is_available():
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

def generate_answer(question, use_internet=False, is_follow_up=False):
    try:
        st.session_state.processing = True
        start_time = time.time()
        
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
        
        # Get search results
        results = st.session_state.rag_system.vector_store.search(question, k=3)  # Get more results initially
        
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
        answer = st.session_state.rag_system.question_handler.process_question(question)  # Use question directly instead of prev_context
        
        # Type out the answer
        type_text(answer, answer_container)
        
        if is_follow_up:
            st.session_state.follow_up_answer = answer
        else:
            st.session_state.main_answer = answer
            st.session_state.main_results = results
        
        # If internet search is requested, do it in parallel
        if use_internet:
            internet_context = """You are a document analysis expert with access to the internet.
            Provide a concise answer using your knowledge and internet access.
            Cite sources for data. If no source exists, mention that.
            Focus on accurate, up-to-date information."""
            
            internet_start = time.time()
            status_text.text("🌐 Searching the internet...")
            
            # Use ThreadPoolExecutor for parallel processing
            with ThreadPoolExecutor(max_workers=2) as executor:
                internet_future = executor.submit(
                    st.session_state.rag_system.question_handler.llm.generate_answer,
                    question,
                    internet_context
                )
                internet_answer = internet_future.result()
            
            # Type out the internet results
            type_text("\n\n### Internet Search Results\n" + internet_answer, answer_container)
            
            if is_follow_up:
                st.session_state.follow_up_answer += "\n\n### Internet Search Results\n" + internet_answer
            else:
                st.session_state.main_answer += "\n\n### Internet Search Results\n" + internet_answer
        
        progress_bar.progress(100)
        status_text.text("✅ Done!")
        
        # Clear status after a shorter delay
        time.sleep(0.5)
        status_text.empty()
        progress_bar.empty()

    except Exception as e:
        st.error(f"Error generating answer: {str(e)}")
    finally:
        st.session_state.processing = False

def generate_multi_analyst_answer(question, use_internet=False):
    """Generate an answer using multiple analysts with different perspectives."""
    try:
        st.session_state.processing = True
        start_time = time.time()
        
        # Create progress bar and status container
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Define different analyst perspectives
        analysts = {
            "Technical": {
                "role": "Technical Analyst",
                "focus": "Data-driven analysis, focusing on facts, metrics, and logical reasoning",
                "perspective": "You are a technical analyst focused on data, facts, and logical reasoning. Analyze the question from a technical, data-driven perspective."
            },
            "Creative": {
                "role": "Creative Analyst",
                "focus": "Innovative thinking and out-of-the-box solutions",
                "perspective": "You are a creative analyst focused on innovative solutions and thinking outside the box. Consider unique angles and creative approaches."
            },
            "Critical": {
                "role": "Critical Analyst",
                "focus": "Identifying potential issues, risks, and challenges",
                "perspective": "You are a critical analyst focused on identifying potential issues and challenges. Analyze potential problems and limitations."
            },
            "Strategic": {
                "role": "Strategic Analyst",
                "focus": "Long-term implications and big-picture thinking",
                "perspective": "You are a strategic analyst focused on long-term implications and big-picture thinking. Consider future impact and strategic value."
            }
        }
        
        # Get initial perspectives
        progress_bar.progress(25)
        status_text.text("🤔 Gathering different perspectives...")
        
        perspectives = {}
        for name, info in analysts.items():
            perspectives[name] = st.session_state.rag_system.question_handler.process_question(info["perspective"])
        
        # Display the debate
        progress_bar.progress(50)
        status_text.text("💭 Analysts are debating...")
        
        st.markdown("### Analyst Perspectives")
        for name, info in analysts.items():
            st.markdown(f"#### {info['role']}")
            st.markdown(f"*{info['focus']}*")
            st.markdown(perspectives[name])
            st.markdown("---")
        
        # Generate consensus
        progress_bar.progress(75)
        status_text.text("🤝 Reaching consensus...")
        
        consensus_context = f"""Based on the following perspectives, provide a clear and concise consensus that:
        1. Summarizes the key points from each analyst
        2. Identifies areas of agreement and disagreement
        3. Provides a clear, actionable conclusion
        4. Suggests next steps or recommendations
        
        Question: {question}
        
        Analyst Perspectives:
        {json.dumps(perspectives, indent=2)}
        """
        
        consensus = st.session_state.rag_system.question_handler.process_question(consensus_context)
        
        # Display the consensus
        st.markdown("### Consensus")
        st.markdown(consensus)
        
        # If internet search is requested
        if use_internet:
            internet_context = """You are a document analysis expert with access to the internet.
            Please provide additional context to the consensus, using your knowledge and internet access.
            Make sure to cite your sources for all data, and if you can't find a source, mention that it does not exist.
            Focus on providing accurate, up-to-date information from reliable sources."""
            
            internet_start = time.time()
            status_text.text("🌐 Searching the internet for additional information...")
            internet_answer = st.session_state.rag_system.question_handler.llm.generate_answer(question, internet_context)
            internet_time = time.time() - internet_start
            
            st.markdown("### Internet Search Results")
            st.markdown(internet_answer)
        
        progress_bar.progress(100)
        status_text.text("✅ Done!")
        total_time = time.time() - start_time
        
        # Display performance metrics
        with st.expander("Performance Metrics"):
            st.write(f"Total Processing Time: {total_time:.2f} seconds")
            if use_internet:
                st.write(f"Internet Search Time: {internet_time:.2f} seconds")
        
        # Clear the status after a short delay
        time.sleep(1)
        status_text.empty()
        progress_bar.empty()

    except Exception as e:
        st.error(f"Error generating answer: {str(e)}")
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
        if 'main_answer' not in st.session_state:
            st.session_state.main_answer = None
        if 'follow_up_answer' not in st.session_state:
            st.session_state.follow_up_answer = None
        
        # Main content area
        st.markdown("""
            <style>
            .main-content {
                padding: 2rem;
                max-width: 1200px;
                margin: 0 auto;
            }
            .app-title {
                font-size: 3.5rem;
                font-weight: 700;
                color: #20293d;
                text-align: center;
                margin-bottom: 0.5rem;
                font-family: 'Helvetica Neue', sans-serif;
                letter-spacing: 2px;
            }
            .app-subtitle {
                font-size: 1rem;
                color: #20293d;
                text-align: center;
                margin-bottom: 2rem;
                font-family: 'Helvetica Neue', sans-serif;
                font-style: italic;
            }
            </style>
            <div class="main-content">
            <div class="app-title">HERON</div>
            <div class="app-subtitle">(Herbert Embedded Retrieval and Oracle Network)</div>
        """, unsafe_allow_html=True)
        
        # Question input
        question = st.text_input(
            label="Ask a question",
            value=st.session_state.question,
            label_visibility="collapsed",
            placeholder="Ask a question...",
            key="question_input"
        )

        # Add Ask Question button
        col1, col2 = st.columns([3, 1])
        with col2:
            ask_button = st.button("Ask Question", type="primary", use_container_width=True)

        # Process question if button is clicked or enter is pressed
        if (ask_button or (question and question != st.session_state.question)) and not st.session_state.processing:
            try:
                st.session_state.question = question
                st.session_state.processing = True
                
                # Initialize variables
                answer_container = st.empty()  # Container for the typing effect
                is_follow_up = False  # Flag for follow-up questions
                use_internet = st.session_state.get('use_internet', False)  # Get internet toggle state
                
                # Show processing status
                with st.spinner("Processing your question..."):
                    # Get search results
                    results = st.session_state.rag_system.vector_store.search(question, k=3)  # Get more results initially
                    
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
                    answer = st.session_state.rag_system.question_handler.process_question(question)  # Use question directly
                    
                    # Type out the answer
                    type_text(answer, answer_container)
                    
                    if is_follow_up:
                        st.session_state.follow_up_answer = answer
                    else:
                        st.session_state.main_answer = answer
                        st.session_state.main_results = results
                    
                    # If internet search is requested, do it in parallel
                    if use_internet:
                        internet_context = """You are a document analysis expert with access to the internet.
                        Provide a concise answer using your knowledge and internet access.
                        Cite sources for data. If no source exists, mention that.
                        Focus on accurate, up-to-date information."""
                        
                        # Use ThreadPoolExecutor for parallel processing
                        with ThreadPoolExecutor(max_workers=2) as executor:
                            internet_future = executor.submit(
                                st.session_state.rag_system.question_handler.llm.generate_answer,
                                question,
                                internet_context
                            )
                            internet_answer = internet_future.result()
                        
                        # Type out the internet results
                        type_text("\n\n### Internet Search Results\n" + internet_answer, answer_container)
                        
                        if is_follow_up:
                            st.session_state.follow_up_answer += "\n\n### Internet Search Results\n" + internet_answer
                        else:
                            st.session_state.main_answer += "\n\n### Internet Search Results\n" + internet_answer
            
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
            col1, col2 = st.columns([3, 1])
            with col1:
                follow_up_question = st.text_input(
                    label="Ask a follow-up question",
                    value=st.session_state.get('follow_up_question', ''),
                    label_visibility="collapsed",
                    placeholder="Ask a follow-up question...",
                    key="follow_up_input"
                )
            with col2:
                ask_follow_up = st.button("Ask Follow-up", type="primary", use_container_width=True)

            # Process follow-up question if entered and button clicked
            if follow_up_question and ask_follow_up and not st.session_state.processing:
                try:
                    st.session_state.follow_up_question = follow_up_question
                    st.session_state.processing = True
                    
                    # Get the toggle states
                    use_internet = st.session_state.get('use_internet', False)
                    
                    # Show processing status
                    with st.spinner("Processing your follow-up question..."):
                        try:
                            # Get answer with previous context
                            follow_up_answer = st.session_state.rag_system.question_handler.process_question(
                                f"Previous question: {st.session_state.question}\nPrevious answer: {st.session_state.main_answer}\n\nFollow-up question: {follow_up_question}"
                            )
                            
                            if follow_up_answer:
                                st.session_state.follow_up_answer = follow_up_answer
                                
                                # If internet search is enabled, add internet results for follow-up
                                if use_internet:
                                    try:
                                        with st.spinner("Searching the internet..."):
                                            internet_context = """You are a document analysis expert with access to the internet.
                                            Provide a concise answer using your knowledge and internet access.
                                            Cite sources for data. If no source exists, mention that.
                                            Focus on accurate, up-to-date information."""
                                            
                                            internet_answer = st.session_state.rag_system.question_handler.llm.generate_answer(
                                                follow_up_question,
                                                internet_context
                                            )
                                            
                                            if internet_answer:
                                                st.session_state.follow_up_answer += "\n\n### Internet Search Results\n" + internet_answer
                                    except Exception as internet_error:
                                        st.error(f"Error during internet search: {str(internet_error)}")
                                        st.info("Continuing with document-based answer only.")
                                else:
                                    st.error("Failed to generate a follow-up answer. Please try again.")
                            
                        except Exception as process_error:
                            st.error(f"Error processing follow-up question: {str(process_error)}")
                            st.info("Please try again or rephrase your follow-up question.")
                        
                except Exception as e:
                    st.error(f"Error in follow-up processing: {str(e)}")
                    st.info("Please try again or rephrase your follow-up question.")
                finally:
                    st.session_state.processing = False
                    # Clean up any temporary files
                    try:
                        temp_dir = "temp"
                        if os.path.exists(temp_dir):
                            for file in os.listdir(temp_dir):
                                try:
                                    file_path = os.path.join(temp_dir, file)
                                    if os.path.isfile(file_path):
                                        os.unlink(file_path)
                                except Exception as e:
                                    print(f"Error deleting temp file {file}: {e}")
                    except Exception as cleanup_error:
                        print(f"Error during cleanup: {cleanup_error}")

            # Display follow-up answer if it exists
            if hasattr(st.session_state, 'follow_up_answer') and st.session_state.follow_up_answer:
                st.markdown("### Follow-up Answer")
                st.markdown(st.session_state.follow_up_answer)

        # Options
        st.session_state.use_internet = st.toggle("Internet")
        st.session_state.use_analysts = st.toggle("Multi-Analyst")
        
        # Only show reset and export buttons if there's an answer
        if hasattr(st.session_state, 'main_answer') and st.session_state.main_answer:
            st.markdown("---")
            col1, col2, col3 = st.columns([2, 1, 1])
            with col2:
                if st.button("Reset Conversation", type="secondary"):
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
                    st.rerun()
            
            with col3:
                if st.button("Export PDF", type="secondary"):
                    try:
                        pdf_path = generate_pdf_summary()
                        if pdf_path:
                            with open(pdf_path, "rb") as file:
                                st.download_button(
                                    label="Download PDF",
                                    data=file,
                                    file_name=os.path.basename(pdf_path),
                                    mime="application/pdf"
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
