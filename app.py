import streamlit as st
import os
from local_draft import RAGSystem, WebFileHandler

# Set page config
st.set_page_config(
    page_title="HERON - Fast Mode",
    page_icon="🦅",
    layout="wide"
)

# Initialize RAG system
def initialize_rag_system():
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = RAGSystem(is_web=True, use_vision_api=False)
        st.session_state.documents_loaded = False

# Simple answer generation
def generate_answer(question):
    try:
        if not st.session_state.documents_loaded:
            return "No documents loaded. Please upload documents first."
        
        # Simple document search
        results = st.session_state.rag_system.vector_store.search(question, k=3)
        
        if not results:
            return "No relevant information found in the documents."
        
        # Build simple context
        context = f"Question: {question}\n\nContext:\n"
        for i, result in enumerate(results[:3]):
            context += f"Source {i+1}: {result.get('text', '')}\n\n"
        
        context += "Answer the question based on the provided context."
        
        # Generate answer
        answer = st.session_state.rag_system.question_handler.process_question(context)
        return answer
        
    except Exception as e:
        return f"Error: {str(e)}"

# Main app
initialize_rag_system()

# Simple sidebar
with st.sidebar:
    st.header("Upload Documents")
    
    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=['pdf'],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        if st.session_state.rag_system.process_web_uploads(uploaded_files):
            st.success(f"Processed {len(uploaded_files)} file(s)")
            st.session_state.documents_loaded = True
        else:
            st.error("Failed to process files")

# Main content
st.title("HERON - Fast Document Analysis")

# Question input
question = st.text_input("Ask a question about your documents:")

# Always show all three buttons
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("Get Answer", type="primary"):
        if st.session_state.documents_loaded:
            with st.spinner("Processing..."):
                answer = generate_answer(question)
                st.write(answer)
        else:
            st.error("Please upload documents first")

with col2:
    if st.button("Clear"):
        st.rerun()

with col3:
    if st.button("Reset"):
        # Clear session state
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun() 
