import streamlit as st
import os
from local_draft import RAGSystem, WebFileHandler

# Set page config
st.set_page_config(
    page_title="HERON",
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
        
        # Check if user is asking to see images
        if any(word in question.lower() for word in ['show', 'display', 'image', 'picture', 'chart', 'graph']):
            return handle_image_request(question)
        
        # Use the logic from local_draft
        answer = st.session_state.rag_system.question_handler.process_question(question)
        
        # Store in conversation history using logic layer
        st.session_state.rag_system.add_to_conversation_history(question, answer, "initial")
        
        return answer
        
    except Exception as e:
        return f"Error: {str(e)}"

# Handle image display requests
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
        st.error(f"Error handling image request: {str(e)}")
        return []

# Follow-up question generation
def generate_follow_up(follow_up_question):
    try:
        if not st.session_state.documents_loaded:
            return "No documents loaded. Please upload documents first."
        
        # Check if user is asking to see images
        if any(word in follow_up_question.lower() for word in ['show', 'display', 'image', 'picture', 'chart', 'graph']):
            return handle_image_request(follow_up_question)
        
        # Use the logic from local_draft
        answer = st.session_state.rag_system.question_handler.process_follow_up(follow_up_question)
        
        # Store in conversation history using logic layer
        st.session_state.rag_system.add_to_conversation_history(follow_up_question, answer, "follow_up")
        
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
st.title("HERON")

# Display conversation history using logic layer
conversation_history = st.session_state.rag_system.get_conversation_history()
if conversation_history:
    st.subheader("Conversation History")
    for i, conv in enumerate(conversation_history):
        with st.expander(f"Q{i+1}: {conv['question'][:50]}..."):
            st.write(f"**Question:** {conv['question']}")
            # If the answer is a list (image info), display images
            if isinstance(conv['answer'], list):
                if conv['answer']:
                    st.write(f"Found {len(conv['answer'])} image(s):")
                    for img_info in conv['answer']:
                        if os.path.exists(img_info['path']):
                            st.image(img_info['path'], caption=f"Page {img_info['page']}, Image {img_info['image_num']}", use_container_width=True)
                else:
                    st.write("No images were found in the uploaded documents.")
            else:
                st.write(f"**Answer:** {conv['answer']}")

# Current question input
if not conversation_history:
    # First question
    question = st.text_input("Ask a question about your documents:")
    
    if st.button("Get Answer", type="primary"):
        if st.session_state.documents_loaded:
            with st.spinner("Processing..."):
                answer = generate_answer(question)
                # If answer is a list (image info), display images
                if isinstance(answer, list):
                    if answer:
                        st.write(f"Found {len(answer)} image(s):")
                        for img_info in answer:
                            if os.path.exists(img_info['path']):
                                st.image(img_info['path'], caption=f"Page {img_info['page']}, Image {img_info['image_num']}", use_container_width=True)
                    else:
                        st.write("No images were found in the uploaded documents.")
                else:
                    st.write(answer)
                st.rerun()  # Refresh to show follow-up input
        else:
            st.error("Please upload documents first")
else:
    # Follow-up question
    follow_up_question = st.text_input("Ask a follow-up question:")
    
    if st.button("Ask Follow-up", type="primary"):
        if st.session_state.documents_loaded:
            with st.spinner("Processing follow-up..."):
                follow_up_answer = generate_follow_up(follow_up_question)
                # If answer is a list (image info), display images
                if isinstance(follow_up_answer, list):
                    if follow_up_answer:
                        st.write(f"Found {len(follow_up_answer)} image(s):")
                        for img_info in follow_up_answer:
                            if os.path.exists(img_info['path']):
                                st.image(img_info['path'], caption=f"Page {img_info['page']}, Image {img_info['image_num']}", use_container_width=True)
                    else:
                        st.write("No images were found in the uploaded documents.")
                else:
                    st.write(follow_up_answer)
                st.rerun()  # Refresh to show next follow-up input
        else:
            st.error("Please upload documents first")

# Control buttons
st.markdown("---")
if st.button("Reset"):
    # Clear session state and conversation history
    st.session_state.rag_system.clear_conversation_history()
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun() 
