# File uploader
uploaded_file = st.file_uploader("Upload a PDF file", type=['pdf'])

# Local file input
local_file_path = st.text_input("Or enter local file path", "")

# Text input for query
query = st.text_input("Enter your question about the document")

# Model selection
model = st.selectbox(
    "Select a model",
    ["gpt-3.5-turbo", "gpt-4"],
    index=0
)

# Temperature slider
temperature = st.slider(
    "Set temperature (higher = more creative, lower = more focused)",
    min_value=0.0,
    max_value=1.0,
    value=0.7,
    step=0.1
)

# Number of results slider
num_results = st.slider(
    "Number of results to retrieve",
    min_value=1,
    max_value=10,
    value=3,
    step=1
)

# Chunk size slider
chunk_size = st.slider(
    "Chunk size for text processing",
    min_value=100,
    max_value=2000,
    value=1000,
    step=100
)

# Chunk overlap slider
chunk_overlap = st.slider(
    "Chunk overlap for text processing",
    min_value=0,
    max_value=500,
    value=200,
    step=50
)

# Submit button
submit = st.button("Submit")

# Add follow-up question section if we have a main answer or a follow-up answer
if (hasattr(st.session_state, 'main_answer') and st.session_state.main_answer) or (hasattr(st.session_state, 'follow_up_answer') and st.session_state.follow_up_answer):
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

    # Process follow-up question if button is clicked or enter is pressed
    if (ask_follow_up or (follow_up_question and follow_up_question != st.session_state.get('follow_up_question', ''))) and not st.session_state.processing:
        try:
            st.session_state.follow_up_question = follow_up_question
            st.session_state.processing = True
            
            # Initialize variables
            follow_up_container = st.empty()  # Container for the typing effect
            use_internet = st.session_state.get('use_internet', False)
            
            # Show processing status
            with st.spinner("Processing your follow-up question..."):
                # Get search results
                results = st.session_state.rag_system.vector_store.search(follow_up_question, k=3)
                
                # Process and filter results
                processed_results = []
                seen_sources = set()
                seen_texts = set()
                
                for result in results:
                    source = result.get('metadata', {}).get('source', 'Unknown source')
                    text = result.get('text', '')
                    
                    if source in seen_sources or text in seen_texts:
                        continue
                        
                    seen_sources.add(source)
                    seen_texts.add(text)
                    
                    raw_score = result.get('score', 0)
                    normalized_score = (raw_score + 1) / 2
                    
                    question_terms = set(follow_up_question.lower().split())
                    source_terms = set(source.lower().split())
                    
                    if any(term in source_terms for term in question_terms):
                        normalized_score = max(normalized_score, 0.9)
                    
                    if any(keyword in source.lower() for keyword in ['financial', 'report', 'filing', 'sec', 'annual', 'quarterly']):
                        normalized_score = max(normalized_score, 0.85)
                    
                    if 'date' in result.get('metadata', {}):
                        doc_date = result['metadata']['date']
                        if doc_date:
                            try:
                                doc_date = datetime.strptime(doc_date, '%Y-%m-%d')
                                days_old = (datetime.now() - doc_date).days
                                if days_old < 365:
                                    normalized_score = max(normalized_score, 0.8)
                            except:
                                pass
                    
                    processed_results.append({
                        'metadata': result.get('metadata', {}),
                        'score': normalized_score,
                        'text': text
                    })
                
                processed_results.sort(key=lambda x: x['score'], reverse=True)
                results = processed_results[:5]
                
                # Build context from previous questions and answers
                context = f"Previous question: {st.session_state.question}\nPrevious answer: {st.session_state.main_answer}"
                if hasattr(st.session_state, 'follow_up_question') and hasattr(st.session_state, 'follow_up_answer'):
                    context += f"\n\nFollow-up question: {st.session_state.follow_up_question}\nFollow-up answer: {st.session_state.follow_up_answer}"
                
                # Generate answer with previous context
                follow_up_answer = st.session_state.rag_system.question_handler.process_question(
                    f"{context}\n\nNew follow-up question: {follow_up_question}"
                )
                
                # Type out the answer
                type_text(follow_up_answer, follow_up_container)
                st.session_state.follow_up_answer = follow_up_answer
                
                # If internet search is enabled, do it in parallel
                if use_internet:
                    internet_context = """You are a document analysis expert with access to the internet.
                    Provide a concise answer using your knowledge and internet access.
                    Cite sources for data. If no source exists, mention that.
                    Focus on accurate, up-to-date information."""
                    
                    # Use ThreadPoolExecutor for parallel processing
                    with ThreadPoolExecutor(max_workers=2) as executor:
                        internet_future = executor.submit(
                            st.session_state.rag_system.question_handler.llm.generate_answer,
                            follow_up_question,
                            internet_context
                        )
                        internet_answer = internet_future.result()
                    
                    # Type out the internet results
                    type_text("\n\n### Internet Search Results\n" + internet_answer, follow_up_container)
                    st.session_state.follow_up_answer += "\n\n### Internet Search Results\n" + internet_answer
            
        except Exception as e:
            st.error(f"Error processing follow-up question: {str(e)}")
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

if submit:
    if uploaded_file is not None:
        # Create a temporary file to store the uploaded content
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        try:
            # Initialize the RAG system with the temporary file
            rag_system = RAGSystem(
                model_name=model,
                temperature=temperature,
                num_results=num_results,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            
            # Process the temporary file
            rag_system.process_file(tmp_file_path)
            
            # Get the answer
            answer = rag_system.get_answer(query)
            
            # Display the answer
            st.write("Answer:", answer)
            
            # Clean up the temporary file
            os.unlink(tmp_file_path)
            
        except Exception as e:
            st.error(f"Error processing uploaded file: {str(e)}")
            if os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)
            
    elif local_file_path:
        try:
            # Initialize the RAG system
            rag_system = RAGSystem(
                model_name=model,
                temperature=temperature,
                num_results=num_results,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            
            # Process the local file
            rag_system.process_file(local_file_path)
            
            # Get the answer
            answer = rag_system.get_answer(query)
            
            # Display the answer
            st.write("Answer:", answer)
            
        except Exception as e:
            st.error(f"Error processing local file: {str(e)}")
    else:
        st.warning("Please either upload a file or provide a local file path.") 
