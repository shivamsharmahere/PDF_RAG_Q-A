import streamlit as st
from app import process_pdf, init_llm, create_rag_chain, generate_response
from langchain_core.messages import HumanMessage, AIMessage

# Set page config
st.set_page_config(
    page_title="PDF Question-Answering System",
    page_icon="📚",
    layout="wide"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "db" not in st.session_state:
    st.session_state.db = None

if "retriever" not in st.session_state:
    st.session_state.retriever = None

if "llm" not in st.session_state:
    st.session_state.llm = None

# Create a sidebar for configuration and file upload
with st.sidebar:
    st.title("PDF RAG System")
    
    # File uploader
    uploaded_file = st.file_uploader("Upload a PDF document", type="pdf")
    
    # Model selection
    model_name = st.selectbox(
        "Select a model",
        options=["deepseek-r1-distill-qwen-32b","gemma2-9b-it","llama-3.1-8b-instant","llama-3.3-70b-versatile","mixtral-8x7b-32768"],
        index=0
    )
    
    # Process button
    process_button = st.button("Process PDF")
    
    # Restart button
    restart_button = st.button("Restart Chat")

# Handle restart
if restart_button:
    st.session_state.messages = []
    st.session_state.chat_history = []
    st.rerun()

# Process PDF if requested
if uploaded_file and process_button:
    with st.spinner("Processing PDF..."):
        # Process the PDF
        st.session_state.db, st.session_state.retriever = process_pdf(uploaded_file)
        
        # Initialize LLM
        st.session_state.llm = init_llm(model_name)
        
        st.success("PDF processed successfully! You can now ask questions about the document.")

# Display chat interface in the main area
st.title("📚 Document Chat Assistant")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Create a chat input
if prompt := st.chat_input("Ask a question about your document"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.write(prompt)
    
    # Check if PDF has been processed
    if st.session_state.db is None:
        with st.chat_message("assistant"):
            st.write("Please upload and process a PDF document first.")
        st.session_state.messages.append({"role": "assistant", "content": "Please upload and process a PDF document first."})
    else:
        # Display assistant response
        with st.chat_message("assistant"):
            # Create a placeholder for streaming output
            message_placeholder = st.empty()
            
            # Generate response
            response, context = generate_response(
                prompt, 
                st.session_state.llm, 
                st.session_state.retriever, 
                st.session_state.chat_history
            )
            
            # Show context if available
            if context:
                with st.expander("View retrieved context"):
                    st.write(context)
            
            # Display response
            message_placeholder.markdown(response)
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})
            
            # Update chat history for context
            st.session_state.chat_history.append(HumanMessage(content=prompt))
            st.session_state.chat_history.append(AIMessage(content=response))

# Add a footer
st.markdown("---")
st.markdown("PDF RAG Application | Built with Streamlit, LangChain, and Groq")
