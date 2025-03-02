import os
from groq import Groq
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import (
    create_history_aware_retriever,
    create_retrieval_chain,
)
from langchain.chains.combine_documents import create_stuff_documents_chain

# Load environment variables
load_dotenv()

# Default settings
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 100
DEFAULT_K_VALUE = 5
DEFAULT_TEMPERATURE = 0.5
DEFAULT_MAX_TOKENS = 500

# Process PDF function
def process_pdf(uploaded_file, chunk_size=DEFAULT_CHUNK_SIZE, chunk_overlap=DEFAULT_CHUNK_OVERLAP, k_value=DEFAULT_K_VALUE):
    # Save the uploaded file to a temporary location
    temp_file_path = "temp_pdf.pdf"
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Load and process PDF
    loader = PyMuPDFLoader(temp_file_path)
    docs = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    documents = text_splitter.split_documents(docs)
    
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.from_documents(documents, embedding_model)
    retriever = db.as_retriever(search_kwargs={"k": k_value})
    
    # Clean up the temporary PDF file
    try:
        os.remove(temp_file_path)
    except:
        pass
    
    return db, retriever

# Initialize LLM
def init_llm(model_name, temperature=DEFAULT_TEMPERATURE, max_tokens=DEFAULT_MAX_TOKENS):
    return ChatGroq(
        api_key=os.environ["API_KEY"],
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens
    )

# Create RAG chain
def create_rag_chain(llm, retriever):
    # Create contextualize question prompt
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, just "
        "reformulate it if needed and otherwise return it as is."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ]
    )
    
    # Create history aware retriever
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )
    
    # Create QA prompt
    qa_system_prompt = (
        "You are an assistant for question-answering tasks about the provided PDF document. "
        "Use the following pieces of retrieved context to answer the question. "
        "If you don't know the answer, just say that you don't know. "
        "Keep your answers conversational and helpful."
    )
    
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="context"),
        ]
    )
    
    # Create the document chain
    document_chain = create_stuff_documents_chain(llm, qa_prompt)
    
    # Create the history-aware retrieval chain
    retrieval_chain = create_retrieval_chain(
        history_aware_retriever,
        document_chain,
    )
    
    return retrieval_chain, history_aware_retriever

# Function to generate a response to a user query
def generate_response(prompt, llm, retriever, chat_history):
    try:
        # First, try to understand if this is a follow-up question
        contextualized_message = llm.invoke(
            f"Given this chat history: {str(chat_history)}\n\n"
            f"And this follow-up question: {prompt}\n\n"
            f"Rewrite the follow-up question to be a standalone question that captures all context needed. "
            f"If it's already a standalone question, return it unchanged."
        )
        
        # Extract the string content from the AIMessage
        contextualized_query = contextualized_message.content if hasattr(contextualized_message, 'content') else str(contextualized_message)
        
        # Then retrieve relevant documents
        docs = retriever.invoke(contextualized_query)
        
        # Create context from documents
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # Generate response with history and context
        history_str = "\n".join([f"Human: {m.content}" if isinstance(m, HumanMessage) else f"AI: {m.content}" for m in chat_history])
        
        response = llm.invoke(
            f"Chat history:\n{history_str}\n\n"
            f"Context from document:\n{context}\n\n"
            f"Human: {prompt}\n"
            f"AI: "
        ).content
        
        return response, context
        
    except Exception as e:
        return f"An error occurred: {str(e)}", ""
