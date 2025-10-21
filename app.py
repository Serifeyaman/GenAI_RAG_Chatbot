import streamlit as st
import os
from datasets import load_dataset
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.llms.base import LLM
from typing import Optional, List, Any
import google.generativeai as genai
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')

# Page configuration with light theme as default
st.set_page_config(
    page_title="Next.js RAG Assistant - Gemini 2.0",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Force light theme as default
st.markdown("""
    <style>
        .stApp {
            background-color: white;
        }
        .chat-message {
            padding: 1.5rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
            display: flex;
            flex-direction: column;
        }
        .user-message {
            background-color: #e3f2fd;
            border-left: 5px solid #2196F3;
        }
        .assistant-message {
            background-color: #f1f8e9;
            border-left: 5px solid #4CAF50;
        }
        .message-header {
            font-weight: bold;
            margin-bottom: 0.5rem;
            font-size: 0.9rem;
        }
        .message-time {
            font-size: 0.75rem;
            color: #666;
            margin-left: 0.5rem;
        }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state for chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Initialize session state for processing flag
if 'is_processing' not in st.session_state:
    st.session_state.is_processing = False

# Initialize session state for last question to prevent duplicates
if 'last_question' not in st.session_state:
    st.session_state.last_question = ""

# Gemini LLM Wrapper
class GeminiLLM(LLM):
    """Custom LangChain wrapper for Google Gemini 2.0"""
    model_name: str = "gemini-2.0-flash-exp"
    temperature: float = 0.7
    
    @property
    def _llm_type(self) -> str:
        return "gemini"
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        try:
            model = genai.GenerativeModel(self.model_name)
            response = model.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(
                    temperature=self.temperature,
                    max_output_tokens=2048,
                )
            )
            return response.text
        except Exception as e:
            return f"Error occurred: {str(e)}"
    
    @property
    def _identifying_params(self) -> dict:
        return {"model_name": self.model_name, "temperature": self.temperature}

# Cache functions for performance
@st.cache_resource
def load_and_prepare_data():
    """Load and prepare the Next.js dataset from Hugging Face"""
    with st.spinner("Loading dataset..."):
        # Load first 200 documents from Hugging Face dataset
        dataset = load_dataset("ChavyvAkvar/Next.js-Dataset-Converted", split="train[:200]")
        
        # Prepare documents
        documents = []
        for item in dataset:
            # Extract text content from dataset
            text_content = item.get('text', '') or item.get('content', '') or str(item)
            
            if text_content and len(text_content.strip()) > 0:
                doc = Document(
                    page_content=text_content,
                    metadata={"source": "Next.js Dataset"}
                )
                documents.append(doc)
        
        st.success(f"✅ {len(documents)} documents loaded successfully!")
        return documents

@st.cache_resource
def create_vector_store(_documents):
    """Create FAISS vector store with embeddings"""
    with st.spinner("Creating embeddings and building vector database..."):
        # Text splitter for chunking documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            length_function=len
        )
        
        # Split documents into chunks
        texts = text_splitter.split_documents(_documents)
        st.info(f"📄 {len(texts)} text chunks created")
        
        # Initialize embeddings model
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        
        # Create FAISS vector store
        vectorstore = FAISS.from_documents(texts, embeddings)
        
        st.success("✅ Vector database ready!")
        return vectorstore

def get_qa_chain(vectorstore, gemini_api_key, temperature):
    """Create QA chain with Gemini LLM and vector store retriever"""
    try:
        # Configure Gemini API
        genai.configure(api_key=gemini_api_key)
        
        # Initialize Gemini LLM
        llm = GeminiLLM(temperature=temperature)
        
        # Custom prompt template for friendly and technical responses
        prompt_template = """You are a friendly and helpful Next.js expert assistant. Use the context below to answer the user's question.

RULES:
- If the question is technical, provide detailed and professional explanations using the context
- If the user wants to chat, greet, or ask general questions, respond naturally and friendly
- If the answer is not in the context, say "I don't have enough information about this, but..." and share your general knowledge
- IMPORTANT: Respond in the SAME LANGUAGE as the user's question (English question → English answer, Turkish question → Turkish answer)
- Match the user's tone (formal questions get formal answers, casual questions get casual answers)
- When providing code examples, be explanatory

Context:
{context}

User Question: {question}

Your Answer:"""
        
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        # Create RetrievalQA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )
        
        return qa_chain
    except Exception as e:
        st.error(f"Error creating QA chain: {str(e)}")
        return None

def display_chat_history():
    """Display chat history in a chat-like interface"""
    for message in st.session_state.chat_history:
        if message['role'] == 'user':
            st.markdown(f"""
            <div class="chat-message user-message">
                <div class="message-header">
                    👤 You <span class="message-time">{message['timestamp']}</span>
                </div>
                <div>{message['content']}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="chat-message assistant-message">
                <div class="message-header">
                    🤖 Assistant <span class="message-time">{message['timestamp']}</span>
                </div>
                <div>{message['content']}</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Show sources if available
            if 'sources' in message and message['sources']:
                with st.expander(f"📚 View {len(message['sources'])} Source Documents"):
                    for i, source in enumerate(message['sources'], 1):
                        st.text(f"Source {i}:\n{source[:400]}...")

def add_to_chat_history(role, content, sources=None):
    """Add a message to chat history"""
    timestamp = datetime.now().strftime("%H:%M")
    message = {
        'role': role,
        'content': content,
        'timestamp': timestamp
    }
    if sources:
        message['sources'] = sources
    st.session_state.chat_history.append(message)

def main():
    """Main application function"""
    st.title("🤖 Next.js RAG Assistant")
    st.markdown("---")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Gemini API Key input
        gemini_api_key = st.text_input(
            "Google Gemini API Key",
            type="password",
            help="Get your free API key from Google AI Studio: https://aistudio.google.com/apikey"
        )
        
        # Temperature slider for creativity control
        temperature = st.slider(
            "Temperature (Creativity)",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.1,
            help="Lower: More deterministic, Higher: More creative"
        )
        
        st.markdown("---")
        st.markdown("### 📊 Model Information")
        st.info("**Model:** Gemini 2.0 Flash Exp")
        
        st.markdown("### 🔗 Resources")
        st.markdown("- [Google AI Studio](https://aistudio.google.com)")
        st.markdown("- [Next.js Dataset](https://huggingface.co/datasets/ChavyvAkvar/Next.js-Dataset-Converted)")
    
    # Main content area
    if not gemini_api_key:
        st.warning("⚠️ Please enter your Google Gemini API Key in the sidebar.")
        st.info("👉 Get your free API key: https://aistudio.google.com/apikey")
        
        # API key guide
        with st.expander("📖 How to Get an API Key?"):
            st.markdown("""
            1. Go to https://aistudio.google.com/apikey
            2. Sign in with your Google account
            3. Click "Create API Key" button
            4. Copy the generated API key
            5. Paste it in the sidebar field
            """)
        return
    
    # Load dataset and create vector store
    try:
        documents = load_and_prepare_data()
        vectorstore = create_vector_store(documents)
        qa_chain = get_qa_chain(vectorstore, gemini_api_key, temperature)
        
        if qa_chain is None:
            return
        
        st.success("✅ System ready! Start chatting with the Next.js Assistant.")
        
        # Sample questions section
        with st.expander("💡 Sample Questions"):
            st.markdown("""
            - How to implement server-side rendering in Next.js?
            - What are the differences between App Router and Pages Router?
            - How to create API routes in Next.js?
            - What is Static Site Generation (SSG)?
            - How does middleware work in Next.js?
            """)
        
        st.markdown("---")
        
        # Display chat history
        if st.session_state.chat_history:
            st.markdown("### 💬 Conversation")
            display_chat_history()
            st.markdown("---")
        
        # Question input section
        st.markdown("### ✍️ Ask Your Question")
        
        # Use columns for input and button
        question = st.text_area(
            "Your question:",
            placeholder="Example: How to implement server-side rendering in Next.js?",
            height=100,
            key="question_input"
        )
        
        col1, col2 = st.columns([1, 5])
        with col1:
            send_button = st.button("🚀 Send", use_container_width=True, type="primary", disabled=st.session_state.is_processing)
        
        # Process question when button is clicked (only once)
        if send_button and question:
            # Check if this is a new question (not a rerun of the same question)
            if question != st.session_state.last_question and not st.session_state.is_processing:
                # Set processing flag to prevent multiple submissions
                st.session_state.is_processing = True
                st.session_state.last_question = question
                
                # Add user question to chat history
                add_to_chat_history('user', question)
                
                # Get response from QA chain
                with st.spinner("🤔 Assistant is thinking..."):
                    try:
                        result = qa_chain({"query": question})
                        
                        # Extract sources
                        sources = []
                        if 'source_documents' in result and result['source_documents']:
                            sources = [doc.page_content for doc in result['source_documents']]
                        
                        # Add assistant response to chat history
                        add_to_chat_history('assistant', result['result'], sources)
                        
                    except Exception as e:
                        error_msg = f"❌ Error: {str(e)}\n\nPlease check your API key or try again."
                        add_to_chat_history('assistant', error_msg)
                
                # Reset processing flag
                st.session_state.is_processing = False
                
                # Force rerun to display new messages
                st.rerun()
        
        elif send_button and not question:
            st.warning("⚠️ Please enter a question.")
    
    except Exception as e:
        st.error(f"❌ Error starting application: {str(e)}")
        st.info("Please make sure all packages from requirements.txt are installed.")

if __name__ == "__main__":
    main()