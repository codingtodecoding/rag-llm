import os
import streamlit as st
from dotenv import load_dotenv
from document_processor import DocumentProcessor
from query_handler import QueryHandler
from parts.header import display_header
from parts.footer import display_footer
from parts.sidebar import display_sidebar

# Load API keys from .env file
load_dotenv()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

if OPENAI_API_KEY == 'xxxxxxxx':
    raise ValueError("Please add your own OpenAI API key in the .env file by replacing 'xxxxxxxx' with your own key.")

# Initialize document processor and query handler
doc_processor = DocumentProcessor()
query_handler = QueryHandler()

# Display custom CSS for background color
st.markdown(
    """
    <style>
        .main {
            # background-color: white;
            padding: 2rem;  /* Optional: Add padding for better spacing */
            
        }
       
        h1, h2, h3, h4, p {  /* Ensure heading and paragraph text is black */
            # color: black;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Display header
display_header()

# Display sidebar and get uploaded files and selected documents
uploaded_files, selected_docs = display_sidebar(doc_processor)

# Main content area for query input
st.markdown("<h2 class='header'>Query Input</h2>", unsafe_allow_html=True)
user_query = st.text_input("Enter your search query:")
method = st.radio("Select query method:", ("Knowledge Base", "OpenAI"))

if st.button("Search", key="search_button"):
    st.write("Processing your request...")
    try:
        selected_method = "knowledge_base" if method == "Knowledge Base" else "openai"
        result = query_handler.generate_answer(selected_docs, user_query, selected_method)
        st.markdown("<h3 class='header'>Answer:</h3>", unsafe_allow_html=True)
        st.write(result['result'])
    except Exception as e:
        st.error(f"An error occurred: {e}")

# Process uploaded files if any
if uploaded_files:
    doc_processor.load_and_embed_documents(uploaded_files)

# Display footer
display_footer()