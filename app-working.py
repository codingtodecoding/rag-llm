import os
import io
import streamlit as st
from dotenv import load_dotenv
from typing import List
from langchain.chains import RetrievalQA
from langchain.schema.document import Document
from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import DocArrayInMemorySearch
import openai

# Load API keys from .env file
load_dotenv()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

if OPENAI_API_KEY == 'xxxxxxxx':
    raise ValueError("Please add your own OpenAI API key in the .env file by replacing 'xxxxxxxx' with your own key.")

# Load model and embeddings
llm = ChatOpenAI(temperature=0, model='gpt-3.5-turbo')
embeddings = OpenAIEmbeddings()

# Target folder for uploaded docs
target_folder = "./docs/"
os.makedirs(target_folder, exist_ok=True)

# CSS for custom styling
st.markdown("""
<style>
    .title {
        color: #0056b3;
        font-size: 36px;
        text-align: center;
        margin-bottom: 20px;
    }
    .header {
        color: #007bff;
        font-size: 24px;
        margin-bottom: 10px;
    }
    .button {
        background-color: #007bff;
        color: white;
        border: none;
        padding: 10px 20px;
        border-radius: 5px;
        cursor: pointer;
    }
    .button:hover {
        background-color: #0056b3;
    }
    .footer {
        text-align: center;
        margin-top: 20px;
        font-size: 12px;
        color: #6c757d;
    }
    .upload {
        border: 2px dashed #007bff;
        border-radius: 5px;
        padding: 20px;
        margin-bottom: 20px;
        background-color: #f8f9fa;
        transition: background-color 0.3s;
    }
    .upload:hover {
        background-color: #e9ecef;
    }
</style>
""", unsafe_allow_html=True)

# Function to process uploaded PDF files
def process_pdf(file_content: bytes) -> str:
    """Extract text from the PDF file content."""
    with open("temp_uploaded.pdf", "wb") as temp_file:
        temp_file.write(file_content)

    loader = PyPDFLoader("temp_uploaded.pdf")
    data = loader.load()

    document_text = ""
    for page in data:
        document_text += page.page_content + "\n"

    return document_text

# Function to load documents and create embeddings for uploaded files
def load_and_embed_documents(uploaded_files: List[io.BytesIO]):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = []

    for uploaded_file in uploaded_files:
        file_name = uploaded_file.name

        if file_name.endswith('.pdf'):
            # Process PDF file content
            file_content = uploaded_file.getvalue()
            document_text = process_pdf(file_content)

            # Save the uploaded document to the target folder
            with open(os.path.join(target_folder, file_name), "wb") as f:
                f.write(file_content)

            # Split the document into smaller chunks
            split_docs.extend(text_splitter.split_documents([Document(page_content=document_text)]))
        else:
            st.warning(f"Unsupported file type: {file_name}")
            continue

    # Create the dataset from the documents and generate embeddings
    data_set = DocArrayInMemorySearch.from_documents(documents=split_docs, embedding=embeddings)
    return data_set

def retrieve_info(data_set: DocArrayInMemorySearch, query: str):
    """Retrieve information from the dataset using the query."""
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=data_set.as_retriever(),
        verbose=True,
    )
    output = qa.invoke(query)
    return output

def generate_answer(selected_docs: List[str], query: str, method: str):
    """Generate answer using selected documents or fallback to OpenAI if necessary."""
    if method == "knowledge_base":
        # If using a knowledge base
        data_set = load_and_embed_selected_documents(selected_docs)

        # Retrieve the answer using the dataset
        response = retrieve_info(data_set, query)

    elif method == "openai":
        openai.api_key = OPENAI_API_KEY
        # Prepare to query OpenAI directly
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": "You are a helpful assistant."},
                      {"role": "user", "content": query}],
            max_tokens=150,
            temperature=0.7,
        )
        
        # Extract the content from the response
        if response.choices and len(response.choices) > 0:
            response_content = response.choices[0].message['content']
        else:
            response_content = "No response received from OpenAI."

        response = {"result": response_content}

    else:
        response = {"result": "Invalid method selected."}

    return response

def load_and_embed_selected_documents(selected_docs: List[str]):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = []

    for doc_name in selected_docs:
        file_path = os.path.join(target_folder, doc_name)
        loader = PyPDFLoader(file_path)
        data = loader.load()
        split_docs.extend(text_splitter.split_documents(data))

    # Create the dataset from the documents
    data_set = DocArrayInMemorySearch.from_documents(documents=split_docs, embedding=embeddings)
    return data_set

# Streamlit interface
st.markdown("<h1 class='title'>Document Upload and Query Search</h1>", unsafe_allow_html=True)

# Create layout with two columns
col1, col2 = st.columns([1, 2])

# Left Column for Uploading Files
with col1:
    st.markdown("<div class='upload'><h2 class='header'>Upload Files</h2></div>", unsafe_allow_html=True)
    uploaded_files = st.file_uploader("Upload your PDF documents", type=["pdf"], accept_multiple_files=True)

    # Show uploaded files currently in the target folder
    st.subheader("Uploaded Documents")
    uploaded_docs = [f for f in os.listdir(target_folder) if os.path.isfile(os.path.join(target_folder, f))]
    selected_docs = st.multiselect("Select documents to include in the query:", uploaded_docs)

# Right Column for Query Input
with col2:
    st.markdown("<h2 class='header'>Query Input</h2>", unsafe_allow_html=True)
    user_query = st.text_input("Enter your search query:")
    method = st.radio("Select query method:", ("Knowledge Base", "OpenAI"))

    if st.button("Search", key="search_button"):
        st.write("Processing your request...")
        try:
            # Determine method for answering
            selected_method = "knowledge_base" if method == "Knowledge Base" else "openai"
            result = generate_answer(selected_docs, user_query, selected_method)
            st.markdown("<h3 class='header'>Answer:</h3>", unsafe_allow_html=True)
            st.write(result['result'])
        except Exception as e:
            st.error(f"An error occurred: {e}")

# Process uploaded files if any
if uploaded_files:
    # Process uploaded files to add them to the data folder
    load_and_embed_documents(uploaded_files)

# Footer
st.markdown("<div class='footer'>Created with ❤️ VAR </div>", unsafe_allow_html=True)