import os
import io
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain.schema.document import Document
from langchain.embeddings import OpenAIEmbeddings  # Ensure you have this import
from langchain.chat_models import ChatOpenAI  # Import the LLM class if you're using OpenAI's chat models

class DocumentProcessor:
    def __init__(self, target_folder="./docs/", model_name="gpt-3.5-turbo"):
        self.target_folder = target_folder
        os.makedirs(self.target_folder, exist_ok=True)
        self.embeddings = OpenAIEmbeddings()  # Initialize embeddings
        self.llm = ChatOpenAI(model_name=model_name)  # Initialize the LLM (set model_name as needed)
        self.data_set = None  # For storing the vector representation of documents

    def process_pdf(self, file_content: bytes) -> str:
        """Extract text from the PDF file content."""
        with open("temp_uploaded.pdf", "wb") as temp_file:
            temp_file.write(file_content)

        loader = PyPDFLoader("temp_uploaded.pdf")
        data = loader.load()
        document_text = "".join(page.page_content + "\n" for page in data)
        return document_text

    def load_and_embed_documents(self, uploaded_files: List[io.BytesIO]):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        split_docs = []

        for uploaded_file in uploaded_files:
            file_name = uploaded_file.name

            if file_name.endswith('.pdf'):
                file_content = uploaded_file.getvalue()
                document_text = self.process_pdf(file_content)

                # Save the uploaded document to the target folder
                with open(os.path.join(self.target_folder, file_name), "wb") as f:
                    f.write(file_content)

                # Split the document into smaller chunks
                split_docs.extend(text_splitter.split_documents([Document(page_content=document_text)]))
            else:
                raise ValueError(f"Unsupported file type: {file_name}")
        
        # Create the dataset from the documents and generate embeddings
        self.data_set = DocArrayInMemorySearch.from_documents(documents=split_docs, embedding=self.embeddings)

    def load_and_embed_selected_documents(self, selected_docs: List[str]):
        """Load and embed selected documents."""
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        split_docs = []

        for doc_name in selected_docs:
            file_path = os.path.join(self.target_folder, doc_name)
            loader = PyPDFLoader(file_path)
            data = loader.load()
            split_docs.extend(text_splitter.split_documents(data))

        # Create the dataset from the documents
        self.data_set = DocArrayInMemorySearch.from_documents(documents=split_docs, embedding=self.embeddings)

    def get_uploaded_documents(self):
        """Get the list of uploaded documents."""
        return [f for f in os.listdir(self.target_folder) if os.path.isfile(os.path.join(self.target_folder, f))]