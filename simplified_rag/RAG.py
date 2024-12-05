import os
from os import listdir
from dotenv import load_dotenv
from os.path import isfile, join
from langchain.chains import RetrievalQA
from langchain.schema.document import Document
from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import DocArrayInMemorySearch

# Load API keys from .env
load_dotenv()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

if OPENAI_API_KEY == 'xxxxxxxx':
    raise ValueError("Please add your own OpenAI API key in the .env file by replacing 'xxxxxxxx' with your own key.")

# Load model and define embedding
llm = ChatOpenAI(temperature=0, model='gpt-3.5-turbo')
embeddings = OpenAIEmbeddings()

# Target folder for uploaded docs
target_folder = "./docs/"

def list_documents():
    """List all documents in the target folder."""
    return [f for f in listdir(target_folder) if isfile(join(target_folder, f))]

def load_selected_documents(selected_docs):
    """Load and process selected documents."""
    if not selected_docs:
        raise ValueError("No documents selected for processing.")
    
    # Fragment the document content to fit in the number of token limitations
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = []

    for doc_name in selected_docs:
        file_path = join(target_folder, doc_name)
        loader = PyPDFLoader(file_path)
        data = loader.load()
        split_docs.extend(text_splitter.split_documents(data))

    # Create the dataset from the documents
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

def query_openai_directly(query: str):
    """Query the OpenAI model directly using a message format."""
    response = llm([{"role": "user", "content": query}])  # Correctly format the input for Chat model
    return {"result": response['choices'][0]['message']['content']}  # Extract the content of the response

def generate_answer(selected_docs, query: str, method: str):
    """Generate an answer using the selected documents or OpenAI depending on the method."""
    if method == "knowledge_base":
        if selected_docs:
            # Load and process selected documents
            data_set = load_selected_documents(selected_docs)
        else:
            # If no documents selected, load all documents in the target folder
            all_docs = list_documents()
            if not all_docs:
                raise ValueError("No documents available for querying.")
            # Load all documents and create a dataset from them
            data_set = load_selected_documents(all_docs)
        
        # Retrieve information using the embeddings
        response = retrieve_info(data_set, query)
        return response

    elif method == "openai":
        # Query OpenAI directly
        response = query_openai_directly(query)
        return response

    else:
        return {"result": "Invalid method selected."}

# Example usage
if __name__ == "__main__":
    # List all documents in the folder
    all_docs = list_documents()

    if not all_docs:
        raise FileNotFoundError("No PDF documents found in the target folder.")

    print("Available Documents:")
    for idx, doc in enumerate(all_docs, start=1):
        print(f"{idx}. {doc}")

    # Allow user to select multiple documents
    selected_doc_indices = input("Enter the document numbers to include (comma-separated), or press Enter to use all documents: ")
    selected_indices = [int(idx.strip()) - 1 for idx in selected_doc_indices.split(",") if idx.strip().isdigit()]
    selected_docs = [all_docs[i] for i in selected_indices if 0 <= i < len(all_docs)]

    print("\nSelected Documents:")
    if selected_docs:
        for doc in selected_docs:
            print(f"- {doc}")
    else:
        print("No documents selected. Will use all available documents for querying.")

    # Get the query from the user
    user_query = input("\nEnter your search query: ")

    # Selecting a method
    method = input("Select query method (knowledge_base/OpenAI): ").strip().lower()
    method = "knowledge_base" if "knowledge" in method else "openai"  # Default to OpenAI if not specified

    # Generate the answer
    print("\nProcessing your request...")
    try:
        result = generate_answer(selected_docs, user_query, method)
        print("\nAnswer:")
        print(result['result'])
    except Exception as e:
        print(f"An error occurred: {e}")