import os.path
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

# loading API keys from env
load_dotenv()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

if OPENAI_API_KEY == 'xxxxxxxx':
    raise ValueError("Please add your own OpenAI API key in the .env file by replacing 'xxxxxxxx' with your own key.")

# loading model and defining embedding
llm = ChatOpenAI(temperature=0, model='gpt-3.5-turbo')
embeddings = OpenAIEmbeddings()

# get target folder for uploaded docs
target_folder = "./docs/"

def load_data_set(query: str):
    # fragmenting the document content to fit in the number of token limitations
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

    # get files from target directory
    my_file = [f for f in listdir(target_folder) if isfile(join(target_folder, f))]
    if not my_file:
        raise FileNotFoundError("No files found in the target folder.")
    
    my_file = target_folder + my_file[0]
    print(f"My file is {my_file}")

    # load uploaded pdf file
    loader = PyPDFLoader(my_file)
    data = loader.load()
    split_docs = text_splitter.split_documents(data)

    data_set = DocArrayInMemorySearch.from_documents(documents=split_docs, embedding=embeddings)

    return data_set


def retrieve_info(data_set: DocArrayInMemorySearch, query: str):
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=data_set.as_retriever(),
        verbose=True,
    )

    output = qa.invoke(query)

    return output


def generate_answer(query: str):
    data_set = load_data_set(query)
    response = retrieve_info(data_set, query)
    
    return response
