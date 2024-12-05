import openai
from langchain.chains import RetrievalQA
from document_processor import DocumentProcessor
from typing import List
import os 
class QueryHandler:
    def __init__(self):
        self.doc_processor = DocumentProcessor()

    def retrieve_info(self, query: str):
        """Retrieve information from the dataset using the query."""
        qa = RetrievalQA.from_chain_type(
            llm=self.doc_processor.llm,
            chain_type="stuff",
            retriever=self.doc_processor.data_set.as_retriever(),
            verbose=True,
        )
        output = qa.invoke(query)
        return output

    def generate_answer(self, selected_docs: List[str], query: str, method: str):
        """Generate answer using selected documents or fallback to OpenAI if necessary."""
        if method == "knowledge_base":
            self.doc_processor.load_and_embed_selected_documents(selected_docs)
            response = self.retrieve_info(query)
        elif method == "openai":
            openai.api_key = os.environ.get("OPENAI_API_KEY")
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "system", "content": "You are a helpful assistant."},
                          {"role": "user", "content": query}],
                max_tokens=150,
                temperature=0.7,
            )
            response_content = response.choices[0].message['content'] if response.choices else "No response received from OpenAI."
            response = {"result": response_content}
        else:
            response = {"result": "Invalid selected."}

        return response

    def load_and_embed_selected_documents(self, selected_docs: List[str]):
        """Load and embed selected documents."""
        return self.doc_processor.load_and_embed_documents(selected_docs)