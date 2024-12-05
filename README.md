# Document Processor Application

This application allows users to upload PDF documents and processes them to extract content and generate embeddings for subsequent use in a document search vector store.

## Features

- Upload PDF documents and extract text content.
- Split documents into smaller chunks for efficient processing.
- Generate embeddings for each document chunk using a specified language model.
- Search through the embedded documents based on user queries.

## Requirements

- Python 3.7 or higher
- Required dependencies (see `requirements.txt` for details)

## Installation

1. Clone the repository:

   ```bash
   git clone <YOUR_REPOSITORY>
   cd <YOUR_PROJECT_DIRECTORY>

   Create a virtual environment (optional but recommended):

Copy Code
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
Install the required packages:

Copy Code
pip install -r requirements.txt
Usage
Set Up Your Environment:

Make sure to set any necessary environment variables, including API keys for any external services (like OpenAI) if applicable.

Run the Application:

Copy Code
streamlit run app.py
Upload PDF Documents:

Open the application in your browser (usually at http://localhost:8501). Use the provided interface to upload PDF files.

Process and Embed Documents:

The application will extract text from the PDF, split it into chunks, and generate embeddings for each chunk.

Search Through Documents:

Utilize the search features of the application to find relevant documents based on the queries you provide.

Code Overview
document_processor.py
This module handles the core functionality of the application:

Document Processing:

Extracts text from uploaded PDFs and splits them into manageable chunks.
Embedding:

Generates embeddings using a specified language model (e.g., OpenAI's models).
Vector Store:

Stores embedded documents using a vector store to enable fast searching.
Example Usage of DocumentProcessor
Here’s a simplified example of how the DocumentProcessor class works:

Copy Code
import os
import io
from typing import List
from document_processor import DocumentProcessor  # Adjust the import based on your file structure

# Initialize the DocumentProcessor
processor = DocumentProcessor(target_folder="./docs/")

# Example of processing uploaded PDF files
uploaded_files = [...]  # List of io.BytesIO objects containing PDF file contents
processor.load_and_embed_documents(uploaded_files)

# Get uploaded documents
documents = processor.get_uploaded_documents()
print(documents)
Contributing
Contributions are welcome! If you have suggestions for improvements or new features, please open an issue or submit a pull request.

License
This project is licensed under the MIT License. See the LICENSE file for more details.

Contact
For any further questions or feedback, feel free to reach out through the repository’s contact page or open an issue.


### Notes:
- Please ensure to replace `<YOUR_REPOSITORY_URL>` and `<YOUR_PROJECT_DIRECTORY>` with the actual values for your project.
- You might want to include a `requirements.txt` file listing the dependencies required for this project.
- Make sure that module names and paths match your actual file structure if they differ.