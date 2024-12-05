import streamlit as st
from parts.footer import display_footer
def display_sidebar(doc_processor):
    # st.sidebar.title("Navigation")
    # st.sidebar.header("Menu")
    st.sidebar.markdown("[Home](#)")
    st.sidebar.markdown("[About Us](#)")

    st.sidebar.header("Upload Files")
    uploaded_files = st.sidebar.file_uploader("Upload your PDF documents", type=["pdf"], accept_multiple_files=True)
    
    # Show uploaded files currently in the target folder
    st.sidebar.subheader("Uploaded Documents")
    uploaded_docs = doc_processor.get_uploaded_documents()
    selected_docs = st.sidebar.multiselect("Select documents to include in the query:", uploaded_docs)

    # Display footer
    display_footer()
    return uploaded_files, selected_docs