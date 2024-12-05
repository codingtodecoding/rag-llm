import os
import shutil
import pathlib
import streamlit as st
from simplified_rag.RAG_old import generate_answer

st.title("RAG Model Benefit Demonstration")
st.header("\n A Simplified Approach")

# Clearing previously loaded pool of docs
target_folder = pathlib.Path().absolute() / "docs/"

if target_folder.exists():
    shutil.rmtree(target_folder)
target_folder.mkdir(parents=True, exist_ok=True)


def research_choice() -> str:
    with st.form(key="doc_upload", clear_on_submit=False):
        uploaded_doc = st.file_uploader(
            label="Please upload your document",
            accept_multiple_files=False,
            type=['pdf']
        )
        research_query = st.text_input(
            label="Please input what you want to search",
            max_chars=256
        )
        submit_button1 = st.form_submit_button("Load Document")

    if submit_button1 and uploaded_doc:
        with open(os.path.join(target_folder, uploaded_doc.name), 'wb') as f:
            f.write(uploaded_doc.getbuffer())
        return research_query


def main():
    research_query = research_choice()

    if research_query:
        with st.spinner("Processing your request..."):
            answer = generate_answer(research_query)

            st.success("Data processing complete!")
            st.write(answer['result'])


if __name__ == "__main__":
    main()
