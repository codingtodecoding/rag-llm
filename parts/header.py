import streamlit as st

def display_header():
    st.markdown("""
        <div class='header'>
            <h1>Document Upload and Query Search</h1>
        </div>
    """, unsafe_allow_html=True)