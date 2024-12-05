import streamlit as st

def display_footer():
    st.markdown(
        """
        <style>
            .footer {
                position: fixed;
                left: 0;
                bottom: 0;
                width: 100%;
                background-color: #f1f1f1;
                color: black;
                text-align: center;
                padding: 10px 0;
            }
        </style>
        <div class='footer'>Created with ❤️ by VAR</div>
        """,
        unsafe_allow_html=True
    )