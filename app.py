import streamlit as st

st.title("Simple Test")
st.write("Hello World")

uploaded_file = st.file_uploader("Choose a file")
if uploaded_file:
    st.write("File uploaded:", uploaded_file.name)
    st.write("Size:", len(uploaded_file.getvalue()), "bytes")