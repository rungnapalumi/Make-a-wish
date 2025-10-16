import streamlit as st
from datetime import datetime

st.set_page_config(page_title="AI People Reader - Test", layout="wide")

st.title("🎬 AI People Reader - Test Version")
st.write("If you can see this, the app is working!")

# Simple test
st.success("✅ Streamlit is running successfully!")
st.info("This is the minimal test version to verify deployment works.")

# Your login system
with st.sidebar:
    st.header("🔐 Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    
    if st.button("Login"):
        if username == "admin" and password == "0108":
            st.success("✅ Admin login successful!")
            st.session_state['user_role'] = 'admin'
        else:
            st.success("✅ User login successful!")
            st.session_state['user_role'] = 'user'

# Demo video section
st.header("📹 Demo Video")
demo_video_path = "Video Interview Simulation.mp4"
if st.button("Check if demo video exists"):
    import os
    if os.path.exists(demo_video_path):
        st.success("✅ Demo video found!")
        st.video(demo_video_path)
    else:
        st.warning("⚠️ Demo video not found")

# Simple file upload test
st.header("📤 File Upload Test")
uploaded_file = st.file_uploader("Choose a file")
if uploaded_file:
    st.success(f"✅ File uploaded: {uploaded_file.name}")
    st.write(f"File size: {len(uploaded_file.getvalue())} bytes")

st.write("---")
st.write("**Current time:**", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
st.write("**This is a minimal test version to verify Render deployment works.**")