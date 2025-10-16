import streamlit as st
from datetime import datetime
import tempfile
import os

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
    if os.path.exists(demo_video_path):
        st.success("✅ Demo video found!")
        st.video(demo_video_path)
    else:
        st.warning("⚠️ Demo video not found")

# Fixed file upload test with size limits
st.header("📤 File Upload Test")
st.info("⚠️ **Note**: For Render free tier, please upload files smaller than 50MB to avoid 502 errors")

uploaded_file = st.file_uploader(
    "Choose a file (max 50MB)", 
    type=['mp4', 'mov', 'jpg', 'png', 'txt', 'pdf'],
    help="Large files may cause 502 errors on free tier"
)

if uploaded_file:
    try:
        # Check file size BEFORE processing
        file_size = len(uploaded_file.getvalue())
        file_size_mb = file_size / (1024 * 1024)
        
        if file_size_mb > 50:
            st.error(f"⚠️ File too large: {file_size_mb:.1f}MB. Please use files under 50MB for free tier.")
        else:
            st.success(f"✅ File uploaded successfully!")
            st.write(f"**File:** {uploaded_file.name}")
            st.write(f"**Size:** {file_size_mb:.1f} MB")
            st.write(f"**Type:** {uploaded_file.type}")
            
            # Save to temp file safely
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp:
                tmp.write(uploaded_file.getvalue())
                temp_path = tmp.name
                st.info(f"File saved to: {temp_path}")
                
                # Clean up temp file
                os.unlink(temp_path)
                st.success("✅ File processed and cleaned up!")
                
    except Exception as e:
        st.error(f"⚠️ Upload error: {str(e)}")
        st.info("💡 Try a smaller file or different file type")

# Test with very small files
st.subheader("🧪 Test Small Files")
st.write("Try uploading a small text file or image (< 1MB) to test functionality:")

small_file = st.file_uploader(
    "Small file test (max 1MB)", 
    type=['txt', 'jpg', 'png'],
    help="Test with very small files first"
)

if small_file:
    try:
        file_size = len(small_file.getvalue())
        if file_size > 1024 * 1024:  # 1MB
            st.error("File too large for test")
        else:
            st.success(f"✅ Small file test successful! Size: {file_size} bytes")
            st.write(f"Content preview: {small_file.getvalue()[:100]}...")
    except Exception as e:
        st.error(f"Small file test error: {str(e)}")

st.write("---")
st.write("**Current time:**", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
st.write("**This is a minimal test version to verify Render deployment works.**")
st.write("**✅ App is running - the 502 error was during file upload, not deployment!**")