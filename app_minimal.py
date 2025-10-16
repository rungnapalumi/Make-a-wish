import streamlit as st
import os

# Custom CSS
st.markdown("""
    <style>
    .stApp {
        background-color: #808080;
    }
    .main {
        background-color: #808080;
    }
    [data-testid="stSidebar"] {
        background-color: #696969;
    }
    .stMarkdown {
        color: #FFFFFF;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #FFFFFF !important;
    }
    p, div {
        color: #FFFFFF;
    }
    </style>
""", unsafe_allow_html=True)

def main():
    st.set_page_config(page_title="AI People Reader - Test", layout="wide")
    
    st.title("🎬 AI People Reader - Test Version")
    st.markdown("Testing basic functionality")
    
    # Demo Video Section
    st.header("📹 Demo Video")
    demo_video_path = "Video Interview Simulation.mp4"
    if os.path.exists(demo_video_path):
        st.video(demo_video_path)
        st.success("✅ Video loaded successfully!")
    else:
        st.warning(f"⚠️ Demo video '{demo_video_path}' not found.")
    
    # Login section
    with st.sidebar:
        st.header("🔐 Login Test")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        
        if st.button("Login"):
            if username == "admin" and password == "0108":
                st.success("✅ Admin login successful!")
            else:
                st.info("👤 User login successful!")
    
    st.info("🚧 This is a minimal test version to verify deployment works.")

if __name__ == "__main__":
    main()
