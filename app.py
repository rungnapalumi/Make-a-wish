import streamlit as st
import cv2
import numpy as np
import pandas as pd
import tempfile
import os
from datetime import datetime
import io
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
import plotly.graph_objects as go
import plotly.io as pio
import matplotlib.pyplot as plt
import openpyxl

# Custom CSS to change background color
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
    .stText {
        color: #FFFFFF;
    }
    .stTextInput > div > div > input {
        background-color: #FFFFFF;
        color: #000000;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #FFFFFF !important;
    }
    p, div {
        color: #FFFFFF;
    }
    .stSelectbox > div > div {
        background-color: #FFFFFF;
        color: #000000;
    }
    .stButton > button {
        color: #FFFFFF;
    }
    .stSuccess {
        color: #FFFFFF;
    }
    .stInfo {
        color: #FFFFFF;
    }
    .stWarning {
        color: #FFFFFF;
    }
    .stError {
        color: #FFFFFF;
    }
    </style>
""", unsafe_allow_html=True)

def main():
    st.set_page_config(page_title="AI People Reader - Motion Detection & Analysis", layout="wide")
    
    st.title("🎬 AI People Reader - Motion Detection & Analysis")
    st.markdown("Upload a video to detect and analyze motion patterns with skeleton overlay")
    
    # Candidate Name Input at the top
    st.subheader("📝 Report Details")
    col_name, col_date = st.columns(2)
    
    with col_name:
        # Initialize session state for candidate name if not exists
        if "candidate_name" not in st.session_state:
            st.session_state.candidate_name = "Khun"
        
        candidate_name = st.text_input(
            "Candidate Name", 
            value=st.session_state.candidate_name,
            key="candidate_name_input",
            help="Enter the candidate's name for the report"
        )
        
        # Update session state when input changes
        if candidate_name:
            st.session_state.candidate_name = candidate_name
    
    with col_date:
        assessment_date = st.text_input(
            "Assessment Date", 
            value=datetime.now().strftime("%m/%d/%Y"), 
            key="assessment_date_input",
            help="Assessment date (automatically set to today)"
        )
    
    st.divider()
    
    # Sidebar with login form
    with st.sidebar:
        st.header("🔐 Login")
        
        # Initialize session state for login if not exists
        if "username" not in st.session_state:
            st.session_state.username = ""
        if "password" not in st.session_state:
            st.session_state.password = ""
        
        username = st.text_input(
            "Username", 
            value=st.session_state.username,
            key="username_input",
            help="Enter your username"
        )
        
        password = st.text_input(
            "Password", 
            value=st.session_state.password,
            type="password",
            key="password_input",
            help="Enter your password"
        )
        
        # Update session state when inputs change
        if username:
            st.session_state.username = username
        if password:
            st.session_state.password = password
        
        # Login button
        if st.button("Login", type="primary", use_container_width=True):
            if username and password:
                if username == "admin" and password == "0108":
                    st.session_state['user_role'] = 'admin'
                    st.success("✅ Admin login successful!")
                else:
                    st.session_state['user_role'] = 'user'
                    st.success("✅ User login successful!")
            else:
                st.error("⚠️ Please enter both username and password")
        
        # Logout button
        if 'user_role' in st.session_state:
            if st.button("Logout", type="secondary", use_container_width=True):
                st.session_state.pop('user_role', None)
                st.session_state.pop('username', None)
                st.session_state.pop('password', None)
                st.rerun()
        
        st.markdown("---")
        st.markdown("**Note:** Use these credentials to access the video recording system.")
    
    # Demo Video Section
    st.header("📹 Demo Video")
    st.markdown("**Video Interview Simulation** - Sample video for demonstration")
    
    demo_video_path = "Video Interview Simulation.mp4"
    if os.path.exists(demo_video_path):
        st.video(demo_video_path)
        
        # Add Thai instructions under the video
            st.markdown("---")
        st.markdown("""
        เป็นที่ทราบกันดีอยู่ว่าในการสื่อสารนั้นคำพูดให้ข้อมูล ส่วนภาษากายที่ไม่ว่าจะเป็น สายตาการเคลื่อนไหว 
        ของลำตัว ศรีษะ มือ แขนและขา บอกผู้ฟังถึงอารมณ์และความรู้สึกของผู้พูด
        
        **AI People Reader App** นั้นถูกสร้างมาให้วิเคราะห์การเคลื่อนไหวโดยรวมในขณะที่สื่อสาร โดยมีวัดตามหลักการ 4 ประเภท ตามนี้
        
        1. ผู้พูดสามารถ engage ผู้ฟังหรือไม่
        2. ผู้พูดสามารถรับมือกับผู้ฟังหลายประเภทหรือไม่
        3. ผู้พูดสามารถโต้แย้ง ยืนกรานในสิ่งที่ตัวเองเชื่อหรือไม่
        4. ผู้พูดสามารถที่จะทำให้ผู้ฟังทำตามสิ่งที่ตัวเองต้องการ เพื่อให้ได้ผลที่ต้องการหรือไม่
        
        โดยเพื่อเป็นการช่วยให้ผู้ที่จัดจัดส่งคลิปวิดีโอมี guideline ทางเราได้ถ่ายคลิปที่มีคำถาม upload ให้แล้ว 
        ผู้ที่กำลังจะถ่ายคลิปเพียงแค่ต้องใช้ username และ password log in เข้าทาง laptop หรือและใช้มือถือในการถ่ายคลิปของตัวเองแบบแนวตั้ง 
        โดยรายละเอียดการถ่ายคลิป และ resolution นั้นอยู่ในหน้า **Upload Your Video** แล้ว
        
        **หมายเหตุ:** ในการตอบคำถามต่างๆขอให้ตอบอย่างละเอียด และเคลื่อนไหวเป็นธรรมชาติ
        """)
                    st.markdown("---")
    else:
        st.warning(f"⚠️ Demo video '{demo_video_path}' not found in the current directory.")
    
    st.divider()
    
    # Main content - different based on user role
    if 'user_role' not in st.session_state:
        st.header("🔒 Please Login")
        st.info("Please login using the sidebar to access video upload functionality.")
    else:
        st.header("Upload Your Own Video")
        
        if st.session_state['user_role'] == 'admin':
            st.success("👑 Admin Mode: Full analysis capabilities enabled")
            st.info("🚧 Motion detection features are being prepared. Basic video upload is available.")
        else:
            st.info("👤 User Mode: Upload videos for admin review")
        
        uploaded_file = st.file_uploader("Choose a video file", type=['mp4', 'mov', 'avi'], help="Maximum file size: 200MB (for free tier)")
        
        if uploaded_file is not None:
            try:
                # Check file size (200MB limit for free tier)
                file_size_mb = len(uploaded_file.read()) / (1024 * 1024)
                uploaded_file.seek(0)  # Reset file pointer
                
                if file_size_mb > 200:
                    st.error(f"⚠️ File too large: {file_size_mb:.1f}MB. Maximum allowed: 200MB (free tier limit)")
                                        else:
                    # Save uploaded file temporarily
                    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                    tfile.write(uploaded_file.read())
                    video_path = tfile.name
            except Exception as e:
                st.error(f"⚠️ Upload error: {str(e)}. Please try again or contact support.")
                video_path = None
            
            # Store video info in session state
            if 'uploaded_videos' not in st.session_state:
                st.session_state['uploaded_videos'] = []
            
            video_info = {
                'name': uploaded_file.name,
                'path': video_path,
                'uploaded_by': st.session_state.get('username', 'Unknown'),
                'upload_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'size': len(uploaded_file.getvalue())
            }
            
            # Add to uploaded videos list if not already there
            if video_info not in st.session_state['uploaded_videos']:
                st.session_state['uploaded_videos'].append(video_info)
            
            st.video(video_path)
            
            # Show different buttons based on user role
            if st.session_state['user_role'] == 'admin':
                # Admin gets a placeholder analysis button
                if st.button("🔍 Complete Analysis (Coming Soon)", type="primary", use_container_width=True):
                    st.info("🚧 Advanced motion detection features are being prepared. Basic functionality is working!")
            else:
                # Regular users just see upload confirmation
                st.success("✅ Video uploaded successfully! Admin will review and analyze your video.")
                st.info("📝 Your video will remain on the system until admin downloads and removes it.")
        
        # Admin Video Management Section
        if st.session_state['user_role'] == 'admin' and 'uploaded_videos' in st.session_state and st.session_state['uploaded_videos']:
            st.markdown("---")
            st.header("📁 Uploaded Videos Management")
            st.markdown("**Videos uploaded by users:**")
            
            for i, video_info in enumerate(st.session_state['uploaded_videos']):
                with st.expander(f"📹 {video_info['name']} - Uploaded by {video_info['uploaded_by']} at {video_info['upload_time']}", expanded=False):
                    col1, col2, col3 = st.columns([2, 1, 1])
                    
        with col1:
                        st.write(f"**File:** {video_info['name']}")
                        st.write(f"**Uploaded by:** {video_info['uploaded_by']}")
                        st.write(f"**Time:** {video_info['upload_time']}")
                        st.write(f"**Size:** {video_info['size']:,} bytes")
                        
                        # Show video
                        if os.path.exists(video_info['path']):
                            st.video(video_info['path'])
                    
                    with col2:
                        # Download button
                        if os.path.exists(video_info['path']):
                            with open(video_info['path'], 'rb') as f:
        st.download_button(
                                    label="📥 Download Video",
                                    data=f.read(),
                                    file_name=video_info['name'],
                                    mime="video/mp4",
                                    key=f"download_{i}"
                                )
                    
                    with col3:
                        # Remove button
                        if st.button("🗑️ Remove", key=f"remove_{i}", type="secondary"):
                            # Remove from session state
                            st.session_state['uploaded_videos'].pop(i)
                            # Try to delete the file
                            try:
                                if os.path.exists(video_info['path']):
                                    os.unlink(video_info['path'])
                            except:
                                pass
                            st.rerun()

if __name__ == "__main__":
    main()
