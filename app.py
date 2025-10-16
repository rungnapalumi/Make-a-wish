import os, tempfile, gc
from datetime import datetime
import streamlit as st
import cv2
import numpy as np

# ---- Memory optimization environment variables
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

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

st.set_page_config(layout="wide", page_title="AI People Reader - Memory Safe")

def save_upload_to_disk(uploaded, suffix):
    """Write upload to a temp file in small chunks (no big .read())."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        while True:
            chunk = uploaded.read(1024 * 1024)  # 1 MB
            if not chunk: break
            tmp.write(chunk)
        return tmp.name

def basic_video_processing(video_path):
    """Basic video processing without MediaPipe - just show frames."""
    cv2.setNumThreads(1)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Cannot open video")
    
    # Get video info
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    st.info(f"📹 Video Info: {frame_count} frames, {fps:.1f} FPS, {width}x{height}")
    
    # Show first frame as preview
    ret, frame = cap.read()
    if ret:
        # Convert BGR to RGB for Streamlit
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        st.image(frame_rgb, caption="First frame preview", use_column_width=True)
    
    cap.release()
    gc.collect()
    
    return {
        'frame_count': frame_count,
        'fps': fps,
        'width': width,
        'height': height,
        'duration': frame_count / fps
    }

def main():
    st.title("🎬 AI People Reader - Motion Detection & Analysis")
    st.markdown("Upload a video to detect and analyze motion patterns")
    
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
        st.header("📤 Upload Your Video (Memory-Safe)")
        
        if st.session_state['user_role'] == 'admin':
            st.success("👑 Admin Mode: Full analysis capabilities enabled")
        else:
            st.info("👤 User Mode: Upload videos for admin review")
        
        f = st.file_uploader("Upload video (limit 200MB per file)", type=["mp4","mov","m4v","avi"])
        
        if f:
            suffix = "." + f.name.split(".")[-1].lower()
            src_path = save_upload_to_disk(f, suffix)
            st.info(f"✅ Saved upload: {f.name}")
            
            try:
                # Basic video processing
                st.write("🔍 Analyzing video...")
                video_info = basic_video_processing(src_path)
                
                # Display video
                st.video(src_path)
                
                # Show video statistics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Duration", f"{video_info['duration']:.1f}s")
                with col2:
                    st.metric("Frames", f"{video_info['frame_count']:,}")
                with col3:
                    st.metric("FPS", f"{video_info['fps']:.1f}")
                with col4:
                    st.metric("Resolution", f"{video_info['width']}x{video_info['height']}")
                
                if st.session_state['user_role'] == 'admin':
                    st.success("✅ Admin: Video analysis complete!")
                    st.info("🚧 Advanced MediaPipe features coming soon!")
                else:
                    st.success("✅ Video uploaded successfully!")
                    st.info("📝 Your video will remain on the system until admin downloads and removes it.")
                    
            except Exception as e:
                st.error(f"⚠️ Processing error: {e}")
            finally:
                # Cleanup temp files
                try:
                    if os.path.exists(src_path):
                        os.unlink(src_path)
                except:
                    pass
                gc.collect()

if __name__ == "__main__":
    main()