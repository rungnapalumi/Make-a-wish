import os, tempfile, subprocess, gc, csv
from datetime import datetime
import streamlit as st
import cv2
import numpy as np
import pandas as pd
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
import plotly.graph_objects as go
import plotly.io as pio
import matplotlib.pyplot as plt
import openpyxl

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

st.set_page_config(layout="wide", page_title="AI People Reader - Full Featured")

def save_upload_to_disk(uploaded, suffix):
    """Write upload to a temp file in small chunks (no big .read())."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        while True:
            chunk = uploaded.read(1024 * 1024)  # 1 MB
            if not chunk: break
            tmp.write(chunk)
        return tmp.name

def transcode_to_720p_mp4(in_path):
    """MOV/MP4 -> H.264 mp4, 720p, 24fps, faststart; returns new path."""
    out_path = in_path + ".720p.mp4"
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", in_path,
        "-vf", "scale='min(1280,iw)':-2",    # cap width at 1280, keep AR
        "-r", "24",                          # cap FPS
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "26",
        "-movflags", "+faststart",
        "-an",                               # drop audio (saves mem/CPU)
        out_path
    ]
    subprocess.run(cmd, check=True)
    return out_path

def memory_safe_process(video_path, out_path, csv_path):
    """Read-Process-Write per frame. No accumulation in memory."""
    cv2.setNumThreads(1)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Cannot open video")

    # Output writer (MP4V keeps mem low and is widely supported)
    w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    # CSV write streaming
    csvf = open(csv_path, "w", newline="", encoding="utf-8")
    writer = csv.writer(csvf)
    writer.writerow(["frame", "time_s", "movement", "confidence"])

    # ---- import mediapipe lazily to avoid early alloc
    try:
        import mediapipe as mp
        mp_pose = mp.solutions.pose
        mediapipe_available = True
    except ImportError:
        mediapipe_available = False
        st.warning("⚠️ MediaPipe not available - running in basic mode")

    frame_idx = 0
    
    if mediapipe_available:
        with mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as pose:
            pbar = st.progress(0.0, text="Processing frames with MediaPipe…")
            total = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 1

            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                # MediaPipe expects RGB
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = pose.process(rgb)

                # Draw light-weight landmarks
                if result.pose_landmarks:
                    for lm in result.pose_landmarks.landmark:
                        # Draw pose landmarks
                        x = int(lm.x * w); y = int(lm.y * h)
                        if 0 <= x < w and 0 <= y < h:
                            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
                    
                    # Simple movement detection based on landmark positions
                    landmarks = result.pose_landmarks.landmark
                    left_shoulder = landmarks[11]
                    right_shoulder = landmarks[12]
                    
                    # Calculate shoulder movement (simple example)
                    shoulder_diff = abs(left_shoulder.y - right_shoulder.y)
                    if shoulder_diff > 0.05:
                        movement = "active"
                        confidence = min(shoulder_diff * 10, 1.0)
                    else:
                        movement = "neutral"
                        confidence = 0.3
                else:
                    movement = "no_pose_detected"
                    confidence = 0.0

                # Write to CSV
                t = frame_idx / fps
                writer.writerow([frame_idx, f"{t:.2f}", movement, f"{confidence:.2f}"])

                out.write(frame)

                # Periodic GC to keep memory stable
                if frame_idx % 60 == 0:
                    del rgb
                    gc.collect()

                frame_idx += 1
                pbar.progress(min(frame_idx / total, 1.0))
    else:
        # Basic mode without MediaPipe
        pbar = st.progress(0.0, text="Processing frames (basic mode)…")
        total = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 1
        
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            
            t = frame_idx / fps
            movement = "neutral"
            confidence = 0.5
            writer.writerow([frame_idx, f"{t:.2f}", movement, f"{confidence:.2f}"])
            out.write(frame)
            
            if frame_idx % 60 == 0:
                gc.collect()
            
            frame_idx += 1
            pbar.progress(min(frame_idx / total, 1.0))

    cap.release(); out.release(); csvf.close()
    gc.collect()

def create_analysis_report(csv_path, candidate_name, assessment_date):
    """Create a comprehensive analysis report."""
    # Read CSV data
    df = pd.read_csv(csv_path)
    
    # Calculate statistics
    total_frames = len(df)
    active_frames = len(df[df['movement'] == 'active'])
    neutral_frames = len(df[df['movement'] == 'neutral'])
    no_pose_frames = len(df[df['movement'] == 'no_pose_detected'])
    
    avg_confidence = df['confidence'].mean()
    movement_percentage = (active_frames / total_frames) * 100
    
    # Create visualization
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df['time_s'], 
        y=df['confidence'],
        mode='lines',
        name='Movement Confidence',
        line=dict(color='green', width=2)
    ))
    
    fig.update_layout(
        title='Movement Analysis Over Time',
        xaxis_title='Time (seconds)',
        yaxis_title='Confidence Score',
        height=400
    )
    
    return {
        'total_frames': total_frames,
        'active_frames': active_frames,
        'neutral_frames': neutral_frames,
        'no_pose_frames': no_pose_frames,
        'avg_confidence': avg_confidence,
        'movement_percentage': movement_percentage,
        'chart': fig
    }

def main():
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
        st.header("📤 Upload Your Video (Full Featured)")
        
        if st.session_state['user_role'] == 'admin':
            st.success("👑 Admin Mode: Full analysis capabilities enabled")
        else:
            st.info("👤 User Mode: Upload videos for admin review")
        
        f = st.file_uploader("Upload video (limit 1GB per file)", type=["mp4","mov","m4v","avi"])
        
        if f:
            suffix = "." + f.name.split(".")[-1].lower()
            src_path = save_upload_to_disk(f, suffix)
            st.info(f"✅ Saved upload: {f.name}")
            
            try:
                # Transcode first to shrink memory footprint while decoding
                st.write("⚙️ Transcoding to 720p / 24fps for stable processing…")
                small_path = transcode_to_720p_mp4(src_path)
                
                # Output paths
                out_vid = small_path.replace(".mp4", ".out.mp4")
                out_csv = small_path.replace(".mp4", ".timestamps.csv")
                
                if st.session_state['user_role'] == 'admin':
                    st.write("🔍 Analyzing with MediaPipe (frame-by-frame, no caching)…")
                    memory_safe_process(small_path, out_vid, out_csv)
                    
                    st.success("✅ Analysis complete!")
                    
                    # Display results
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        st.video(out_vid)
                    
                    with col2:
                        # Create and display analysis report
                        analysis = create_analysis_report(out_csv, candidate_name, assessment_date)
                        
                        st.subheader("📊 Analysis Results")
                        st.metric("Total Frames", f"{analysis['total_frames']:,}")
                        st.metric("Active Movement", f"{analysis['movement_percentage']:.1f}%")
                        st.metric("Avg Confidence", f"{analysis['avg_confidence']:.2f}")
                        
                        st.plotly_chart(analysis['chart'], use_container_width=True)
                        
                        # Download buttons
                        with open(out_csv, "rb") as fh:
                            st.download_button(
                                "📥 Download timestamps CSV", 
                                fh, 
                                file_name=os.path.basename(out_csv),
                                mime="text/csv"
                            )
                        
                        with open(out_vid, "rb") as fh:
                            st.download_button(
                                "📥 Download processed video", 
                                fh, 
                                file_name=os.path.basename(out_vid),
                                mime="video/mp4"
                            )
                else:
                    st.success("✅ Video uploaded successfully!")
                    st.video(small_path)
                    st.info("📝 Your video will remain on the system until admin downloads and removes it.")
                    
            except Exception as e:
                st.error(f"⚠️ Processing error: {e}")
            finally:
                # Cleanup temp files
                gc.collect()

if __name__ == "__main__":
    main()