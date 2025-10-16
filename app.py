import streamlit as st
import tempfile
import os

st.set_page_config(page_title="Video Upload", layout="wide")

st.title("🎬 AI People Reader - Motion Detection & Analysis")

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

st.header("🎥 Upload Your Video")

uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "mov", "avi", "mpeg4"])

if uploaded_file is not None:
    # Save file to a temp path
    suffix = os.path.splitext(uploaded_file.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        temp_path = tmp.name

    st.success(f"✅ File uploaded successfully: {uploaded_file.name}")
    st.video(temp_path)