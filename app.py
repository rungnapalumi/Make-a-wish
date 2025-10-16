import streamlit as st
import os

st.set_page_config(page_title="AI People Reader", layout="wide")

st.title("🎬 AI People Reader")

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
    
    
    **หมายเหตุ:** ในการตอบคำถามต่างๆขอให้ตอบอย่างละเอียด และเคลื่อนไหวเป็นธรรมชาติ
    
    ---
    
    **📧 ส่งวิดีโอของคุณไปที่:**
    - alisa@imagematters.at
    - rungnapa@imagematters.at
    """)
    st.markdown("---")
else:
    st.warning(f"⚠️ Demo video '{demo_video_path}' not found in the current directory.")

# Second Demo Video Section - YouTube Example
st.header("🎥 ตัวอย่างการอัดวีดีโอ")
st.markdown("**ตัวอย่างการอัดวีดีโอที่ชัดเจนโดยเห็นทั้งลำตัวและการเคลื่อนไหวที่ชัดเจน**")
st.video("https://www.youtube.com/watch?v=R5JAHNbXPYs")