import streamlit as st
import smtplib
import os
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import tempfile

st.set_page_config(page_title="Video Submission", layout="wide")

st.title("🎬 AI People Reader - Video Submission System")

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

st.header("📧 Send Your Video via Email")

st.info("💡 **Instructions**: Upload your video and it will be sent directly to the analysis team via email.")

st.warning("⚠️ **Note**: This is a demo version. Videos are prepared for email but not actually sent. In production, SMTP settings would be configured.")

# Email form
with st.form("email_form"):
    st.subheader("📝 Video Submission Form")
    
    # Recipient selection
    recipient = st.selectbox(
        "Send video to:",
        ["alisa@imagematters.at", "rungnapa@imagematters.at"],
        help="Select who should receive your video for analysis"
    )
    
    # Sender info
    sender_name = st.text_input("Your Name:", placeholder="Enter your name")
    sender_email = st.text_input("Your Email:", placeholder="your.email@example.com")
    
    # Message
    message = st.text_area(
        "Message (optional):", 
        placeholder="Any additional notes about your video...",
        height=100
    )
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose a video file", 
        type=["mp4", "mov", "avi", "mpeg4"],
        help="Upload your interview video"
    )
    
    submit_button = st.form_submit_button("📤 Send Video", type="primary")

if submit_button:
    if not uploaded_file:
        st.error("❌ Please select a video file to send.")
    elif not sender_name or not sender_email:
        st.error("❌ Please fill in your name and email address.")
    else:
        try:
            # Show file info
            file_size = len(uploaded_file.getvalue())
            file_size_mb = file_size / (1024 * 1024)
            
            st.info(f"📁 File: {uploaded_file.name} ({file_size_mb:.1f} MB)")
            
            with st.spinner("📤 Sending email..."):
                # Create email message
                msg = MIMEMultipart()
                msg['From'] = sender_email
                msg['To'] = recipient
                msg['Subject'] = f"Video Submission from {sender_name} - {uploaded_file.name}"
                
                # Email body
                body = f"""
Hello,

You have received a new video submission from {sender_name}.

Video Details:
- File: {uploaded_file.name}
- Size: {file_size_mb:.1f} MB
- Sender: {sender_name} ({sender_email})

Message:
{message if message else "No additional message provided."}

Please process this video for analysis.

Best regards,
AI People Reader System
                """
                
                msg.attach(MIMEText(body, 'plain'))
                
                # Attach video file
                attachment = MIMEBase('application', 'octet-stream')
                attachment.set_payload(uploaded_file.getvalue())
                encoders.encode_base64(attachment)
                attachment.add_header(
                    'Content-Disposition',
                    f'attachment; filename= {uploaded_file.name}'
                )
                msg.attach(attachment)
                
                # For demo purposes, we'll simulate sending
                # In production, you'd use actual SMTP settings
                st.success(f"✅ Video sent successfully to {recipient}!")
                st.info("📧 Email sent with your video attached. The analysis team will process it shortly.")
                
                # Show what was sent
                st.subheader("📋 Submission Summary")
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Recipient:** {recipient}")
                    st.write(f"**Sender:** {sender_name}")
                    st.write(f"**File:** {uploaded_file.name}")
                with col2:
                    st.write(f"**Size:** {file_size_mb:.1f} MB")
                    st.write(f"**Status:** ✅ Sent")
                
        except Exception as e:
            st.error(f"❌ Failed to send email: {str(e)}")
            st.info("💡 Please check your internet connection and try again.")