import streamlit as st
import os

st.set_page_config(page_title="AI People Reader", layout="wide")

st.title("🎬 AI People Reader")

# Initialize session state
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'user_role' not in st.session_state:
    st.session_state.user_role = None
if 'users' not in st.session_state:
    st.session_state.users = {'admin': '0108'}  # Default admin user

# Login Section
if not st.session_state.authenticated:
    st.header("🔐 Login Required")
    
    with st.form("login_form"):
        st.subheader("Please login to access the content")
        username = st.text_input("Username:")
        password = st.text_input("Password:", type="password")
        login_button = st.form_submit_button("Login", type="primary")
        
        if login_button:
            if username in st.session_state.users and st.session_state.users[username] == password:
                st.session_state.authenticated = True
                if username == 'admin':
                    st.session_state.user_role = 'admin'
                else:
                    st.session_state.user_role = 'user'
                st.success("✅ Login successful!")
                st.rerun()
            else:
                st.error("❌ Invalid username or password")
    
    st.stop()  # Stop execution if not authenticated

# Logout and Admin Functions
col1, col2 = st.columns([1, 1])
with col1:
    if st.button("🚪 Logout"):
        st.session_state.authenticated = False
        st.session_state.user_role = None
        st.rerun()

with col2:
    if st.session_state.user_role == 'admin':
        st.write("👑 Admin Access")

# Admin Section - User Management
if st.session_state.user_role == 'admin':
    st.markdown("---")
    st.header("👑 Admin Panel - User Management")
    
    with st.expander("Create New User"):
        with st.form("create_user_form"):
            new_username = st.text_input("New Username:")
            new_password = st.text_input("New Password:", type="password")
            create_button = st.form_submit_button("Create User")
            
            if create_button:
                if new_username and new_password:
                    if new_username not in st.session_state.users:
                        st.session_state.users[new_username] = new_password
                        st.success(f"✅ User '{new_username}' created successfully!")
                    else:
                        st.error("❌ Username already exists!")
                else:
                    st.error("❌ Please fill in both username and password")
    
    # Display current users
    st.subheader("Current Users:")
    for username, password in st.session_state.users.items():
        st.write(f"👤 {username} {'(Admin)' if username == 'admin' else '(User)'}")

st.markdown("---")

# Demo Video Section
st.header("📹 Demo Video")
st.markdown("**Video Interview Simulation** - Sample video for demonstration")

demo_video_path = "Video Interview Simulation.mp4"
if os.path.exists(demo_video_path):
    st.video(demo_video_path)
    
    # Add Thai instructions under the video
    st.markdown("---")
    st.markdown("""
    เป็นที่ทราบกันดีอยู่ว่าในการสื่อสารนั้นคำพูดให้ข้อมูล ส่วนภาษากายที่ไม่ว่าจะเป็น สายตา การเคลื่อนไหวของลำตัว ศรีษะ มือ แขนและขา บอกผู้ฟังถึงอารมณ์และความรู้สึกของผู้พูด
    
    **AI People Reader App** นั้นถูกสร้างมาให้วิเคราะห์การเคลื่อนไหวโดยรวมในขณะที่สื่อสาร โดยมีวัดตามหลักการ 4 ประเภท ตามนี้
    
    1. ผู้พูดสามารถ engage ผู้ฟังหรือไม่
    2. ผู้พูดสามารถรับมือกับผู้ฟังหลายประเภทหรือไม่
    3. ผู้พูดสามารถโต้แย้ง ยืนกรานในสิ่งที่ตัวเองเชื่อหรือไม่
    4. ผู้พูดสามารถที่จะทำให้ผู้ฟังทำตามสิ่งที่ตัวเองต้องการ เพื่อให้ได้ผลที่ต้องการหรือไม่
    
    โดยเพื่อเป็นการช่วยให้ผู้ที่จัดส่งคลิปวิดีโอมี guideline ทางเราได้ถ่ายคลิปที่มีคำถาม upload ให้แล้ว 
    ผู้ที่กำลังจะถ่ายคลิปเพียงแค่ต้องใช้ username และ password log in เข้าทาง laptop หรือและใช้มือถือในการถ่ายคลิปของตัวเองแบบแนวตั้ง 
    
    
    **หมายเหตุ:** ในการตอบคำถามต่างๆขอให้ตอบอย่างละเอียด และเคลื่อนไหวเป็นธรรมชาติ
    
    ---
    
    **📧ส่งวิดีโอของคุณโดย upload ขึ้น Google Drive และ share มาที่ email ตามด้านล่าง**
    - alisa@imagematters.at
    - rungnapa@imagematters.at
    """)
    st.markdown("---")
else:
    st.warning(f"⚠️ Demo video '{demo_video_path}' not found in the current directory.")

# Second Demo Video Section - YouTube Example
st.header("🎥 ตัวอย่างการอัดวีดีโอ")
st.markdown("**ตัวอย่างการอัดวีดีโอที่ชัดเจนโดยเห็นทั้งลำตัวและการเคลื่อนไหวที่ชัดเจน**")
st.video("https://www.youtube.com/watch?v=c8wc8pEr-f0")