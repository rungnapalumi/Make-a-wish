import streamlit as st
import os
import json
import pandas as pd
import io

st.set_page_config(page_title="AI People Reader", layout="wide")


st.title("🎬 AI People Reader")

# User data file
USER_DATA_FILE = "users.json"

def load_users():
    """Load users from JSON file with persistent storage"""
    try:
        # First try to load from file
        if os.path.exists(USER_DATA_FILE):
            with open(USER_DATA_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Ensure admin user exists
                if 'admin' not in data:
                    data['admin'] = '0108'
                    save_users(data)  # Save the updated data
                return data
        else:
            # Create default file with admin user
            default_users = {'admin': '0108'}
            save_users(default_users)
            return default_users
    except Exception as e:
        print(f"❌ Error loading users: {e}")
        # Return default admin if any error
        return {'admin': '0108'}

def save_users(users):
    """Save users to JSON file with error handling"""
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(USER_DATA_FILE) if os.path.dirname(USER_DATA_FILE) else '.', exist_ok=True)
        
        # Create a backup of existing file
        if os.path.exists(USER_DATA_FILE):
            backup_file = f"{USER_DATA_FILE}.backup"
            with open(USER_DATA_FILE, 'r', encoding='utf-8') as original:
                with open(backup_file, 'w', encoding='utf-8') as backup:
                    backup.write(original.read())
        
        # Save new data
        with open(USER_DATA_FILE, 'w', encoding='utf-8') as f:
            json.dump(users, f, ensure_ascii=False, indent=2)
        
        print(f"✅ Users saved to {USER_DATA_FILE}: {len(users)} users")
        return True
    except Exception as e:
        print(f"❌ Failed to save users: {e}")
        # Try to restore from backup if available
        try:
            backup_file = f"{USER_DATA_FILE}.backup"
            if os.path.exists(backup_file):
                with open(backup_file, 'r', encoding='utf-8') as backup:
                    backup_data = json.load(backup)
                with open(USER_DATA_FILE, 'w', encoding='utf-8') as f:
                    json.dump(backup_data, f, ensure_ascii=False, indent=2)
                print("✅ Restored from backup")
        except:
            pass
        return False

def ensure_admin_exists():
    """Ensure admin user always exists in the system"""
    current_users = load_users()
    if 'admin' not in current_users:
        current_users['admin'] = '0108'
        save_users(current_users)
        print("✅ Admin user restored")
    return current_users

# Initialize session state with persistent user management
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'user_role' not in st.session_state:
    st.session_state.user_role = None
if 'users' not in st.session_state:
    st.session_state.users = ensure_admin_exists()
else:
    # Always ensure admin exists in session state
    if 'admin' not in st.session_state.users:
        st.session_state.users['admin'] = '0108'
        save_users(st.session_state.users)

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
                        save_success = save_users(st.session_state.users)  # Save to file
                        if save_success:
                            st.success(f"✅ User '{new_username}' created and saved successfully!")
                        else:
                            st.warning(f"⚠️ User '{new_username}' created but failed to save to file!")
                        st.rerun()  # Refresh to show updated user list
                    else:
                        st.error("❌ Username already exists!")
                else:
                    st.error("❌ Please fill in both username and password")
    
    # Display current users
    st.subheader("Current Users:")
    if st.session_state.users:
        for username, password in st.session_state.users.items():
            st.write(f"👤 {username} {'(Admin)' if username == 'admin' else '(User)'}")
    else:
        st.write("No users found.")
    
    # Excel Export/Import Section
    st.markdown("---")
    st.subheader("📊 Excel User Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**📤 Export Users to Excel**")
        if st.button("Download Excel File"):
            # Create DataFrame
            users_data = []
            for username, password in st.session_state.users.items():
                users_data.append({
                    'Username': username,
                    'Password': password,
                    'Role': 'Admin' if username == 'admin' else 'User'
                })
            
            df = pd.DataFrame(users_data)
            
            # Create Excel file in memory
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='Users', index=False)
            
            # Download button
            st.download_button(
                label="📥 Download users.xlsx",
                data=output.getvalue(),
                file_name="users.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    
    with col2:
        st.write("**📥 Import Users from Excel**")
        uploaded_file = st.file_uploader("Choose Excel file", type=['xlsx', 'xls'])
        
        if uploaded_file is not None:
            try:
                # Read Excel file
                df = pd.read_excel(uploaded_file)
                
                # Validate columns
                required_columns = ['Username', 'Password']
                if all(col in df.columns for col in required_columns):
                    st.success(f"✅ File loaded successfully! Found {len(df)} users.")
                    
                    # Show preview
                    st.write("**Preview:**")
                    st.dataframe(df.head())
                    
                    if st.button("Import Users"):
                        imported_count = 0
                        for _, row in df.iterrows():
                            username = str(row['Username']).strip()
                            password = str(row['Password']).strip()
                            
                            if username and password:
                                st.session_state.users[username] = password
                                imported_count += 1
                        
                        # Save to file
                        save_success = save_users(st.session_state.users)
                        if save_success:
                            st.success(f"✅ Imported {imported_count} users and saved successfully!")
                        else:
                            st.warning(f"⚠️ Imported {imported_count} users but failed to save to file!")
                        st.rerun()
                        
                else:
                    st.error("❌ Excel file must have 'Username' and 'Password' columns!")
                    
            except Exception as e:
                st.error(f"❌ Error reading Excel file: {str(e)}")
    
    # Manual Password Change Section
    st.markdown("---")
    st.subheader("🔑 Change User Password")
    
    with st.form("change_password_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            selected_user = st.selectbox("Select User:", list(st.session_state.users.keys()))
        
        with col2:
            new_password = st.text_input("New Password:", type="password")
        
        with col3:
            change_button = st.form_submit_button("Change Password")
        
        if change_button and new_password:
            if selected_user != 'admin' or st.session_state.user_role == 'admin':
                st.session_state.users[selected_user] = new_password
                save_success = save_users(st.session_state.users)
                if save_success:
                    st.success(f"✅ Password for '{selected_user}' changed and saved successfully!")
                else:
                    st.warning(f"⚠️ Password for '{selected_user}' changed but failed to save to file!")
                st.rerun()
            else:
                st.error("❌ Cannot change admin password!")
    
    # File Status Section
    st.markdown("---")
    st.subheader("💾 Storage Status")
    
    # Check if file exists and is readable
    file_exists = os.path.exists(USER_DATA_FILE)
    file_readable = False
    file_users = {}
    
    if file_exists:
        try:
            with open(USER_DATA_FILE, 'r', encoding='utf-8') as f:
                file_users = json.load(f)
                file_readable = True
        except:
            file_readable = False
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.write(f"**File Status:** {'✅ Exists' if file_exists else '❌ Missing'}")
    
    with col2:
        st.write(f"**File Readable:** {'✅ Yes' if file_readable else '❌ No'}")
    
    with col3:
        st.write(f"**Users in File:** {len(file_users)}")
    
    with col4:
        if st.button("🔄 Refresh Storage"):
            st.session_state.users = load_users()
            st.success("✅ Storage refreshed!")
            st.rerun()
    
    if file_users:
        st.write("**Users stored in file:**")
        for username, password in file_users.items():
            st.write(f"👤 {username} {'(Admin)' if username == 'admin' else '(User)'}")
    
    # Storage persistence info
    st.info("💡 **Note:** Users are saved permanently to the server. They will persist until manually deleted by an admin.")
    
    # Debug info (remove this in production)
    with st.expander("Debug Info"):
        st.write(f"Total users: {len(st.session_state.users)}")
        st.write(f"Users dict: {st.session_state.users}")
        st.write(f"File path: {USER_DATA_FILE}")
        st.write(f"File exists: {file_exists}")
        st.write(f"File readable: {file_readable}")

st.markdown("---")

# Demo Video Section
st.header("📹 Demo Video")

# Add Thai instructions above the video
st.markdown("""
เป็นที่ทราบกันดีอยู่ว่าในการสื่อสารนั้นคำพูดให้ข้อมูล ส่วนภาษากายที่ไม่ว่าจะเป็น สายตา การเคลื่อนไหวของลำตัว ศรีษะ มือ แขนและขา บอกผู้ฟังถึงอารมณ์และความรู้สึกของผู้พูด

**AI People Reader App** นั้นถูกสร้างมาให้วิเคราะห์การเคลื่อนไหวโดยรวมในขณะที่สื่อสาร โดยมีวัดตามหลักการ 4 ประเภท ตามนี้

1. ผู้พูดสามารถ engage ผู้ฟังหรือไม่
2. ผู้พูดสามารถรับมือกับผู้ฟังหลายประเภทหรือไม่
3. ผู้พูดสามารถโต้แย้ง ยืนกรานในสิ่งที่ตัวเองเชื่อหรือไม่
4. ผู้พูดสามารถที่จะทำให้ผู้ฟังทำตามสิ่งที่ตัวเองต้องการ เพื่อให้ได้ผลที่ต้องการหรือไม่

โดยเพื่อเป็นการช่วยให้ผู้ที่จัดส่งคลิปวิดีโอมี guideline ทางเราได้ถ่ายคลิปที่มีคำถาม upload ให้แล้ว 
ผู้ที่กำลังจะถ่ายคลิปเพียงแค่ต้องใช้ username และ password log in เข้าทาง laptop หรือและใช้มือถือในการถ่ายคลิปของตัวเองแบบแนวตั้ง 


**หมายเหตุ:ก่อนถ่ายวิดีโอรบกวนนำกระดาษ ดินสอและหนังสือมาหนึ่งเล่ม
ในการตอบคำถามต่างๆขอให้ตอบอย่างละเอียด และเคลื่อนไหวเป็นธรรมชาติ

---

**📧ส่งวิดีโอของคุณโดย upload ขึ้น Google Drive และ share มาที่ email ตามด้านล่าง**
- alisa@imagemattersasia.com
- petchpat@gmail.com
""")

st.markdown("---")
st.markdown("**Instruction** - คำแนะนำในการถ่ายวิดีโอ")

instruction_video_path = "Instruction.mp4"
if os.path.exists(instruction_video_path):
    try:
        st.video(instruction_video_path, format="video/mp4")
    except Exception as e:
        st.error(f"❌ Error playing instruction video: {str(e)}")
        st.info("💡 Try refreshing the page or using a different browser")
        # Alternative HTML5 video display
        st.markdown(f"""
        <video width="100%" controls>
            <source src="{instruction_video_path}" type="video/mp4">
            Your browser does not support the video tag.
        </video>
        """, unsafe_allow_html=True)
else:
    st.warning(f"⚠️ Instruction video '{instruction_video_path}' not found in the current directory.")

st.markdown("---")
st.header("🎥 ตัวอย่างการอัดวีดีโอ")
st.markdown("**สามารถดูตัวอย่างวิธีการตั้งกล้อง การเลือก background การถ่ายวีดีโอได้ที่**")

example_video_path = "example.mp4"
if os.path.exists(example_video_path):
    try:
        st.video(example_video_path, format="video/mp4")
    except Exception as e:
        st.error(f"❌ Error playing example video: {str(e)}")
        st.info("💡 Try refreshing the page or using a different browser")
        # Alternative HTML5 video display
        st.markdown(f"""
        <video width="100%" controls>
            <source src="{example_video_path}" type="video/mp4">
            Your browser does not support the video tag.
        </video>
        """, unsafe_allow_html=True)
else:
    st.warning(f"⚠️ Example video '{example_video_path}' not found in the current directory.")

st.markdown("---")
st.markdown("**Video Interview Simulation** - Sample video for demonstration")

demo_video_path = "interview simulation 4 min.mp4"
if os.path.exists(demo_video_path):
    try:
        st.video(demo_video_path, format="video/mp4")
    except Exception as e:
        st.error(f"❌ Error playing demo video: {str(e)}")
        st.info("💡 Try refreshing the page or using a different browser")
        # Alternative HTML5 video display
        st.markdown(f"""
        <video width="100%" controls>
            <source src="{demo_video_path}" type="video/mp4">
            Your browser does not support the video tag.
        </video>
        """, unsafe_allow_html=True)
else:
    st.warning(f"⚠️ Demo video '{demo_video_path}' not found in the current directory.")