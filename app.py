# app.py — AI People Reader "Make a Wish" Webapp
# ----------------------------------------------
# ฟังก์ชันหลัก:
# 1) ให้ผู้ใช้ upload วิดีโอสัมภาษณ์
# 2) ส่งไฟล์ขึ้น S3 + สร้างไฟล์ job JSON ใน jobs/pending/
# 3) ให้ ai-people-reader-worker ไปประมวลผลแล้วเขียนผลลัพธ์ที่ jobs/finished/{job_id}.json
# 4) หน้าเว็บสามารถใส่ job_id แล้วกด Check status เพื่อดูผลลัพธ์ / download report

import os
import io
import json
import uuid
from datetime import datetime, timezone
from typing import Optional, Dict, Any

import streamlit as st
import boto3
from botocore.exceptions import ClientError

# ----------------------------------------------------------
# Config AWS
# ----------------------------------------------------------
AWS_BUCKET = os.environ.get("AWS_BUCKET") or os.environ.get("S3_BUCKET")
AWS_REGION = os.environ.get("AWS_REGION", "ap-southeast-1")

if not AWS_BUCKET:
    raise RuntimeError("Missing AWS_BUCKET (or S3_BUCKET) environment variable")

s3 = boto3.client("s3", region_name=AWS_REGION)

# ----------------------------------------------------------
# Streamlit Config
# ----------------------------------------------------------
st.set_page_config(
    page_title="AI People Reader - Make a Wish",
    layout="wide",
)

# ----------------------------------------------------------
# Helper Functions
# ----------------------------------------------------------
def new_job_id() -> str:
    """สร้าง job id ใหม่"""
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    rand = uuid.uuid4().hex[:5]
    return f"{ts}__{rand}"


def upload_bytes_to_s3(data: bytes, key: str, content_type: Optional[str] = None) -> None:
    """อัปโหลด bytes ไป S3"""
    extra_args: Dict[str, Any] = {}
    if content_type:
        extra_args["ContentType"] = content_type

    s3.put_object(
        Bucket=AWS_BUCKET,
        Key=key,
        Body=data,
        **extra_args,
    )


def s3_key_exists(key: str) -> bool:
    """เช็คว่า key นี้มีอยู่ใน S3 ไหม"""
    try:
        s3.head_object(Bucket=AWS_BUCKET, Key=key)
        return True
    except ClientError as e:
        if e.response["Error"]["Code"] in ("404", "NoSuchKey", "NotFound"):
            return False
        # error อื่นให้ raise ต่อ
        raise


def read_json_from_s3(key: str) -> Dict[str, Any]:
    """อ่านไฟล์ JSON จาก S3"""
    obj = s3.get_object(Bucket=AWS_BUCKET, Key=key)
    data = obj["Body"].read().decode("utf-8")
    return json.loads(data)


def generate_presigned_url_from_key(
    key: str,
    expires_in: int = 3600,
) -> Optional[str]:
    """สร้าง presigned URL จาก S3 key (ถ้ามีสิทธิ์)"""
    try:
        url = s3.generate_presigned_url(
            "get_object",
            Params={"Bucket": AWS_BUCKET, "Key": key},
            ExpiresIn=expires_in,
        )
        return url
    except Exception:
        return None


# ----------------------------------------------------------
# UI Layout
# ----------------------------------------------------------
st.title("🎬 AI People Reader – Make a Wish")
st.markdown(
    """
ยินดีต้อนรับสู่ระบบ **AI People Reader – Make a Wish**  

1. อัปโหลดวิดีโอสัมภาษณ์ของคุณ  
2. ระบบจะสร้าง **Job ID** และส่งไฟล์ไปให้ AI ประมวลผล  
3. ใช้ Job ID เพื่อตรวจสอบสถานะ และดูผลลัพธ์เมื่อวิเคราะห์เสร็จ  

---
"""
)

if "job_id" not in st.session_state:
    st.session_state["job_id"] = ""


# ----------------------------------------------------------
# Section 1: Upload Video & Create Job
# ----------------------------------------------------------
st.header("① Upload Video for Analysis")

col_left, col_right = st.columns([2, 1])

with col_left:
    uploaded_file = st.file_uploader(
        "Upload interview video file",
        type=["mp4", "mov", "avi", "mkv"],
        help="รองรับไฟล์วิดีโอทั่วไป เช่น .mp4, .mov, .avi, .mkv",
    )

    note = st.text_area(
        "Optional note (สำหรับคุณครู/ผู้ประเมิน)",
        placeholder="เช่น Candidate A – Final Interview – Leadership Focus",
    )

    start_button = st.button("🚀 Submit for AI analysis")

with col_right:
    st.markdown("#### Tips")
    st.markdown(
        """
- วิดีโอควรมีความยาวไม่เกินตามที่กำหนดในระบบ
- ผู้พูดควรเห็นตัวเต็มครึ่งตัวขึ้นไป
- ใช้ Job ID ที่ได้เพื่อกลับมาตรวจสอบภายหลัง
"""
    )

# เมื่อกด submit
if start_button:
    if uploaded_file is None:
        st.warning("กรุณาอัปโหลดวิดีโอก่อนค่ะ")
    else:
        # อ่านไฟล์เป็น bytes
        file_bytes = uploaded_file.read()

        # สร้าง job_id
        job_id = new_job_id()
        st.session_state["job_id"] = job_id

        # เตรียม key ต่าง ๆ ใน S3
        # ปรับ path ได้ตามที่ worker.py คาดหวัง
        video_key = f"jobs/{job_id}/input/{uploaded_file.name}"
        job_key = f"jobs/pending/{job_id}.json"

        # 1) อัปโหลดไฟล์วิดีโอ
        upload_bytes_to_s3(
            data=file_bytes,
            key=video_key,
            content_type=uploaded_file.type,
        )

        # 2) สร้าง job record ให้ worker ใช้
        #    IMPORTANT: worker ต้องการ field ชื่อ "input_key"
        job_record: Dict[str, Any] = {
            "job_id": job_id,
            "input_key": video_key,   # <-- ให้ worker ใช้โหลดวิดีโอ
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "pending",
            "user_note": note,
            # ถ้า worker ใช้ค่าอื่น เช่น job_type, config, ฯลฯ
            # สามารถเติมเพิ่มได้ตรงนี้
            # "job_type": "dots_skeleton_report",
            # "params": {...},
        }

        upload_bytes_to_s3(
            data=json.dumps(job_record, ensure_ascii=False, indent=2).encode("utf-8"),
            key=job_key,
            content_type="application/json",
        )

        st.success(f"สร้างงานสำเร็จ! 🎉 Job ID ของคุณคือ: `{job_id}`")
        st.info("กรุณาจดหรือ copy Job ID ไว้เพื่อใช้ตรวจสอบผลภายหลัง")


st.markdown("---")

# ----------------------------------------------------------
# Section 2: Check Job Status & View Result
# ----------------------------------------------------------
st.header("② Check Job Status & View Report")

col1, col2 = st.columns([2, 1])

with col1:
    job_id_input = st.text_input(
        "Enter Job ID",
        value=st.session_state.get("job_id", ""),
        placeholder="เช่น 20260116__abc12",
    )

    check_btn = st.button("🔍 Check status")

with col2:
    st.markdown(
        """
##### วิธีใช้
1. วาง Job ID ที่คุณได้หลังจากอัปโหลดวิดีโอ  
2. กดปุ่ม **Check status**  
3. ถ้าประมวลผลเสร็จแล้ว ระบบจะแสดงสรุปและ link สำหรับดาวน์โหลด report  
"""
    )

if check_btn:
    if not job_id_input.strip():
        st.warning("กรุณากรอก Job ID ก่อนค่ะ")
    else:
        job_id_lookup = job_id_input.strip()

        finished_key = f"jobs/finished/{job_id_lookup}.json"
        pending_key = f"jobs/pending/{job_id_lookup}.json"
        failed_key = f"jobs/failed/{job_id_lookup}.json"

        # ตรวจสอบสถานะจากไฟล์ที่มีใน S3
        try:
            if s3_key_exists(failed_key):
                st.error("❌ งานนี้ประมวลผลไม่สำเร็จ (อยู่ใน failed).")
                failed_info = read_json_from_s3(failed_key)
                with st.expander("ดูรายละเอียด error (จาก worker)"):
                    st.json(failed_info)

            elif s3_key_exists(finished_key):
                result = read_json_from_s3(finished_key)
                st.success("✅ งานนี้ประมวลผลเสร็จสิ้นแล้ว! 🎉")

                # แสดงข้อมูลหลัก ๆ
                st.subheader("Result Summary")

                summary = result.get("summary") or result.get("message")
                if summary:
                    st.write(summary)

                scores = result.get("scores") or result.get("metrics")
                if scores:
                    st.subheader("Scores / Metrics")
                    st.json(scores)

                # หาก worker ส่ง S3 key ของ report มา
                report_url: Optional[str] = None

                # กรณี 1: มี URL โดยตรงใน JSON
                if "report_url" in result:
                    report_url = result["report_url"]

                # กรณี 2: มี S3 key แล้วเราสร้าง presigned URL เอง
                elif "report_s3_key" in result:
                    report_url = generate_presigned_url_from_key(
                        result["report_s3_key"], expires_in=3600
                    )

                if report_url:
                    st.markdown(f"[📄 Download full report]({report_url})")
                else:
                    st.info(
                        "ไม่มี URL ของ report ในผลลัพธ์ "
                        "กรุณาตรวจสอบรูปแบบ JSON ที่ worker เขียนกลับมาว่ามี field "
                        "`report_url` หรือ `report_s3_key` หรือไม่"
                    )

                with st.expander("ดู JSON ทั้งหมดที่ได้จาก worker"):
                    st.json(result)

            elif s3_key_exists(pending_key):
                st.info("🕒 งานนี้กำลังรอประมวลผล (อยู่ใน pending) กรุณาลองใหม่ภายหลัง")
            else:
                st.warning(
                    "ไม่พบ Job นี้ในระบบ (ไม่มีใน pending / finished / failed). "
                    "กรุณาตรวจสอบว่า Job ID ถูกต้องหรือไม่"
                )
        except ClientError as e:
            st.error(f"เกิดข้อผิดพลาดจากฝั่ง S3: {e}")
        except Exception as e:
            st.error(f"เกิดข้อผิดพลาดไม่คาดคิด: {e}")
