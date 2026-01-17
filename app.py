# app.py  --- Streamlit frontend: Make-a-wish (หน้า user)
#
# หน้าที่:
# 1) ให้ user upload วิดีโอสัมภาษณ์ → สร้าง job "dots" → ส่งข้อมูลขึ้น S3
# 2) ให้ user ใส่ Job ID เพื่อตรวจสถานะ job และดาวน์โหลดวิดีโอที่ประมวลผลเสร็จแล้ว

import os
import json
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import boto3
from botocore.exceptions import ClientError
import streamlit as st

# ----------------------------------------------------------
# Config
# ----------------------------------------------------------

AWS_BUCKET = os.getenv("AWS_BUCKET") or os.getenv("S3_BUCKET")
AWS_REGION = os.getenv("AWS_REGION", "ap-southeast-1")

if not AWS_BUCKET:
    raise RuntimeError("Missing AWS_BUCKET (or S3_BUCKET) environment variable")

s3 = boto3.client("s3", region_name=AWS_REGION)

JOBS_PENDING_PREFIX = "jobs/pending/"
JOBS_PROCESSING_PREFIX = "jobs/processing/"
JOBS_FINISHED_PREFIX = "jobs/finished/"
JOBS_FAILED_PREFIX = "jobs/failed/"
JOBS_OUTPUT_PREFIX = "jobs/output/"

st.set_page_config(page_title="Make-a-wish – AI People Reader", layout="wide")


# ----------------------------------------------------------
# Helper functions
# ----------------------------------------------------------

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_job_id() -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    rand = os.urandom(3).hex()
    return f"{ts}__{rand}"


def upload_bytes_to_s3(data: bytes, key: str, content_type: str) -> None:
    s3.put_object(
        Bucket=AWS_BUCKET,
        Key=key,
        Body=data,
        ContentType=content_type,
    )


def s3_get_json_if_exists(key: str) -> Optional[Dict[str, Any]]:
    try:
        obj = s3.get_object(Bucket=AWS_BUCKET, Key=key)
    except ClientError as ce:
        if ce.response.get("Error", {}).get("Code") == "NoSuchKey":
            return None
        raise
    data = obj["Body"].read()
    return json.loads(data.decode("utf-8"))


def find_job_by_id(job_id: str) -> Optional[Dict[str, Any]]:
    """
    ลองหา job ตาม id จากทั้ง 4 prefix และบอกด้วยว่าตอนนี้อยู่สถานะอะไร
    """
    candidates = [
        (JOBS_PENDING_PREFIX, "pending"),
        (JOBS_PROCESSING_PREFIX, "processing"),
        (JOBS_FINISHED_PREFIX, "finished"),
        (JOBS_FAILED_PREFIX, "failed"),
    ]
    for prefix, status in candidates:
        key = f"{prefix}{job_id}.json"
        job = s3_get_json_if_exists(key)
        if job is not None:
            job["status"] = status  # ให้ status ตรงกับ prefix
            return job
    return None


def create_job(file_bytes: bytes, original_name: str, user_note: str) -> Dict[str, Any]:
    """
    สร้าง job ใหม่สำหรับ make-a-wish
    - mode ถูก fix เป็น "dots" (ลูกค้าได้แค่ Johansson dots)
    """
    job_id = new_job_id()
    mode = "dots"

    # เก็บไฟล์ input ไว้ที่ jobs/pending/<job_id>/input/input.mp4
    input_key = f"{JOBS_PENDING_PREFIX}{job_id}/input/input.mp4"
    output_key = f"{JOBS_OUTPUT_PREFIX}{job_id}/result.mp4"

    upload_bytes_to_s3(file_bytes, input_key, content_type="video/mp4")

    job = {
        "job_id": job_id,
        "status": "pending",
        "mode": mode,
        "input_key": input_key,
        "output_key": output_key,
        "created_at_utc": utc_now_iso(),
        "updated_at_utc": utc_now_iso(),
        "error": None,
        "user_note": user_note or "",
        "original_filename": original_name,
    }

    job_json_key = f"{JOBS_PENDING_PREFIX}{job_id}.json"
    upload_bytes_to_s3(
        json.dumps(job, ensure_ascii=False).encode("utf-8"),
        job_json_key,
        content_type="application/json",
    )

    return job


def download_output_video(job: Dict[str, Any]) -> bytes:
    output_key = job.get("output_key")
    if not output_key:
        raise ValueError("Job does not contain 'output_key'")
    obj = s3.get_object(Bucket=AWS_BUCKET, Key=output_key)
    return obj["Body"].read()


def build_download_filename(job: Dict[str, Any]) -> str:
    """
    สร้างชื่อไฟล์สวย ๆ เช่น CandidateA_dots.mp4
    ถ้า user_note ว่าง ใช้ job_id แทน
    """
    note = (job.get("user_note") or "").strip()
    base = note if note else job.get("job_id", "result")
    # ล้างอักษรแปลก ๆ ออก
    safe = "".join(ch for ch in base if ch.isalnum() or ch in (" ", "_", "-")).strip()
    if not safe:
        safe = job.get("job_id", "result")
    return f"{safe.replace(' ', '_')}_dots.mp4"


# ----------------------------------------------------------
# UI
# ----------------------------------------------------------

st.title("✨ Make-a-wish – AI People Reader")
st.markdown(
    """
ระบบช่วยวิเคราะห์ **ทักษะการนำเสนอและการสื่อสาร** จากวิดีโอสัมภาษณ์  
เบื้องหลังใช้ AI People Reader (Johansson dots) ทำงานแบบ background worker บน S3 + Render
"""
)

st.markdown("---")

# ==========================================================
# ① Upload section
# ==========================================================
st.header("① Upload Video for Analysis")

col_upload_left, col_upload_right = st.columns([2, 1])

with col_upload_left:
    uploaded_file = st.file_uploader(
        "Upload interview video file (สูงสุด ~1GB ต่อไฟล์ – mp4/mov/m4v)",
        type=["mp4", "mov", "m4v", "avi", "mkv"],
        accept_multiple_files=False,
    )

    user_note = st.text_input(
        "Optional note (สำหรับคุณครูหรือผู้ประเมิน เช่น ชื่อ Candidate)",
        "",
        placeholder="เช่น Candidate A – Final Interview – Leadership Focus",
    )

    if st.button("🚀 Submit for AI analysis"):
        if not uploaded_file:
            st.warning("กรุณาอัปโหลดวิดีโอก่อน")
        else:
            bytes_data = uploaded_file.read()
            job = create_job(bytes_data, uploaded_file.name, user_note)
            st.success("สร้างงานประมวลผลเรียบร้อยแล้ว 🎉")
            st.write("**Job ID:**", job["job_id"])
            st.caption(
                "กรุณาจดจำ Job ID นี้ไว้ เพื่อนำไปตรวจสอบสถานะและดาวน์โหลดวิดีโอภายหลัง"
            )

with col_upload_right:
    st.subheader("Tips")
    st.markdown(
        """
- วิดีโอควรยาวพอดี ไม่สั้นหรือยาวเกินไป
- ให้เห็นหน้าผู้พูดชัดเจน และท่าทางเต็มตัวเท่าที่ทำได้
- บันทึก Job ID ไว้เสมอเพื่อกลับมาตรวจผล
"""
    )

st.markdown("---")

# ==========================================================
# ② Check job status & download
# ==========================================================
st.header("② Check Job Status & View Report")

job_id_input = st.text_input(
    "Enter Job ID",
    "",
    placeholder="เช่น 20260117_010307__3dfd6",
)


if st.button("🔎 Check status"):
    if not job_id_input.strip():
        st.warning("กรุณาใส่ Job ID ก่อน")
    else:
        job = find_job_by_id(job_id_input.strip())
        if not job:
            st.error("ไม่พบ Job ID นี้ในระบบ")
        else:
            status = job.get("status", "unknown")
            if status == "pending":
                st.info("งานของคุณยังอยู่ในคิว (pending) กรุณารอสักครู่แล้วลองใหม่")
            elif status == "processing":
                st.warning("ระบบกำลังประมวลผลวิดีโอของคุณ (processing)…")
            elif status == "failed":
                st.error("งานนี้ประมวลผลไม่สำเร็จ (failed)")
                if job.get("error"):
                    with st.expander("ดูรายละเอียด error จาก worker"):
                        st.text(job.get("error"))
                        if job.get("traceback"):
                            st.text(job.get("traceback"))
            elif status == "finished":
                st.success("งานประมวลผลเสร็จสิ้นแล้ว 🎉")
            else:
                st.write(f"สถานะปัจจุบัน: {status}")

            # แสดง JSON แบบ raw สำหรับคุณครู / developer
            with st.expander("📦 ดู JSON ทั้งหมดที่ได้จาก worker"):
                st.json(job)

            # ถ้าเสร็จแล้วและมี output_key → ให้โหลดได้เลย
            if status == "finished" and job.get("output_key"):
                try:
                    video_bytes = download_output_video(job)
                except ClientError as ce:
                    st.error(f"ไม่สามารถดาวน์โหลดวิดีโอจาก S3 ได้: {ce}")
                else:
                    dl_name = build_download_filename(job)
                    st.download_button(
                        label="⬇️ Download processed video (result.mp4)",
                        data=video_bytes,
                        file_name=dl_name,
                        mime="video/mp4",
                    )
