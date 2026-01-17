# app.py — AI People Reader (Make-a-wish - Dots Skeleton Report)
#
# หน้าที่หลัก:
#   1) ให้ผู้ใช้ upload วิดีโอสัมภาษณ์ + ใส่ note (สำหรับคุณครู/ผู้ประเมิน)
#   2) สร้าง job JSON (mode="dots") ตาม schema เดียวกับ worker.py
#   3) เซฟ input video + job JSON ลง S3
#   4) ให้ผู้ใช้กรอก Job ID เพื่อตรวจสถานะ และดาวน์โหลดวิดีโอที่ประมวลผลแล้ว

import os
import json
import uuid
import re
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import boto3
import streamlit as st
from botocore.exceptions import ClientError

# ----------------------------------------------------------
# Config
# ----------------------------------------------------------

AWS_BUCKET = os.environ.get("AWS_BUCKET") or os.environ.get("S3_BUCKET")
AWS_REGION = os.environ.get("AWS_REGION", "ap-southeast-1")

if not AWS_BUCKET:
    raise RuntimeError("Missing AWS_BUCKET (or S3_BUCKET) environment variable")

s3 = boto3.client("s3", region_name=AWS_REGION)

JOBS_PENDING_PREFIX = "jobs/pending/"
JOBS_PROCESSING_PREFIX = "jobs/processing/"
JOBS_FINISHED_PREFIX = "jobs/finished/"
JOBS_FAILED_PREFIX = "jobs/failed/"
JOBS_OUTPUT_PREFIX = "jobs/output/"

# ----------------------------------------------------------
# Streamlit page
# ----------------------------------------------------------

st.set_page_config(
    page_title="AI People Reader – Interview Analyzer",
    layout="wide",
)

# ----------------------------------------------------------
# Helper functions
# ----------------------------------------------------------


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_job_id() -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    rand = uuid.uuid4().hex[:5]
    return f"{ts}__{rand}"


def upload_bytes_to_s3(data: bytes, key: str, content_type: str) -> None:
    s3.put_object(
        Bucket=AWS_BUCKET,
        Key=key,
        Body=data,
        ContentType=content_type,
    )


def s3_put_json(key: str, payload: Dict[str, Any]) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    s3.put_object(
        Bucket=AWS_BUCKET,
        Key=key,
        Body=body,
        ContentType="application/json",
    )


def s3_get_json(key: str) -> Dict[str, Any]:
    obj = s3.get_object(Bucket=AWS_BUCKET, Key=key)
    data = obj["Body"].read()
    return json.loads(data.decode("utf-8"))


def s3_key_exists(key: str) -> bool:
    try:
        s3.head_object(Bucket=AWS_BUCKET, Key=key)
        return True
    except ClientError as ce:
        if ce.response.get("Error", {}).get("Code") == "404":
            return False
        raise


def generate_presigned_url_from_key(key: str, expires_in: int = 3600) -> Optional[str]:
    try:
        url = s3.generate_presigned_url(
            "get_object",
            Params={"Bucket": AWS_BUCKET, "Key": key},
            ExpiresIn=expires_in,
        )
        return url
    except ClientError:
        return None


def get_s3_bytes(key: str) -> bytes:
    obj = s3.get_object(Bucket=AWS_BUCKET, Key=key)
    return obj["Body"].read()


def slugify_filename(text: str) -> str:
    """
    แปลง note (เช่น 'Candidate A - Final Interview')
    ให้เป็นชื่อไฟล์ปลอดภัย เช่น 'Candidate_A_Final_Interview'
    """
    text = text.strip()
    if not text:
        return ""
    # แทนทุกอย่างที่ไม่ใช่ตัวอักษร/ตัวเลขด้วย _
    text = re.sub(r"[^A-Za-z0-9ก-๙]+", "_", text)
    # ตัด _ ซ้ำ ๆ ทิ้ง
    text = re.sub(r"_+", "_", text)
    return text.strip("_")


# ----------------------------------------------------------
# Job creation
# ----------------------------------------------------------


def create_job(file_bytes: bytes, filename: str, user_note: str) -> Dict[str, Any]:
    """
    สร้าง job ใหม่:
      - เซฟวิดีโอที่ S3: jobs/pending/<job_id>/input/<original_filename>
      - สร้าง JSON ที่ jobs/pending/<job_id>.json
      - mode ถูก fix เป็น "dots"
    """
    job_id = new_job_id()

    # เก็บชื่อไฟล์ต้นฉบับ (เพื่อ debug / audit)
    safe_filename = filename or "input.mp4"

    input_key = f"{JOBS_PENDING_PREFIX}{job_id}/input/{safe_filename}"
    output_key = f"{JOBS_OUTPUT_PREFIX}{job_id}/result.mp4"

    # Upload video (ให้ ContentType เป็น video/mp4 แบบกว้าง ๆ)
    upload_bytes_to_s3(file_bytes, input_key, content_type="video/mp4")

    now = utc_now_iso()
    job: Dict[str, Any] = {
        "job_id": job_id,
        "status": "pending",
        "mode": "dots",  # ⭐ บอก worker ว่าต้อง run โหมด dots
        "input_key": input_key,
        "output_key": output_key,
        "created_at": now,
        "updated_at": now,
        "error": None,
        "user_note": user_note or "",
        "original_filename": safe_filename,
    }

    job_json_key = f"{JOBS_PENDING_PREFIX}{job_id}.json"
    s3_put_json(job_json_key, job)

    return job


# ----------------------------------------------------------
# UI – Layout
# ----------------------------------------------------------

st.markdown(
    """
## ① Upload Video for Analysis

1. อัปโหลดวิดีโอสัมภาษณ์ของคุณ  
2. ระบบจะสร้าง Job ID และส่งไฟล์ไปให้ AI ประมวลผล  
3. ใช้ Job ID เพื่อตรวจสอบสถานะ และดาวน์โหลดวิดีโอที่ประมวลผลแล้ว  
"""
)

col_upload, col_tips = st.columns([2.2, 1.1])

with col_upload:
    uploaded_file = st.file_uploader(
        "Upload interview video file",
        type=["mp4", "mov", "m4v", "mpeg4"],
        accept_multiple_files=False,
        help="Limit ประมาณ 1GB ต่อไฟล์ (แนะนำให้ใช้ 720p หรือไฟล์ไม่ใหญ่มากเพื่อความเร็วในการอัปโหลด)",
    )

    note = st.text_area(
        "Optional note (สำหรับคุณครู/ผู้ประเมิน)",
        placeholder="เช่น Candidate A – Final Interview – Leadership Focus",
    )

    submit_clicked = st.button("🚀 Submit for AI analysis")

    if submit_clicked:
        if not uploaded_file:
            st.error("กรุณาอัปโหลดวิดีโอก่อนค่ะ")
        else:
            file_bytes = uploaded_file.read()
            filename = uploaded_file.name

            # (เลือกได้ว่าจะตรวจขนาดไฟล์เพิ่มไหม ถ้าอยาก limit)
            size_mb = len(file_bytes) / (1024 * 1024)
            if size_mb <= 0:
                st.error("ไฟล์วิดีโอไม่ถูกต้อง")
            else:
                with st.spinner("กำลังอัปโหลดวิดีโอและสร้างงานใหม่..."):
                    job = create_job(file_bytes, filename, note)

                st.success("สร้างงานใหม่สำเร็จแล้ว! 🎉")
                st.write("**Job ID:**")
                st.code(job["job_id"], language="text")

                with st.expander("ดูรายละเอียดงาน (JSON ที่ส่งให้ worker)"):
                    st.json(job)

with col_tips:
    st.subheader("Tips")
    st.markdown(
        """
- วิดีโอควรมีความยาวพอดี ไม่ยาวเกินไป  
- ผู้พูดควรเห็นหน้าและท่าทางอย่างชัดเจน  
- บันทึก Job ID ไว้ เพื่อกลับมาตรวจผลภายหลัง  
"""
    )

st.markdown("---")

# ----------------------------------------------------------
# ② Check Job Status & Download Result
# ----------------------------------------------------------

st.markdown("## ② Check Job Status & View Report")

job_id_input = st.text_input(
    "Enter Job ID",
    placeholder="เช่น 20260117_010307__3dfd6",
)

check_clicked = st.button("🔍 Check status")

if check_clicked:
    if not job_id_input.strip():
        st.error("กรุณาใส่ Job ID ก่อนค่ะ")
    else:
        job_id = job_id_input.strip()

        pending_key = f"{JOBS_PENDING_PREFIX}{job_id}.json"
        processing_key = f"{JOBS_PROCESSING_PREFIX}{job_id}.json"
        finished_key = f"{JOBS_FINISHED_PREFIX}{job_id}.json"
        failed_key = f"{JOBS_FAILED_PREFIX}{job_id}.json"

        # ลำดับการเช็คสถานะ
        if s3_key_exists(failed_key):
            result = s3_get_json(failed_key)
            st.error("งานนี้ประมวลผลล้มเหลว 😥")
            with st.expander("ดู JSON ทั้งหมดที่ได้จาก worker"):
                st.json(result)

        elif s3_key_exists(finished_key):
            result = s3_get_json(finished_key)
            st.success("✅ งานนี้ประมวลผลเสร็จสิ้นแล้ว! 🎉")

            st.subheader("Result Summary")

            with st.expander("ดู JSON ทั้งหมดที่ได้จาก worker"):
                st.json(result)

            # -------------------------------
            # ปุ่มดาวน์โหลดไฟล์จาก S3 โดยตรง
            # -------------------------------
            output_key: Optional[str] = None

            if "output_key" in result and result["output_key"]:
                output_key = result["output_key"]
            elif "report_s3_key" in result and result["report_s3_key"]:
                output_key = result["report_s3_key"]

            # เตรียมชื่อไฟล์สวย ๆ สำหรับดาวน์โหลด
            user_note = result.get("user_note", "") or ""
            base_name = slugify_filename(user_note)
            if not base_name:
                base_name = job_id
            download_name = f"{base_name}_dots.mp4"

            if output_key:
                try:
                    video_bytes = get_s3_bytes(output_key)

                    st.download_button(
                        label="📥 Download processed video / report",
                        data=video_bytes,
                        file_name=download_name,
                        mime="video/mp4",
                    )

                except ClientError as ce:
                    st.warning(
                        f"ไม่สามารถโหลดไฟล์จาก S3 โดยตรงได้ ({ce}). "
                        "จะลองสร้างลิงก์ชั่วคราวสำหรับเปิดไฟล์แทน"
                    )
                    report_url = generate_presigned_url_from_key(output_key, expires_in=3600)
                    if report_url:
                        st.markdown(f"[เปิดไฟล์จาก S3]({report_url})")
                    else:
                        st.error("ไม่สามารถสร้างลิงก์ดาวน์โหลดจาก S3 ได้")
            elif "report_url" in result and result["report_url"]:
                # fallback กรณี worker สร้าง URL มาให้เอง
                st.markdown(
                    f"[📄 Download processed video / report]({result['report_url']})"
                )
            else:
                st.info(
                    "ไม่มีข้อมูลตำแหน่งไฟล์ในผลลัพธ์ "
                    "กรุณาตรวจสอบว่า worker เขียน `output_key` หรือ `report_s3_key` หรือไม่"
                )

        elif s3_key_exists(processing_key):
            st.info("⏳ งานนี้กำลังประมวลผลอยู่ (processing)...")
        elif s3_key_exists(pending_key):
            st.info("⌛ งานนี้ยังอยู่ในคิว (pending) รอ worker มาประมวลผล...")
        else:
            st.warning("ไม่พบงานนี้ในระบบ กรุณาตรวจสอบว่า Job ID ถูกต้องหรือไม่")
