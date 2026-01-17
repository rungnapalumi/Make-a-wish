# app.py  --- Streamlit frontend สำหรับ repo Make-a-wish

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict

import boto3
import streamlit as st
from botocore.exceptions import ClientError

# ---------------------------------------------------------------------
# Config S3
# ---------------------------------------------------------------------

AWS_BUCKET = os.getenv("AWS_BUCKET") or os.getenv("S3_BUCKET")
AWS_REGION = os.getenv("AWS_REGION", "ap-southeast-1")

if not AWS_BUCKET:
    raise RuntimeError("Missing AWS_BUCKET (or S3_BUCKET) environment variable")

s3 = boto3.client("s3", region_name=AWS_REGION)

JOBS_PENDING_PREFIX = "jobs/pending/"
JOBS_OUTPUT_PREFIX = "jobs/output/"

st.set_page_config(
    page_title="Make-a-wish – AI People Reader",
    layout="wide",
)


# ---------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f%z")


def new_job_id() -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    rand = os.urandom(3).hex()
    return f"{ts}__{rand}"


def s3_put_json(key: str, payload: Dict[str, Any]) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    s3.put_object(
        Bucket=AWS_BUCKET,
        Key=key,
        Body=body,
        ContentType="application/json",
    )


def s3_get_json(key: str) -> Dict[str, Any]:
    resp = s3.get_object(Bucket=AWS_BUCKET, Key=key)
    data = resp["Body"].read()
    return json.loads(data.decode("utf-8"))


# ---------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------

st.title("✨ Make-a-wish – AI People Reader")

st.markdown(
    """
1. อัปโหลดวิดีโอสัมภาษณ์ของคุณ  
2. ระบบจะสร้าง Job ID และส่งไฟล์ไปให้ AI ประมวลผล  
3. ใช้ Job ID เพื่อตรวจสอบสถานะ และดาวน์โหลดวิดีโอ + รายงานเมื่อเสร็จแล้ว  
"""
)

tab_upload, tab_status = st.tabs(["① Upload video for analysis", "② Check Job Status & Download"])

# =========================================================
# TAB 1: Upload
# =========================================================
with tab_upload:
    st.header("① Upload Video for Analysis")

    uploaded_file = st.file_uploader(
        "Upload interview video file (สูงสุด 1GB ต่อไฟล์)",
        type=["mp4", "mov", "m4v", "avi", "mkv"],
        accept_multiple_files=False,
    )

    user_note = st.text_area("Optional note (สำหรับคุณครู/ผู้ประเมิน)", "")

    if st.button("🚀 Submit for AI analysis"):
        if not uploaded_file:
            st.warning("กรุณาอัปโหลดวิดีโอก่อนค่ะ")
        else:
            # อ่านไฟล์เป็น bytes
            file_bytes = uploaded_file.read()
            if not file_bytes:
                st.error("วิดีโอว่างเปล่าหรืออ่านไฟล์ไม่สำเร็จ")
            else:
                job_id = new_job_id()
                original_filename = uploaded_file.name

                input_key = f"{JOBS_PENDING_PREFIX}{job_id}/input/input.mp4"
                output_key = f"{JOBS_OUTPUT_PREFIX}{job_id}/result.mp4"
                job_json_key = f"{JOBS_PENDING_PREFIX}{job_id}.json"

                # 1) อัปโหลดวิดีโอไป S3
                s3.put_object(
                    Bucket=AWS_BUCKET,
                    Key=input_key,
                    Body=file_bytes,
                    ContentType="video/mp4",
                )

                # 2) สร้าง job JSON ให้ worker
                now = utc_now_iso()
                job = {
                    "job_id": job_id,
                    "status": "pending",
                    "mode": "dots",  # ตอนนี้ใช้โหมด dots + report
                    "input_key": input_key,
                    "output_key": output_key,
                    "created_at_utc": now,
                    "updated_at_utc": now,
                    "error": None,
                    "user_note": user_note,
                    "original_filename": original_filename,
                }
                s3_put_json(job_json_key, job)

                st.success("สร้างงานเรียบร้อย 🎉")
                st.write("**Your Job ID:**")
                st.code(job_id, language="text")

                with st.expander("ดูรายละเอียด job JSON ที่สร้าง"):
                    st.json(job)


# =========================================================
# TAB 2: Check status + download
# =========================================================
with tab_status:
    st.header("② Check Job Status & View Report")

    job_id_input = st.text_input(
        "Enter Job ID",
        value="",
        placeholder="เช่น 20260117_044336__b051e1",
    )

    if st.button("🔍 Check status"):
        if not job_id_input.strip():
            st.warning("กรุณากรอก Job ID ก่อนค่ะ")
        else:
            job_json_key = f"{JOBS_PENDING_PREFIX}{job_id_input.strip()}.json"

            try:
                job = s3_get_json(job_json_key)
            except ClientError as ce:
                code = ce.response.get("Error", {}).get("Code")
                if code == "NoSuchKey":
                    st.error("ไม่พบ Job ID นี้ในระบบ กรุณาตรวจสอบอีกครั้งค่ะ")
                else:
                    st.error(f"เกิดข้อผิดพลาดขณะอ่านข้อมูลจาก S3: {ce}")
            else:
                status = job.get("status", "unknown")
                st.success(f"งานนี้ประมวลผลสถานะ: **{status}** 🎉" if status == "finished"
                           else f"Current status: **{status}**")

                with st.expander("📦 JSON ทั้งหมดที่ได้จาก worker"):
                    st.json(job)

                # ถ้าทำเสร็จแล้ว ลองดึงผลลัพธ์
                if status == "finished":
                    # ----- Download video -----
                    output_key = job.get("output_key")
                    if output_key:
                        try:
                            resp = s3.get_object(Bucket=AWS_BUCKET, Key=output_key)
                            video_bytes = resp["Body"].read()

                            file_name = f"{job_id_input}_result.mp4"
                            st.download_button(
                                "📥 Download processed video (result.mp4)",
                                data=video_bytes,
                                file_name=file_name,
                                mime="video/mp4",
                            )
                        except ClientError as ce:
                            st.error(f"ไม่สามารถโหลดวิดีโอจาก S3 ได้: {ce}")

                    # ----- Download report (ถ้ามี) -----
                    report_key = job.get("report_s3_key")
                    if report_key:
                        try:
                            resp = s3.get_object(Bucket=AWS_BUCKET, Key=report_key)
                            report_bytes = resp["Body"].read()

                            report_name = f"{job_id_input}_presentation_report_TH_EN.docx"
                            st.download_button(
                                "📄 Download Presentation Skill Report (TH/EN, .docx)",
                                data=report_bytes,
                                file_name=report_name,
                                mime=(
                                    "application/vnd.openxmlformats-officedocument."
                                    "wordprocessingml.document"
                                ),
                            )
                        except ClientError as ce:
                            st.error(f"ไม่สามารถโหลดไฟล์รายงานจาก S3 ได้: {ce}")
                    else:
                        st.info("ยังไม่มี report_s3_key ใน JSON (worker ยังไม่เขียน field นี้กลับมา)")
                else:
                    st.info("ถ้าสถานะยังไม่เป็น finished ให้รอสักครู่แล้วกด Check status ซ้ำอีกครั้งค่ะ")
