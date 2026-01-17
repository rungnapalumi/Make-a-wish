# app.py — AI People Reader "Make a Wish" Webapp
# ----------------------------------------------
# ใช้ schema เดียวกับ App-maker-App-maker:
# - create_job(): สร้าง job และเซฟที่ jobs/pending/<job_id>.json
# - worker จะอ่านไฟล์นี้ไปประมวลผล แล้วเขียนผลลัพธ์ที่ jobs/finished/<job_id>.json
# - output video อยู่ที่ jobs/output/<job_id>/result.mp4

import os
import json
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import streamlit as st
import boto3
from botocore.exceptions import ClientError

# ----------------------------------------------------------
# AWS Config
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
# Streamlit config
# ----------------------------------------------------------

st.set_page_config(page_title="AI People Reader - Make a Wish", layout="wide")
st.title("🎬 AI People Reader – Make a Wish")

st.markdown(
    """
1. อัปโหลดวิดีโอสัมภาษณ์ของคุณ  
2. ระบบจะสร้าง Job ID และส่งไฟล์ไปให้ AI ประมวลผล  
3. ใช้ Job ID เพื่อตรวจสอบสถานะ และดาวน์โหลดวิดีโอที่ประมวลผลแล้ว  

---
"""
)

if "job_id" not in st.session_state:
    st.session_state["job_id"] = ""


# ----------------------------------------------------------
# Helper functions (ใช้ schema เดียวกับ App-maker)
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


def s3_key_exists(key: str) -> bool:
    try:
        s3.head_object(Bucket=AWS_BUCKET, Key=key)
        return True
    except ClientError as e:
        if e.response["Error"]["Code"] in ("404", "NoSuchKey", "NotFound"):
            return False
        raise


def s3_get_json(key: str) -> Dict[str, Any]:
    obj = s3.get_object(Bucket=AWS_BUCKET, Key=key)
    return json.loads(obj["Body"].read().decode("utf-8"))


def generate_presigned_url_from_key(key: str, expires_in: int = 3600) -> Optional[str]:
    try:
        url = s3.generate_presigned_url(
            "get_object",
            Params={"Bucket": AWS_BUCKET, "Key": key},
            ExpiresIn=expires_in,
        )
        return url
    except Exception:
        return None


def create_job(file_bytes: bytes, note: str, mode: str = "dots") -> Dict[str, Any]:
    """
    ใช้ logic เดียวกับ App-maker-App-maker:
      - input video:  jobs/pending/<job_id>/input/input.mp4
      - output video: jobs/output/<job_id>/result.mp4
      - job JSON:     jobs/pending/<job_id>.json
    """
    job_id = new_job_id()

    input_key = f"{JOBS_PENDING_PREFIX}{job_id}/input/input.mp4"
    output_key = f"{JOBS_OUTPUT_PREFIX}{job_id}/result.mp4"

    # Upload video
    upload_bytes_to_s3(
        file_bytes,
        input_key,
        content_type="video/mp4",
    )

    now = utc_now_iso()
    job: Dict[str, Any] = {
        "job_id": job_id,
        "status": "pending",
        "mode": mode,           # ⭐ สำคัญ - บอก worker ให้ทำ "dots"
        "input_key": input_key,
        "output_key": output_key,
        "created_at": now,
        "updated_at": now,
        "error": None,
        "user_note": note,
    }

    job_json_key = f"{JOBS_PENDING_PREFIX}{job_id}.json"
    s3_put_json(job_json_key, job)

    return job


# ----------------------------------------------------------
# Section 1: Upload & create job
# ----------------------------------------------------------

st.header("① Upload Video for Analysis")

col_left, col_right = st.columns([2, 1])

with col_left:
    uploaded_file = st.file_uploader(
        "Upload interview video file",
        type=["mp4", "mov", "m4v"],
        help="Limit ~1GB per file • เห็นตัวผู้พูดตั้งแต่ครึ่งตัวขึ้นไปจะดีที่สุด",
    )

    note = st.text_area(
        "Optional note (สำหรับคุณครู/ผู้ประเมิน)",
        placeholder="เช่น Candidate A – Final Interview – Leadership Focus",
    )

    submit_btn = st.button("🚀 Submit for AI analysis")

with col_right:
    st.markdown("#### Tips")
    st.markdown(
        """
- วิดีโอควรมีความยาวพอดี ไม่ยาวเกินไป  
- ผู้พูดควรเห็นหน้าและท่าทางอย่างชัดเจน  
- บันทึก Job ID ไว้เพื่อตรวจสอบผลภายหลัง  
"""
    )

if submit_btn:
    if not uploaded_file:
        st.warning("กรุณาอัปโหลดวิดีโอก่อนค่ะ")
    else:
        # hard limit ฝั่งเว็บ (1GB)
        max_bytes = 1024 * 1024 * 1024
        if uploaded_file.size > max_bytes:
            st.error("⚠️ วิดีโอควรมีขนาดไม่เกิน 1GB")
            st.stop()

        file_bytes = uploaded_file.read()

        # ใช้ mode="dots" เพื่อให้ worker ทำวิดีโอ dot เหมือนใน App-maker
        job = create_job(file_bytes, note=note, mode="dots")

        st.session_state["job_id"] = job["job_id"]
        st.success(f"สร้างงานสำเร็จ! 🎉 Job ID ของคุณคือ: `{job['job_id']}`")
        st.info("กรุณาคัดลอก Job ID ไว้ เพื่อใช้ในขั้นตอนถัดไป")

st.markdown("---")

# ----------------------------------------------------------
# Section 2: Check job status & download
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
1. วาง Job ID ที่คุณได้หลังอัปโหลดวิดีโอ  
2. กดปุ่ม **Check status**  
3. ถ้าประมวลผลเสร็จแล้ว ระบบจะแสดงสรุปและลิงก์สำหรับดาวน์โหลดวิดีโอ  
"""
    )

if check_btn:
    if not job_id_input.strip():
        st.warning("กรุณากรอก Job ID ก่อนค่ะ")
    else:
        job_id = job_id_input.strip()

        finished_key = f"{JOBS_FINISHED_PREFIX}{job_id}.json"
        pending_key = f"{JOBS_PENDING_PREFIX}{job_id}.json"
        failed_key = f"{JOBS_FAILED_PREFIX}{job_id}.json"

        try:
            if s3_key_exists(failed_key):
                st.error("❌ งานนี้ประมวลผลไม่สำเร็จ (อยู่ใน failed).")
                failed_info = s3_get_json(failed_key)
                with st.expander("ดูรายละเอียด error (จาก worker)"):
                    st.json(failed_info)

            elif s3_key_exists(finished_key):
                result = s3_get_json(finished_key)
                st.success("✅ งานนี้ประมวลผลเสร็จสิ้นแล้ว! 🎉")

                st.subheader("Result Summary")

                with st.expander("ดู JSON ทั้งหมดที่ได้จาก worker"):
                    st.json(result)

                # ดึง URL ของวิดีโอที่ได้
                report_url: Optional[str] = None

                # worker เขียน output_key ไว้ใน JSON (ตาม create_job)
                if "output_key" in result:
                    report_url = generate_presigned_url_from_key(
                        result["output_key"], expires_in=3600
                    )

                # เผื่ออนาคต worker ส่งอย่างอื่นมา
                elif "report_s3_key" in result:
                    report_url = generate_presigned_url_from_key(
                        result["report_s3_key"], expires_in=3600
                    )
                elif "report_url" in result:
                    report_url = result["report_url"]

                if report_url:
                    st.markdown(f"[📄 Download processed video / report]({report_url})")
                else:
                    st.info(
                        "ไม่มี URL ของ report ในผลลัพธ์ กรุณาตรวจสอบ JSON จาก worker "
                        "ว่ามี field `output_key`, `report_s3_key` หรือ `report_url` หรือไม่"
                    )

            elif s3_key_exists(pending_key):
                st.info("🕒 งานนี้กำลังรอหรือกำลังประมวลผล กรุณาลองใหม่ภายหลัง")
            else:
                st.warning(
                    "ไม่พบ Job นี้ในระบบ (ไม่มีใน pending / finished / failed). "
                    "กรุณาตรวจสอบว่า Job ID ถูกต้องหรือไม่"
                )

        except ClientError as e:
            st.error(f"เกิดข้อผิดพลาดจากฝั่ง S3: {e}")
        except Exception as e:
            st.error(f"เกิดข้อผิดพลาดไม่คาดคิด: {e}")
