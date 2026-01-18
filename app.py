# app.py  --- AI People Reader - Presentation Analysis Job Manager

import os
import io
import json
import uuid
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

import boto3
import pandas as pd
import streamlit as st

# ----------------------------------------------------------
# Config
# ----------------------------------------------------------
AWS_BUCKET = os.environ.get("AWS_BUCKET") or os.environ.get("S3_BUCKET")
AWS_REGION = os.environ.get("AWS_REGION", "ap-southeast-1")

if not AWS_BUCKET:
    raise RuntimeError("Missing AWS_BUCKET (or S3_BUCKET) environment variable")

s3 = boto3.client("s3", region_name=AWS_REGION)

JOBS_PREFIX = "jobs"
PENDING_PREFIX = f"{JOBS_PREFIX}/pending"
PROCESSING_PREFIX = f"{JOBS_PREFIX}/processing"
FINISHED_PREFIX = f"{JOBS_PREFIX}/finished"
FAILED_PREFIX = f"{JOBS_PREFIX}/failed"
OUTPUT_PREFIX = f"{JOBS_PREFIX}/output"

INPUT_PREFIX = f"{JOBS_PREFIX}/input"

st.set_page_config(
    page_title="AI People Reader - Presentation Analysis Job Manager",
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
    s3.upload_fileobj(io.BytesIO(data), AWS_BUCKET, key, ExtraArgs={"ContentType": content_type})


def s3_get_json(key: str) -> Dict[str, Any]:
    obj = s3.get_object(Bucket=AWS_BUCKET, Key=key)
    raw = obj["Body"].read()
    return json.loads(raw.decode("utf-8"))


def s3_put_json(key: str, payload: Dict[str, Any]) -> None:
    body_str = json.dumps(payload, ensure_ascii=False)
    s3.put_object(
        Bucket=AWS_BUCKET,
        Key=key,
        Body=body_str.encode("utf-8"),
        ContentType="application/json",
    )


def list_json_under_prefix(prefix: str) -> List[str]:
    keys: List[str] = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=AWS_BUCKET, Prefix=prefix):
        for item in page.get("Contents", []):
            key = item["Key"]
            if key.endswith(".json"):
                keys.append(key)
    return keys


def collect_jobs() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    def add_from_prefix(prefix: str, status_label: str) -> None:
        for key in list_json_under_prefix(prefix):
            try:
                job = s3_get_json(key)
            except Exception:
                continue

            job_id = job.get("job_id", "")
            mode = job.get("mode", "")
            created_at = job.get("created_at", "")
            updated_at = job.get("updated_at", "")

            rows.append(
                {
                    "job_id": job_id,
                    "status": job.get("status", status_label),
                    "mode": mode,
                    "created_at": created_at,
                    "updated_at": updated_at,
                    "error": job.get("error", ""),
                    "json_key": key,
                    "output_key": job.get("output_key", ""),
                }
            )

    add_from_prefix(PENDING_PREFIX, "pending")
    add_from_prefix(PROCESSING_PREFIX, "processing")
    add_from_prefix(FINISHED_PREFIX, "finished")
    add_from_prefix(FAILED_PREFIX, "failed")

    # sort latest first
    def parse_dt(s: str) -> str:
        return s or ""

    rows.sort(key=lambda r: parse_dt(r.get("created_at", "")), reverse=True)
    return rows


def load_finished_jobs() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for key in list_json_under_prefix(FINISHED_PREFIX):
        try:
            job = s3_get_json(key)
        except Exception:
            continue

        job_id = job.get("job_id", "")
        rows.append(
            {
                "job_id": job_id,
                "json_key": key,
                "output_key": job.get("output_key", ""),
                "mode": job.get("mode", ""),
            }
        )
    return rows


def get_output_bytes(output_key: str) -> Optional[bytes]:
    try:
        obj = s3.get_object(Bucket=AWS_BUCKET, Key=output_key)
        return obj["Body"].read()
    except Exception:
        return None


# ----------------------------------------------------------
# UI layout
# ----------------------------------------------------------

st.title("AI People Reader - Presentation Analysis Job Manager")

col_left, col_right = st.columns([1.1, 1.3])

# ------------------------------ Create New Job -----------------------------
with col_left:
    st.subheader("Create New Job")

    mode = st.selectbox(
        "Mode",
        options=["dots"],
        index=0,
        help="ตอนนี้ใช้เฉพาะ Johansson dots mode (ภายหลังเพิ่ม mode อื่นได้)",
    )

    uploaded_file = st.file_uploader(
        "Upload video file",
        type=["mp4", "mov", "m4v", "avi", "mpeg4"],
        help="Limit ~1GB per file • MP4, MOV, M4V, AVI, MPEG4",
    )

    create_col1, create_col2 = st.columns([1, 1])

    with create_col1:
        create_clicked = st.button("Create job", type="primary")

    if create_clicked:
        if uploaded_file is None:
            st.error("กรุณาเลือกไฟล์วิดีโอก่อน")
        else:
            # อ่าน bytes จาก uploader
            data = uploaded_file.read()
            if not data:
                st.error("ไม่สามารถอ่านไฟล์วิดีโอได้")
            else:
                job_id = new_job_id()

                ext = os.path.splitext(uploaded_file.name)[1] or ".mp4"
                video_key = f"{INPUT_PREFIX}/{job_id}{ext}"

                # upload วิดีโอเข้า S3
                upload_bytes_to_s3(
                    data,
                    video_key,
                    content_type="video/mp4",
                )

                # สร้าง job JSON
                created = utc_now_iso()
                job_json = {
                    "job_id": job_id,
                    "mode": mode,
                    "status": "pending",
                    "created_at": created,
                    "updated_at": created,
                    "input_key": video_key,
                    # output_key จะให้ worker ใส่ตอนจบก็ได้
                }

                pending_key = f"{PENDING_PREFIX}/{job_id}.json"
                s3_put_json(pending_key, job_json)

                st.success(f"สร้างงานใหม่เรียบร้อยแล้ว (job_id={job_id})")
                # เคลียร์ไฟล์ใน uploader (trick: ใช้ session_state)
                st.session_state["last_created_job_id"] = job_id

# ------------------------------ Jobs table ---------------------------------
with col_right:
    st.subheader("Jobs")

    refresh_clicked = st.button("Refresh job list")

    # แค่กดปุ่ม Streamlit ก็ rerun script ให้อยู่แล้ว ไม่ต้องเรียก st.rerun()
    jobs = collect_jobs()
    if jobs:
        df = pd.DataFrame(
            [
                {
                    "job_id": j["job_id"],
                    "status": j["status"],
                    "mode": j["mode"],
                    "created_at": j["created_at"],
                    "updated_at": j["updated_at"],
                    "error": j["error"],
                }
                for j in jobs
            ]
        )
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("ยังไม่พบงานใด ๆ ในระบบ")

# --------------------------- Download result video -------------------------
st.markdown("---")
st.subheader("Download result video ⬇️")

finished_jobs = load_finished_jobs()
if not finished_jobs:
    st.info("No finished jobs yet.")
else:
    job_labels = [f"{j['job_id']}  ({j['mode']})" for j in finished_jobs]
    selected_index = st.selectbox(
        "Select job (will download if result.mp4 exists)",
        options=list(range(len(finished_jobs))),
        format_func=lambda i: job_labels[i],
    )

    selected_job = finished_jobs[selected_index]
    output_key = selected_job.get("output_key") or f"{OUTPUT_PREFIX}/{selected_job['job_id']}/result.mp4"

    if st.button("Prepare download"):
        data = get_output_bytes(output_key)
        if data is None:
            st.error("ไม่พบไฟล์วิดีโอผลลัพธ์ใน S3 (output_key = %s)" % output_key)
        else:
            st.success("พร้อมดาวน์โหลดแล้ว 👇")
            st.download_button(
                "Download result video",
                data=data,
                file_name=f"{selected_job['job_id']}.mp4",
                mime="video/mp4",
            )
