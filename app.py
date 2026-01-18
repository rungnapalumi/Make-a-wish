# app.py  --- AI People Reader - Presentation Analysis Job Manager
import os
import io
import json
import uuid
from datetime import datetime, timezone
from typing import List, Dict, Any

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


# ----------------------------------------------------------
# Helper functions
# ----------------------------------------------------------
def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(tzinfo=timezone.utc).isoformat()


def new_job_id() -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    rand = uuid.uuid4().hex[:5]
    return f"{ts}__{rand}"


def upload_fileobj_to_s3(file_obj, key: str, content_type: str = "video/mp4") -> None:
    st.write("")  # keep Streamlit happy
    s3.upload_fileobj(
        Fileobj=file_obj,
        Bucket=AWS_BUCKET,
        Key=key,
        ExtraArgs={"ContentType": content_type},
    )


def put_job_json(job: Dict[str, Any], prefix: str) -> str:
    key = f"{prefix}/{job['job_id']}.json"
    body = json.dumps(job, ensure_ascii=False).encode("utf-8")
    s3.put_object(
        Bucket=AWS_BUCKET,
        Key=key,
        Body=body,
        ContentType="application/json",
    )
    return key


def list_jobs_all() -> List[Dict[str, Any]]:
    jobs: List[Dict[str, Any]] = []

    def load_prefix(prefix: str):
        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=AWS_BUCKET, Prefix=prefix):
            for item in page.get("Contents", []):
                key = item["Key"]
                if not key.endswith(".json"):
                    continue
                obj = s3.get_object(Bucket=AWS_BUCKET, Key=key)
                data = obj["Body"].read()
                try:
                    job = json.loads(data.decode("utf-8"))
                    jobs.append(job)
                except Exception:
                    continue

    load_prefix(PENDING_PREFIX)
    load_prefix(PROCESSING_PREFIX)
    load_prefix(FINISHED_PREFIX)
    load_prefix(FAILED_PREFIX)
    return jobs


def get_job_by_id(job_id: str) -> Dict[str, Any] | None:
    for prefix in (PENDING_PREFIX, PROCESSING_PREFIX, FINISHED_PREFIX, FAILED_PREFIX):
        key = f"{prefix}/{job_id}.json"
        try:
            obj = s3.get_object(Bucket=AWS_BUCKET, Key=key)
        except s3.exceptions.NoSuchKey:  # type: ignore[attr-defined]
            continue
        data = obj["Body"].read()
        return json.loads(data.decode("utf-8"))
    return None


def download_video_bytes(output_key: str) -> bytes:
    buf = io.BytesIO()
    s3.download_fileobj(AWS_BUCKET, output_key, buf)
    buf.seek(0)
    return buf.read()


# ----------------------------------------------------------
# Streamlit UI
# ----------------------------------------------------------
st.set_page_config(
    page_title="AI People Reader - Presentation Analysis Job Manager",
    layout="wide",
)

st.title("AI People Reader - Presentation Analysis Job Manager")

col_left, col_right = st.columns([1, 1.4])

# --------------------- Create New Job ---------------------
with col_left:
    st.subheader("Create New Job")

    mode = st.selectbox(
        "Mode",
        options=["dots", "passthrough"],
        index=0,
        help="ตอนนี้ worker ยังทำแค่ copy วิดีโอ แต่เผื่อโหมด dots ไว้ก่อน",
    )

    uploaded = st.file_uploader(
        "Upload video file",
        type=["mp4", "mov", "m4v", "avi"],
        help="Limit ~100MB ต่อไฟล์ (ขึ้นกับแผน Render)",
    )

    if st.button("Create job", type="primary", disabled=uploaded is None):
        if uploaded is None:
            st.error("กรุณาเลือกวิดีโอก่อน")
        else:
            try:
                job_id = new_job_id()
                input_key = f"{INPUT_PREFIX}/{job_id}/input.mp4"

                # upload วิดีโอขึ้น S3
                upload_fileobj_to_s3(uploaded, input_key)

                job = {
                    "job_id": job_id,
                    "mode": mode,
                    "status": "pending",
                    "created_at": utc_now_iso(),
                    "updated_at": utc_now_iso(),
                    "input_key": input_key,
                    "output_key": None,
                    "error": None,
                }

                put_job_json(job, PENDING_PREFIX)
                st.success(f"สร้างงานเรียบร้อยแล้ว: {job_id}")
            except Exception as exc:
                st.error(f"สร้างงานไม่สำเร็จ: {exc!r}")

# --------------------- Jobs table ---------------------
with col_right:
    st.subheader("Jobs")

    if st.button("Refresh job list"):
        st.experimental_rerun()

    jobs = list_jobs_all()
    if not jobs:
        st.info("ยังไม่มีงานในระบบ")
    else:
        df = pd.DataFrame(
            [
                {
                    "job_id": j.get("job_id"),
                    "status": j.get("status"),
                    "mode": j.get("mode"),
                    "created_at": j.get("created_at"),
                    "updated_at": j.get("updated_at"),
                    "error": j.get("error"),
                }
                for j in jobs
            ]
        ).sort_values("created_at", ascending=False)

        st.dataframe(
            df,
            hide_index=True,
            use_container_width=True,
        )

# --------------------- Download result video ---------------------
st.markdown("---")
st.subheader("Download result video ⬇️")

finished_jobs = [j for j in jobs if j.get("status") == "finished" and j.get("output_key")]
if not finished_jobs:
    st.info("No finished jobs yet.")
else:
    job_labels = [f"{j['job_id']}  ({j.get('mode')})" for j in finished_jobs]
    selected_idx = st.selectbox(
        "Select job",
        options=list(range(len(finished_jobs))),
        format_func=lambda i: job_labels[i],
    )

    selected_job = finished_jobs[selected_idx]
    output_key = selected_job["output_key"]

    if st.button("Prepare download"):
        try:
            video_bytes = download_video_bytes(output_key)
            st.download_button(
                label="Download result.mp4",
                data=video_bytes,
                file_name=f"{selected_job['job_id']}_result.mp4",
                mime="video/mp4",
            )
        except Exception as exc:
            st.error(f"ดาวน์โหลดไม่สำเร็จ: {exc!r}")
