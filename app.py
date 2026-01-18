# app.py  -- AI People Reader - Presentation Analysis Job Manager
import os
import io
import json
from datetime import datetime, timezone
from typing import List, Dict, Any

import streamlit as st
import boto3

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
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

# ------------------------------------------------------------------
# Small helpers
# ------------------------------------------------------------------


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_job_id() -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return ts


def upload_bytes_to_s3(data: bytes, key: str, content_type: str = "video/mp4") -> None:
    s3.upload_fileobj(
        io.BytesIO(data),
        AWS_BUCKET,
        key,
        ExtraArgs={"ContentType": content_type},
    )


def s3_get_json(key: str) -> Dict[str, Any]:
    obj = s3.get_object(Bucket=AWS_BUCKET, Key=key)
    data = obj["Body"].read()
    return json.loads(data.decode("utf-8"))


def list_all_jobs() -> List[Dict[str, Any]]:
    jobs: List[Dict[str, Any]] = []
    paginator = s3.get_paginator("list_objects_v2")

    for prefix in [PENDING_PREFIX, PROCESSING_PREFIX, FINISHED_PREFIX, FAILED_PREFIX]:
        for page in paginator.paginate(Bucket=AWS_BUCKET, Prefix=prefix):
            for item in page.get("Contents", []):
                key = item["Key"]
                if not key.endswith(".json"):
                    continue
                try:
                    job = s3_get_json(key)
                    jobs.append(job)
                except Exception:
                    # ถ้าอ่านไม่ได้ก็ข้ามไป
                    continue

    # sort ใหม่โดยใช้ created_at ล่าสุดอยู่บน
    def _key(j: Dict[str, Any]) -> str:
        return j.get("created_at", "")

    jobs.sort(key=_key, reverse=True)
    return jobs


def list_finished_jobs() -> List[Dict[str, Any]]:
    return [j for j in list_all_jobs() if j.get("status") == "finished"]


# ------------------------------------------------------------------
# Streamlit UI
# ------------------------------------------------------------------

st.set_page_config(
    page_title="AI People Reader - Presentation Analysis Job Manager",
    layout="wide",
)

st.title("AI People Reader - Presentation Analysis Job Manager")

col_left, col_right = st.columns([1, 2])

# --------------- Create New Job -----------------
with col_left:
    st.subheader("Create New Job")

    mode = st.selectbox("Mode", options=["dots"], index=0)

    upload = st.file_uploader(
        "Upload video file",
        type=["mp4", "mov", "m4v", "mpeg", "avi"],
    )

    if st.button("Create job", type="primary", disabled=upload is None):
        if upload is None:
            st.error("Please upload a video first.")
        else:
            # สร้าง job id และ key ต่าง ๆ
            job_id = new_job_id()
            input_key = f"{JOBS_PREFIX}/input/{job_id}.mp4"
            output_key = f"{OUTPUT_PREFIX}/{job_id}/result.mp4"

            # อัปโหลดวิดีโอ
            data = upload.read()
            upload_bytes_to_s3(data, input_key, content_type="video/mp4")

            # บันทึก job JSON ไปที่ pending
            now = utc_now_iso()
            job = {
                "job_id": job_id,
                "mode": mode,
                "status": "pending",
                "created_at": now,
                "updated_at": now,
                "input_key": input_key,
                "output_key": output_key,
                "error": None,
            }
            pending_key = f"{PENDING_PREFIX}/{job_id}.json"
            s3.put_object(
                Bucket=AWS_BUCKET,
                Key=pending_key,
                Body=json.dumps(job, ensure_ascii=False).encode("utf-8"),
                ContentType="application/json",
            )

            st.success(f"Created job {job_id}. Worker will process it shortly.")

# --------------- Jobs table -----------------
with col_right:
    st.subheader("Jobs ↻")

    if st.button("Refresh job list"):
        st.experimental_rerun()

    jobs = list_all_jobs()

    if not jobs:
        st.info("No jobs yet.")
    else:
        import pandas as pd

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
        )
        st.dataframe(df, use_container_width=True)


st.markdown("---")

# --------------- Download result video -----------------
st.subheader("Download result video ↴")

finished_jobs = list_finished_jobs()
if not finished_jobs:
    st.info("No finished jobs yet.")
else:
    job_ids = [j["job_id"] for j in finished_jobs]
    selected_id = st.selectbox("Select job", job_ids)

    selected_job = next(j for j in finished_jobs if j["job_id"] == selected_id)
    output_key = selected_job.get("output_key")

    if st.button("Prepare download"):
        if not output_key:
            st.error("This job has no output_key.")
        else:
            try:
                obj = s3.get_object(Bucket=AWS_BUCKET, Key=output_key)
                data = obj["Body"].read()
                st.download_button(
                    "Download result.mp4",
                    data=data,
                    file_name=f"{selected_id}_result.mp4",
                    mime="video/mp4",
                )
            except Exception as exc:
                st.error(f"Error downloading result: {exc}")
