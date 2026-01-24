import os
import json
import tempfile
from typing import List, Optional

import streamlit as st
import boto3
from botocore.exceptions import ClientError

st.set_page_config(page_title="S3 Browser", layout="wide")
st.title("📦 S3 Browser (Read-only)")

AWS_BUCKET = os.getenv("AWS_BUCKET") or os.getenv("S3_BUCKET")
AWS_REGION = os.getenv("AWS_REGION", "ap-southeast-1")

with st.expander("🔧 Environment", expanded=False):
    st.write("AWS_BUCKET =", AWS_BUCKET)
    st.write("AWS_REGION =", AWS_REGION)

if not AWS_BUCKET:
    st.error("Missing AWS_BUCKET (or S3_BUCKET)")
    st.stop()

s3 = boto3.client("s3", region_name=AWS_REGION)

def list_objects(prefix: str, max_keys: int = 200) -> List[str]:
    keys: List[str] = []
    try:
        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=AWS_BUCKET, Prefix=prefix):
            for obj in page.get("Contents", []) or []:
                keys.append(obj["Key"])
                if len(keys) >= max_keys:
                    return keys
        return keys
    except ClientError as e:
        st.error("S3 list failed")
        st.exception(e)
        return []

def list_folders(prefix: str) -> List[str]:
    out: List[str] = []
    try:
        resp = s3.list_objects_v2(Bucket=AWS_BUCKET, Prefix=prefix, Delimiter="/")
        for p in resp.get("CommonPrefixes", []) or []:
            out.append(p["Prefix"])
        return out
    except ClientError as e:
        st.error("S3 folder list failed")
        st.exception(e)
        return []

def get_bytes(key: str) -> bytes:
    obj = s3.get_object(Bucket=AWS_BUCKET, Key=key)
    return obj["Body"].read()

def guess_content_type(key: str) -> str:
    k = key.lower()
    if k.endswith(".mp4"):
        return "video/mp4"
    if k.endswith(".mov"):
        return "video/quicktime"
    if k.endswith(".m4v"):
        return "video/x-m4v"
    if k.endswith(".webm"):
        return "video/webm"
    if k.endswith(".json"):
        return "application/json"
    if k.endswith(".pdf"):
        return "application/pdf"
    if k.endswith(".docx"):
        return "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    return "application/octet-stream"

def presign(
    key: str,
    exp: int = 3600,
    filename: Optional[str] = None,
    content_type: Optional[str] = None,
) -> Optional[str]:
    """
    ✅ Force download (not open in browser tab) by setting:
      ResponseContentDisposition = attachment; filename="..."
    """
    try:
        params = {"Bucket": AWS_BUCKET, "Key": key}

        if filename:
            params["ResponseContentDisposition"] = f'attachment; filename="{filename}"'

        if content_type:
            params["ResponseContentType"] = content_type

        return s3.generate_presigned_url(
            "get_object",
            Params=params,
            ExpiresIn=exp,
        )
    except ClientError:
        return None

def is_video(key: str) -> bool:
    return key.lower().endswith((".mp4", ".mov", ".m4v", ".webm"))

def is_json(key: str) -> bool:
    return key.lower().endswith(".json")

def temp_download(key: str) -> str:
    data = get_bytes(key)
    tf = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(key)[1])
    tf.write(data)
    tf.flush()
    return tf.name

st.subheader("Quick S3 test")
prefix = st.text_input("Prefix", value="jobs/")
if st.button("List objects"):
    keys = list_objects(prefix)
    st.code("\n".join(keys[:200]) if keys else "No objects found")

st.divider()
st.subheader("Browse jobs")

root = st.text_input("Root folder", value="jobs/")
folders = list_folders(root)

if folders:
    labels = [f[len(root):].strip("/") for f in folders]
    idx = st.selectbox("Select job", list(range(len(labels))), format_func=lambda i: labels[i])
    job_prefix = folders[idx]
else:
    job_prefix = st.text_input("Manual job prefix", value=root)

out_folder = st.text_input("Output folder", value="output/")
out_prefix = job_prefix.rstrip("/") + "/" + out_folder.strip("/") + "/"

st.caption(f"Listing: {out_prefix}")
files = list_objects(out_prefix)

if files:
    key = st.selectbox("Select file", files)
    st.write("Key:", key)

    # ✅ Force download URL (attachment)
    forced_name = os.path.basename(key) or "download.bin"
    url = presign(
        key,
        exp=3600,
        filename=forced_name,
        content_type=guess_content_type(key),
    )
    if url:
        st.link_button("⬇️ Download (file)", url)

    st.markdown("### Preview")
    try:
        if is_json(key):
            data = get_bytes(key)
            st.json(json.loads(data.decode("utf-8", errors="replace")))
        elif is_video(key):
            st.video(temp_download(key))
        else:
            st.download_button(
                "Download file (direct)",
                data=get_bytes(key),
                file_name=os.path.basename(key),
            )
    except Exception as e:
        st.error("Preview failed")
        st.exception(e)
else:
    st.warning("No files found")
