import os
import json
import time
import logging
import tempfile
from datetime import datetime, timezone

import boto3

# Optional heavy libs – รองรับกรณี import ไม่ได้ ด้วยการ fail job สวย ๆ
try:
    import cv2  # type: ignore
except Exception:  # ImportError หรือ error อื่น ๆ
    cv2 = None  # type: ignore

try:
    import numpy as np  # type: ignore
except Exception:
    np = None  # type: ignore

# เรา "พยายาม" import mediapipe แต่ไม่บังคับว่าต้องมีแล้ว
# (requirements.txt ไม่มี mediapipe แล้ว ดังนั้นที่นี่จะกลายเป็น mp=None)
try:
    import mediapipe as mp  # type: ignore
    MP_HAS_SOLUTIONS = hasattr(mp, "solutions")
except Exception:
    mp = None  # type: ignore
    MP_HAS_SOLUTIONS = False

# ---------------------------------------------------------------------------
# Config & logger
# ---------------------------------------------------------------------------

AWS_BUCKET = os.getenv("AWS_BUCKET") or os.getenv("S3_BUCKET")
AWS_REGION = os.getenv("AWS_REGION", "ap-southeast-1")
POLL_INTERVAL = int(os.getenv("JOB_POLL_INTERVAL", "10"))

if not AWS_BUCKET:
    raise RuntimeError("Missing AWS_BUCKET (or S3_BUCKET) environment variable")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(threadName)s] %(message)s",
)
logger = logging.getLogger("worker")

s3 = boto3.client("s3", region_name=AWS_REGION)

JOBS_PREFIX = "jobs"
PENDING_PREFIX = f"{JOBS_PREFIX}/pending"
PROCESSING_PREFIX = f"{JOBS_PREFIX}/processing"
FINISHED_PREFIX = f"{JOBS_PREFIX}/finished"
FAILED_PREFIX = f"{JOBS_PREFIX}/failed"
OUTPUT_PREFIX = f"{JOBS_PREFIX}/output"


# ---------------------------------------------------------------------------
# Small S3 helpers
# ---------------------------------------------------------------------------


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def s3_get_json(key: str) -> dict:
    logger.info("[s3_get_json] key=%s", key)
    obj = s3.get_object(Bucket=AWS_BUCKET, Key=key)
    data = obj["Body"].read()
    return json.loads(data.decode("utf-8"))


def s3_put_json(key: str, payload: dict) -> None:
    # ใช้ ensure_ascii=False เพื่อรองรับข้อความภาษาไทยใน JSON
    body_str = json.dumps(payload, ensure_ascii=False)
    logger.info("[s3_put_json] key=%s size=%d bytes", key, len(body_str))
    body = body_str.encode("utf-8")
    s3.put_object(Bucket=AWS_BUCKET, Key=key, Body=body, ContentType="application/json")


def download_to_temp(key: str, suffix: str = ".mp4") -> str:
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    logger.info("[s3_download] %s -> %s", key, path)
    with open(path, "wb") as f:
        s3.download_fileobj(AWS_BUCKET, key, f)
    return path


def upload_from_path(path: str, key: str, content_type: str = "video/mp4") -> None:
    logger.info("[s3_upload] %s -> %s", path, key)
    with open(path, "rb") as f:
        s3.upload_fileobj(f, AWS_BUCKET, key, ExtraArgs={"ContentType": content_type})


def copy_video_in_s3(input_key: str, output_key: str) -> None:
    logger.info("[copy_object] %s -> %s", input_key, output_key)
    s3.copy_object(
        Bucket=AWS_BUCKET,
        CopySource={"Bucket": AWS_BUCKET, "Key": input_key},
        Key=output_key,
        ContentType="video/mp4",
    )


# ---------------------------------------------------------------------------
# Job lifecycle helpers
# ---------------------------------------------------------------------------


def list_pending_json_keys():
    # คืน key ของ job JSON ที่อยู่ใน jobs/pending/
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=AWS_BUCKET, Prefix=PENDING_PREFIX):
        for item in page.get("Contents", []):
            key = item["Key"]
            if key.endswith(".json"):
                yield key


def find_one_pending_job_key() -> str | None:
    for key in list_pending_json_keys():
        logger.info("[find_one_pending_job_key] found %s", key)
        return key
    logger.debug("[find_one_pending_job_key] no pending jobs")
    return None


def move_json(old_key: str, new_key: str, payload: dict) -> None:
    # เขียน payload ไป new_key แล้วลบ old_key
    s3_put_json(new_key, payload)
    if old_key != new_key:
        logger.info("[s3_delete] key=%s", old_key)
        s3.delete_object(Bucket=AWS_BUCKET, Key=old_key)


def update_status(job: dict, status: str, error: str | None = None) -> dict:
    job["status"] = status
    job["updated_at"] = utc_now_iso()
    if error is not None:
        job["error"] = error
    return job


# ---------------------------------------------------------------------------
# Dots (Johansson) processing — version ไม่ใช้ Mediapipe
# ---------------------------------------------------------------------------


def process_dots_video(input_key: str, output_key: str, multi_person: bool = False) -> None:
    """
    สร้างวิดีโอ Johansson-style dots โดยใช้ OpenCV + Background subtraction
    (ไม่พึ่ง Mediapipe):
      - อ่านวิดีโอจาก S3
      - ใช้ background subtractor หา silhouette
      - สุ่มเลือกจุดตาม contour ของร่างกาย
      - วาดจุดสีขาวลงบนพื้นหลังดำ
    """
    if cv2 is None or np is None:
        raise RuntimeError(
            "Johansson dots mode requires OpenCV and NumPy to be installed"
        )

    input_path = download_to_temp(input_key, suffix=".mp4")
    out_path = tempfile.mktemp(suffix=".mp4")

    logger.info(
        "[dots] starting Johansson processing (no Mediapipe) input=%s out=%s multi_person=%s",
        input_path,
        out_path,
        multi_person,
    )

    cap = cv2.VideoCapture(input_path)  # type: ignore[arg-type]
    if not cap.isOpened():
        cap.release()
        raise RuntimeError("Could not open input video")

    # ใช้ metadata จากวิดีโอเป็นขนาดมาตรฐาน
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    # กันกรณี metadata พัง (เช่น width/height = 0)
    if width <= 0 or height <= 0:
        ok, frame0 = cap.read()
        if not ok:
            cap.release()
            raise RuntimeError("Cannot read any frame from input video")
        height, width = frame0.shape[:2]
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # ย้อนกลับไปเฟรมแรก

    # *** สำคัญ: encoding เหมือนเวอร์ชันที่ QuickTime เปิดได้ ***
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        writer.release()
        raise RuntimeError("Could not open VideoWriter for output")

    # Background subtractor สำหรับหา silhouette
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(
        history=500, varThreshold=25, detectShadows=False
    )
    kernel = np.ones((3, 3), np.uint8)
    RADIUS = 5  # ขนาดจุดเท่ากันทุกเฟรม
    MAX_DOTS = 20

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            # บังคับทุกเฟรมให้ขนาดเท่ากับ (width, height)
            frame = cv2.resize(frame, (width, height))

            # หาร่างกายด้วย background subtraction
            fgmask = bg_subtractor.apply(frame)

            # ทำความสะอาด noise
            _, thresh = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
            thresh = cv2.dilate(thresh, kernel, iterations=2)

            contours, _ = cv2.findContours(
                thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            # สร้างพื้นหลังดำขนาดเดียวกันทุกเฟรม
            black = np.zeros((height, width, 3), dtype=np.uint8)

            if contours:
                # ใช้ contour ที่ใหญ่สุดเป็น silhouette หลัก
                largest = max(contours, key=cv2.contourArea)
                if cv2.contourArea(largest) > 100:  # กันกรณี noise เล็ก ๆ
                    epsilon = 0.01 * cv2.arcLength(largest, True)
                    hull = cv2.approxPolyDP(largest, epsilon, True)
                    pts = hull.reshape(-1, 2)

                    if len(pts) > 0:
                        step = max(1, len(pts) // MAX_DOTS)
                        for (x, y) in pts[::step]:
                            x = int(x)
                            y = int(y)
                            if 0 <= x < width and 0 <= y < height:
                                cv2.circle(
                                    black,
                                    (x, y),
                                    RADIUS,
                                    (255, 255, 255),
                                    -1,
                                    lineType=cv2.LINE_AA,
                                )

            writer.write(black)

    finally:
        cap.release()
        writer.release()
        try:
            upload_from_path(out_path, output_key)
        finally:
            for path in (input_path, out_path):
                try:
                    os.remove(path)
                except OSError:
                    pass


# ---------------------------------------------------------------------------
# Job processor
# ---------------------------------------------------------------------------


def process_job(job_json_key: str) -> None:
    raw_job = s3_get_json(job_json_key)

    job_id = raw_job.get("job_id")
    mode = raw_job.get("mode", "passthrough")
    input_key = raw_job.get("input_key")

    if not job_id:
        raise ValueError("Job JSON missing 'job_id'")
    if not input_key:
        raise ValueError("Job JSON missing 'input_key'")

    # ถ้า app ไม่ใส่ output_key มา ใช้ default path เดิม
    output_key = raw_job.get("output_key") or f"{OUTPUT_PREFIX}/{job_id}/result.mp4"

    logger.info(
        "[process_job] job_id=%s mode=%s input_key=%s output_key=%s",
        job_id,
        mode,
        input_key,
        output_key,
    )

    # ย้าย JSON ไป processing
    job = dict(raw_job)
    job = update_status(job, "processing")
    job["output_key"] = output_key

    processing_key = f"{PROCESSING_PREFIX}/{job_id}.json"
    move_json(job_json_key, processing_key, job)

    try:
        # รองรับทั้งชื่อ mode แบบเก่าและแบบใหม่
        if mode in ("dots", "dots_1p", "dots_single"):
            process_dots_video(input_key, output_key, multi_person=False)
        elif mode in ("dots_2p", "dots_multi"):
            process_dots_video(input_key, output_key, multi_person=True)
        else:
            # default: passthrough / copy เฉย ๆ
            copy_video_in_s3(input_key, output_key)

        job = update_status(job, "finished", error=None)
        finished_key = f"{FINISHED_PREFIX}/{job_id}.json"
        move_json(processing_key, finished_key, job)
        logger.info("[process_job] job_id=%s finished", job_id)

    except Exception as exc:
        logger.exception("[process_job] job_id=%s FAILED: %s", job_id, exc)
        job = update_status(job, "failed", error=str(exc))
        failed_key = f"{FAILED_PREFIX}/{job_id}.json"
        move_json(processing_key, failed_key, job)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def main() -> None:
    logger.info("====== AI People Reader Worker (Presentation / Johansson dots) ======")
    logger.info("Using bucket: %s", AWS_BUCKET)
    logger.info("Region       : %s", AWS_REGION)
    logger.info("Poll every   : %s seconds", POLL_INTERVAL)
    logger.info("MP available : %s", bool(mp and MP_HAS_SOLUTIONS))
    logger.info("cv2 available: %s", cv2 is not None)
    logger.info("numpy avail. : %s", np is not None)

    while True:
        try:
            job_key = find_one_pending_job_key()
            if job_key:
                process_job(job_key)
            else:
                time.sleep(POLL_INTERVAL)
        except Exception as exc:
            logger.exception("[main] Unexpected error: %s", exc)
            time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()
