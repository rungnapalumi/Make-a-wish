# report_core.py — shared report logic for both app.py and report_worker.py
#
# ✅ RULE: Copy/paste the SAME functions from app.py here WITHOUT changing logic.
#         app.py should only be UI, and import these functions.

# --- Imports (copy from app.py) ---
import os
import io
import math
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from docx import Document
from docx.shared import Pt, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK

from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, PageBreak, Image as RLImage,
    ListFlowable, ListItem, Table, TableStyle,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_RIGHT

try:
    import mediapipe as mp  # type: ignore
except Exception:
    mp = None

# --- Dataclasses (copy from app.py) ---
@dataclass
class CategoryResult:
    name_en: str
    name_th: str
    score: int
    scale: str
    positives: int
    total: int
    description: str = ""

@dataclass
class ReportData:
    client_name: str
    analysis_date: str
    video_length_str: str
    overall_score: int
    categories: list
    summary_comment: str
    generated_by: str

# --- Helpers that report_worker imports (copy from app.py) ---
def format_seconds_to_mmss(total_seconds: float) -> str:
    total_seconds = max(0, float(total_seconds))
    mm = int(total_seconds // 60)
    ss = int(round(total_seconds - mm * 60))
    if ss == 60:
        mm += 1
        ss = 0
    return f"{mm:02d}:{ss:02d}"

def get_video_duration_seconds(video_path: str) -> float:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0.0
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
    cap.release()
    if fps <= 0:
        return 0.0
    return float(frames / fps)

# --- IMPORTANT: Copy these functions exactly from app.py ---
# 1) analyze_video_mediapipe(...)
# 2) analyze_video_placeholder(...)
# 3) generate_effort_graph(...)
# 4) generate_shape_graph(...)
# 5) build_docx_report(...)
# 6) build_pdf_report(...)
#
# (Copy/paste the full bodies from app.py without changes.)
