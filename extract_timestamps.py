#!/usr/bin/env python3
"""
Kerala Political Poster — Image Timestamp Sorter (Improved OCR)
================================================================
Sorts images into date-based folders using timestamps extracted from
GPS Map Camera overlays via multi-strategy OCR.

Key improvements over v1:
  • Multi-region OCR  — tries 4 different crops of the GPS strip
  • Multi-preprocess  — inverted + direct + CLAHE for each region
  • OCR digit repair  — fixes O→0, l→1, I→1, B→8, S→5 inside numeric tokens
  • Multi-separator   — handles / - | and mixed separators in dates/times
  • Multi-format      — parses DD/MM/YYYY, YYYY-MM-DD, MMM DD YYYY, 2-digit year
  • Multi-PSM         — tries PSM 6 and PSM 4 when PSM 6 yields nothing
  • Parallel workers  — thread-pool for speed

Usage:
    python timestamp_sorter.py                  # uses config below
    python timestamp_sorter.py --help
"""

import os
import re
import shutil
import threading
import argparse
import concurrent.futures
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pytesseract

# ─── CONFIGURATION ────────────────────────────────────────────────────────────
SOURCE_DIR   = '/Users/harryfrz/image-mapping/output_images_final/Nenmara'
BASE_DEST_DIR = '/Users/harryfrz/image-mapping/test_run'
MAX_WORKERS  = 4

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp', '.tiff'}

# ─── Thread safety ────────────────────────────────────────────────────────────
_print_lock = threading.Lock()
_file_lock  = threading.Lock()

def safe_print(msg: str):
    with _print_lock:
        print(msg)

# ─── OCR preprocessing strategies ────────────────────────────────────────────
def _scale(img, factor=3.0):
    return cv2.resize(img, None, fx=factor, fy=factor,
                      interpolation=cv2.INTER_CUBIC)

def _to_gray(img):
    if len(img.shape) == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def preprocess_invert_otsu(region):
    """White text on dark BG → invert → Otsu. Best for GPS Map Camera."""
    gray = _to_gray(_scale(region))
    inv  = cv2.bitwise_not(gray)
    _, th = cv2.threshold(inv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th

def preprocess_direct_otsu(region):
    """Dark text on light BG → Otsu directly."""
    gray = _to_gray(_scale(region))
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th

def preprocess_clahe_invert(region):
    """CLAHE equalization → invert → Otsu. Handles low-contrast overlays."""
    gray  = _to_gray(_scale(region))
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    eq    = clahe.apply(gray)
    inv   = cv2.bitwise_not(eq)
    _, th = cv2.threshold(inv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th

def preprocess_adaptive(region):
    """Adaptive threshold — best when lighting is uneven across the strip."""
    gray = _to_gray(_scale(region, factor=4.0))
    inv  = cv2.bitwise_not(gray)
    th   = cv2.adaptiveThreshold(inv, 255,
               cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 4)
    return th

PREPROCESS_FNS = [
    preprocess_invert_otsu,
    preprocess_direct_otsu,
    preprocess_clahe_invert,
    preprocess_adaptive,
]

# ─── Strip region extraction ──────────────────────────────────────────────────
def get_gps_regions(img: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Return candidate sub-regions of the GPS Map Camera overlay strip.
    The strip occupies the bottom ~28% of the image.
    Layout (approximately):
        [satellite map thumbnail | text panel with timestamp + location]
    """
    h, w = img.shape[:2]
    strip_y = int(h * 0.72)
    strip   = img[strip_y:, :]
    sh, sw  = strip.shape[:2]

    # Thumbnail is roughly the leftmost 20-25% of the strip
    text_x = sw // 4

    return {
        "full_strip":  strip,
        "text_panel":  strip[:, text_x:],            # skip thumbnail
        "text_top":    strip[: sh // 2, text_x:],    # timestamp area
        "text_bottom": strip[sh // 2:, text_x:],     # location area
    }

# ─── OCR digit / punctuation repair ──────────────────────────────────────────
_DATE_TOKEN  = re.compile(r'[\dOolIlSBZGq]{1,4}[\/\-\|\\][\dOolIlSBZGq]{1,2}[\/\-\|\\][\dOolIlSBZGq]{2,4}')
_TIME_TOKEN  = re.compile(r'[\dOolIlSBZGq]{1,2}[:;,\.·][\dOolIlSBZGq]{2}(?:[:;,\.][\dOolIlSBZGq]{2})?(?:\s*[AaPp][Mm])?')

_DIGIT_MAP = str.maketrans({
    'O': '0', 'o': '0',
    'l': '1', 'L': '1', 'I': '1',
    'S': '5',
    'B': '8',
    'Z': '2',
    'G': '6',
    'q': '9', 'Q': '9',
})

def _fix_token(m: re.Match) -> str:
    tok = m.group(0)
    # Normalise separators inside dates
    tok = tok.replace('|', '/').replace('\\', '/').replace(' / ', '/')
    return tok.translate(_DIGIT_MAP)

def repair_ocr_text(text: str) -> str:
    """Fix common OCR misreads in date/time token regions only."""
    text = _DATE_TOKEN.sub(_fix_token, text)
    text = _TIME_TOKEN.sub(_fix_token, text)
    # Colon substitute in time (but not in date)
    text = re.sub(r'(?<=\d)[;,·](?=\d{2}\b)', ':', text)
    return text

# ─── Timestamp parsing ────────────────────────────────────────────────────────
_MONTH_MAP = {
    'JAN': '01','FEB': '02','MAR': '03','APR': '04',
    'MAY': '05','JUN': '06','JUL': '07','AUG': '08',
    'SEP': '09','OCT': '10','NOV': '11','DEC': '12',
}

_DATE_PATTERNS = [
    # DD/MM/YYYY  or  MM/DD/YYYY
    re.compile(r'\b(\d{1,2})[\/\-](\d{1,2})[\/\-](\d{4})\b'),
    # YYYY/MM/DD  or  YYYY-MM-DD
    re.compile(r'\b(\d{4})[\/\-](\d{1,2})[\/\-](\d{1,2})\b'),
    # DD/MM/YY
    re.compile(r'\b(\d{1,2})[\/\-](\d{1,2})[\/\-](\d{2})\b'),
    # MMM DD, YYYY  (e.g. Mar 19, 2026)
    re.compile(r'\b(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)[A-Z]*\s+(\d{1,2}),?\s+(\d{4})\b'),
    # DD MMM YYYY  (e.g. 19 Mar 2026)
    re.compile(r'\b(\d{1,2})\s+(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)[A-Z]*\s+(\d{4})\b'),
]

_TIME_PATTERNS = [
    re.compile(r'\b(\d{1,2})[:\.·](\d{2})[:\.·](\d{2})\s*([AaPp][Mm])\b'),  # HH:MM:SS AM/PM
    re.compile(r'\b(\d{1,2})[:\.·](\d{2})\s*([AaPp][Mm])\b'),                # HH:MM AM/PM
    re.compile(r'\b(\d{1,2})[:\.·](\d{2})[:\.·](\d{2})\b'),                  # HH:MM:SS 24h
    re.compile(r'\b(\d{1,2})[:\.·](\d{2})\b'),                                # HH:MM 24h
]

def _fuzzy_parse_date(text: str) -> Optional[Tuple[int, int, int]]:
    """
    Fallback date parser for heavily corrupted OCR output.
    Handles:
      - '{' '}' as separators  (e.g. "21{93/2026" → "21/03/2026")
      - Merged day+month       (e.g. "2143/2026"  → day=21, month=03)
      - '9' or '4' misread as '0' in month position
    Only called when the standard _parse_date returns None.
    """
    # Normalize exotic separators to /
    t = text
    for ch in ['{', '}']:
        t = t.replace(ch, '/')

    # Try standard pattern again after normalization
    pat = re.compile(r'\b(\d{1,2})[\/\-](\d{1,2})[\/\-](\d{4})\b')
    for m in pat.finditer(t):
        a, b, c = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if 2000 <= c <= 2100:
            yr, mo, d = c, b, a
        elif 2000 <= a <= 2100:
            yr, mo, d = a, b, c
        else:
            continue
        # Fix corrupted month digit: '9'→'0', '4'→'0', '8'→'0'
        if mo > 12:
            mo_str = str(mo)
            for bad, good in [('9','0'), ('4','0'), ('8','0')]:
                mo_fixed = int(mo_str.replace(bad, good))
                if 1 <= mo_fixed <= 12:
                    mo = mo_fixed
                    break
        if 1 <= mo <= 12 and 1 <= d <= 31 and 2000 <= yr <= 2100:
            return yr, mo, d

    # Looser: find a 4-digit year, reconstruct day+month from merged preceding digits
    # e.g. "2143/2026" → year=2026, merged="2143" → day=21, month=43→03
    year_m = re.search(r'\b(20\d{2})\b', t)
    if year_m:
        yr = int(year_m.group(1))
        before = t[:year_m.start()]
        nums = re.findall(r'\d+', before)
        if nums:
            last = nums[-1]
            if len(last) == 4:
                d_part = int(last[:2])
                m_part = int(last[2:])
                if m_part > 12:
                    for bad in ['9', '4', '8']:
                        m_try = int(str(m_part).replace(bad, '0'))
                        if 1 <= m_try <= 12:
                            m_part = m_try
                            break
                if 1 <= m_part <= 12 and 1 <= d_part <= 31:
                    return yr, m_part, d_part
    return None


def _parse_date(upper_text: str) -> Optional[Tuple[int, int, int]]:
    """Returns (year, month, day) or None. Tries standard then fuzzy."""
    for pat in _DATE_PATTERNS:
        m = pat.search(upper_text)
        if not m:
            continue
        g = m.groups()

        # Month-name formats
        if len(g) == 3 and not g[0].isdigit():           # MMM DD YYYY
            mon = _MONTH_MAP.get(g[0][:3], None)
            if not mon:
                continue
            return int(g[2]), int(mon), int(g[1])
        if len(g) == 3 and g[1] in _MONTH_MAP:            # DD MMM YYYY
            return int(g[2]), int(_MONTH_MAP[g[1][:3]]), int(g[0])

        a, b, c = int(g[0]), int(g[1]), int(g[2])

        # Distinguish YYYY-MM-DD vs DD/MM/YYYY by magnitude
        if a > 31:                                         # YYYY-MM-DD
            yr, mo, dy = a, b, c
        elif c < 100:                                      # 2-digit year DD/MM/YY
            yr, mo, dy = 2000 + c, b, a
        else:                                              # DD/MM/YYYY
            yr, mo, dy = c, b, a

        # Sanity check
        if 1 <= mo <= 12 and 1 <= dy <= 31 and 2000 <= yr <= 2100:
            return yr, mo, dy

    # Standard patterns failed — try fuzzy fallback (corrupted separators/digits)
    return _fuzzy_parse_date(upper_text)

def _parse_time(upper_text: str) -> Optional[Tuple[int, int, int]]:
    """Returns (hour_24, minute, second) or None."""
    for pat in _TIME_PATTERNS:
        m = pat.search(upper_text)
        if not m:
            continue
        g = m.groups()

        if len(g) == 4:        # HH:MM:SS AM/PM
            h, mi, s, ampm = int(g[0]), int(g[1]), int(g[2]), g[3].upper()
            h = h % 12 + (12 if ampm == 'PM' else 0)
            return h, mi, s
        if len(g) == 3:
            if g[2].upper() in ('AM', 'PM'):   # HH:MM AM/PM
                h, mi, ampm = int(g[0]), int(g[1]), g[2].upper()
                h = h % 12 + (12 if ampm == 'PM' else 0)
                return h, mi, 0
            else:                               # HH:MM:SS 24h
                return int(g[0]), int(g[1]), int(g[2])
        if len(g) == 2:                         # HH:MM 24h
            h, mi = int(g[0]), int(g[1])
            if 0 <= h <= 23 and 0 <= mi <= 59:
                return h, mi, 0

    return None

def parse_timestamp(raw_text: str) -> Optional[datetime]:
    """
    Extract a datetime from OCR text.
    Returns datetime or None.
    Date-only images get time 00:00:00 (not 12:00 AM).
    """
    if not raw_text:
        return None

    # Repair OCR glyphs FIRST (before upper-casing, so 'l'→'1' works before 'l'→'L')
    text = repair_ocr_text(raw_text)
    # Flatten multi-line and upper-case for pattern matching
    text = ' '.join(text.upper().split())

    date_parts = _parse_date(text)

    # If standard+fuzzy both failed on the uppercased text,
    # try fuzzy directly on the raw text (preserves { } chars before upper mangling)
    if not date_parts:
        date_parts = _fuzzy_parse_date(raw_text)

    time_parts = _parse_time(text)

    if not date_parts:
        return None

    yr, mo, dy = date_parts

    if time_parts:
        hh, mi, ss = time_parts
        try:
            return datetime(yr, mo, dy, hh, mi, ss)
        except ValueError:
            pass

    # Date-only: explicitly use 00:00:00
    try:
        return datetime(yr, mo, dy, 0, 0, 0)
    except ValueError:
        return None

# ─── Multi-strategy OCR ───────────────────────────────────────────────────────
def extract_timestamp_from_image(image_path: str) -> Tuple[Optional[datetime], str]:
    """
    Try all region × preprocessing combinations until a timestamp is found.
    Returns (datetime_or_None, best_ocr_text).
    """
    img = cv2.imread(image_path)
    if img is None:
        return None, ""

    regions = get_gps_regions(img)
    best_text = ""

    for region_name, region in regions.items():
        if region.size == 0:
            continue
        for prep_fn in PREPROCESS_FNS:
            try:
                processed = prep_fn(region)
            except Exception:
                continue

            for psm in (6, 4):
                try:
                    cfg  = f'--oem 3 --psm {psm}'
                    text = pytesseract.image_to_string(processed, config=cfg)
                except Exception:
                    continue

                if not best_text and text.strip():
                    best_text = text.strip()

                ts = parse_timestamp(text)
                if ts:
                    return ts, text.strip()

    return None, best_text

# ─── Single-image worker ──────────────────────────────────────────────────────
def process_single_image(args: tuple[str, str, str]):
    filename, source_dir, dest_dir = args
    filepath = os.path.join(source_dir, filename)

    ts, ocr_raw = extract_timestamp_from_image(filepath)

    # Build folder key and new filename suffix
    if ts:
        date_key = ts.strftime('%Y-%m-%d')
        # If time is exactly midnight AND we got it from a date-only parse → use 0000
        is_date_only = (ts.hour == 0 and ts.minute == 0 and ts.second == 0)
        if is_date_only:
            time_key = "0000"
        else:
            h12      = ts.hour % 12 or 12
            time_key = f"{h12}{ts.strftime('%M')}{ts.strftime('%p')}"
        status   = f"✓  {date_key}  {time_key}"
    else:
        date_key = "unknown_date"
        time_key = "notime"
        snippet  = ocr_raw.replace('\n', ' ')[:70]
        status   = f"✗  FAILED  OCR: '{snippet}'"

    base_name, ext = os.path.splitext(filename)

    with _file_lock:
        target_dir = os.path.join(dest_dir, date_key)
        os.makedirs(target_dir, exist_ok=True)

        if time_key == "notime":
            new_name = filename
        else:
            new_name = f"{base_name}_{time_key}{ext}"

        dst = os.path.join(target_dir, new_name)
        counter = 1
        while os.path.exists(dst):
            suffix = f"_{counter}"
            new_name = (f"{base_name}_{time_key}{suffix}{ext}"
                        if time_key != "notime"
                        else f"{base_name}{suffix}{ext}")
            dst = os.path.join(target_dir, new_name)
            counter += 1

        shutil.copy2(filepath, dst)

    safe_print(f"[{status}]  {filename}  →  {date_key}/{new_name}")

# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Sort GPS-tagged political poster images by timestamp."
    )
    parser.add_argument('--source',   default=SOURCE_DIR,
                        help=f'Input folder (default: {SOURCE_DIR})')
    parser.add_argument('--dest',     default=BASE_DEST_DIR,
                        help=f'Output folder (default: {BASE_DEST_DIR})')
    parser.add_argument('--workers',  type=int, default=MAX_WORKERS,
                        help=f'Parallel workers (default: {MAX_WORKERS})')
    args = parser.parse_args()

    os.makedirs(args.dest, exist_ok=True)

    images = [
        f for f in os.listdir(args.source)
        if Path(f).suffix.lower() in IMAGE_EXTENSIONS
    ]

    if not images:
        print(f"No images found in {args.source}")
        return

    print(f"Found {len(images)} images.")
    print(f"Workers: {args.workers}  |  Output: {args.dest}")
    print("=" * 70)

    jobs = [(f, args.source, args.dest) for f in images]

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        pool.map(process_single_image, jobs)

    print("=" * 70)
    print(f"Done. Output: {args.dest}")


if __name__ == '__main__':
    main()