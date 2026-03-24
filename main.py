#!/usr/bin/env python3
import os
import re
import json
import shutil
import logging
import argparse
import base64
import requests
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image
import pytesseract
from rapidfuzz import process, fuzz
from shapely.geometry import Point

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# GeoJSON data cache
_geojson_cache: Optional[dict] = None
_constituency_polygons: Optional[dict] = None

# OSM Nominatim cache
_osm_cache: dict[str, Optional[dict]] = {}


def reverse_geocode_osm(lat: float, lon: float) -> Optional[dict]:
    cache_key = f"{lat},{lon}"
    if cache_key in _osm_cache:
        return _osm_cache[cache_key]

    try:
        import requests
        url = "https://nominatim.openstreetmap.org/reverse"
        params = {
            "lat": lat,
            "lon": lon,
            "format": "json",
            "zoom": 18,
            "addressdetails": 1,
        }
        headers = {"User-Agent": "ImageMapping/1.0"}

        response = requests.get(url, params=params, headers=headers, timeout=10)
        if response.status_code == 200:
            data = response.json()
            result = {
                "display_name": data.get("display_name", ""),
                "address": data.get("address", {}),
                "place_id": data.get("place_id"),
            }
            _osm_cache[cache_key] = result
            return result
    except Exception as e:
        log.warning(f"OSM reverse geocode failed: {e}")

    _osm_cache[cache_key] = None
    return None


def extract_place_from_osm(osm_data: dict) -> Optional[list[str]]:
    address = osm_data.get("address", {})
    
    candidates = []
    
    priority_keys = ["town", "village", "municipality", "suburb", "neighbourhood", "hamlet", "city", "county"]
    
    for key in priority_keys:
        if key in address:
            candidates.append(address[key].lower())
    
    display_name = osm_data.get("display_name", "")
    if display_name:
        parts = [p.strip() for p in display_name.split(",")]
        for part in parts[:4]:
            if part.lower() not in ["kerala", "india"]:
                candidates.append(part.lower())
    
    seen = set()
    unique_candidates = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            unique_candidates.append(c)
    
    return unique_candidates if unique_candidates else None


def load_geojson(geojson_path: str) -> dict:
    global _geojson_cache
    if _geojson_cache is None:
        with open(geojson_path, "r", encoding="utf-8") as f:
            _geojson_cache = json.load(f)
    return _geojson_cache


def build_constituency_polygons(geojson_path: str) -> dict:
    global _constituency_polygons
    if _constituency_polygons is not None:
        return _constituency_polygons

    _constituency_polygons = {}

    path = Path(geojson_path)
    if path.suffix.lower() == ".shp":
        _constituency_polygons = _load_from_shapefile(geojson_path)
    else:
        _constituency_polygons = _load_from_geojson(geojson_path)

    log.info(f"Loaded {len(_constituency_polygons)} constituency polygons")
    return _constituency_polygons


def _load_from_shapefile(shapefile_path: str) -> dict:
    import geopandas as gpd

    gdf = gpd.read_file(shapefile_path)
    gdf = gdf[gdf["ST_NAME"].str.lower() == "kerala"]

    polygons = {}
    for _, row in gdf.iterrows():
        name = row["AC_NAME"]
        district = row["DIST_NAME"]
        geom = row.geometry
        if geom is not None:
            name_key = name.lower().replace(" ", "")
            polygons[name_key] = {
                "name": name,
                "district": district,
                "polygon": geom,
            }

    return polygons


def _load_from_geojson(geojson_path: str) -> dict:
    from shapely.geometry import shape

    geojson = load_geojson(geojson_path)
    polygons = {}

    for feature in geojson["features"]:
        name = feature["properties"]["Asmbly_Con"]
        district = feature["properties"]["District"]
        try:
            geom = shape(feature["geometry"])
            polygons[name.lower()] = {
                "name": name,
                "district": district,
                "polygon": geom,
            }
        except Exception as e:
            log.warning(f"Failed to parse polygon for {name}: {e}")

    return polygons


def extract_coordinates(ocr_text: str) -> Optional[tuple[float, float]]:
    text = ocr_text.upper()

    # More comprehensive patterns for GPS coordinates
    patterns = [
        r"LAT\s*[:.]?\s*([0-9.]+)\s*°?\s*[NS]?\s*[,/]?\s*LONG\s*[:.]?\s*([0-9.]+)\s*°?\s*[EW]?",
        r"Lat\s*[:.]?\s*([0-9.]+)\s*°\s*(?:N|S)?\s*[,/]?\s*Long\s*[:.]?\s*([0-9.]+)\s*°\s*(?:E|W)?",
        r"([0-9]{1,2}\.[0-9]{4,})\s*°?\s*N?\s*[,/]\s*([0-9]{1,2}\.[0-9]{4,})\s*°?\s*E?",
        r"([0-9]{1,2}\.[0-9]+)°?\s*N?\s*[,/]\s*([0-9]{1,2}\.[0-9]+)°?\s*E?",
        r"([0-9]{1,2}\.[0-9]+)[,\s]+([0-9]{1,2}\.[0-9]+)",
        r"([0-9]{1,2}\.[0-9]{5,})\s*,\s*([0-9]{1,2}\.[0-9]{5,})",
    ]

    for pattern in patterns:
        matches = re.findall(pattern, text)
        if matches:
            try:
                lat = float(matches[0][0])
                lon = float(matches[0][1])
                if 8.0 <= lat <= 13.0 and 74.0 <= lon <= 77.5:
                    return (lat, lon)
            except (ValueError, IndexError):
                continue

    # Try without direction check - look for any two decimal numbers that could be coordinates
    loose_pattern = r"([0-9]{1,2}\.[0-9]{4,})"
    all_matches = re.findall(loose_pattern, text)
    if len(all_matches) >= 2:
        try:
            lat = float(all_matches[0])
            lon = float(all_matches[1])
            # Be more lenient with boundary check
            if 8.0 <= lat <= 13.5 and 74.0 <= lon <= 78.0:
                return (lat, lon)
        except:
            pass

    return None


def find_constituency_by_coords(
    lat: float,
    lon: float,
    geojson_path: str,
) -> Optional[tuple[str, str]]:
    polygons = build_constituency_polygons(geojson_path)

    point = Point(lon, lat)

    for key, data in polygons.items():
        if data["polygon"].contains(point) or data["polygon"].touches(point):
            return (data["name"], data["district"])

    closest_name = None
    closest_dist = float("inf")
    closest_district = None

    for key, data in polygons.items():
        dist = data["polygon"].exterior.distance(point)
        if dist < closest_dist:
            closest_dist = dist
            closest_name = data["name"]
            closest_district = data["district"]

    if closest_dist < 0.5:  # More lenient threshold (about 50km)
        log.info(f"Using closest constituency: {closest_name} (dist={closest_dist:.4f})")
        return (closest_name, closest_district)

    return None


def load_constituencies(json_path: str) -> dict:
    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    # Build flat alias → canonical_key lookup
    alias_map: dict[str, str] = {}
    canonical_names: dict[str, str] = {}  # key → display name

    for key, meta in raw.items():
        display = key.replace("_", " ").title()
        canonical_names[key] = display
        for alias in meta.get("aliases", []):
            alias_map[alias.lower().strip()] = key

    return {"raw": raw, "alias_map": alias_map, "canonical_names": canonical_names}


def detect_and_crop_geotag_strip(image_path: str) -> tuple[np.ndarray, bool]:
    """
    Use OpenCV to detect and crop the geotag strip from the image.
    Returns: (cropped_strip, was_geotag_detected)
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Try multiple strip ratios to find the geotag
    best_strip = None
    best_score = 0

    for strip_ratio in [0.15, 0.20, 0.25, 0.28, 0.35, 0.40]:
        strip = gray[int(h * (1 - strip_ratio)):, :]

        # Check if strip has text-like features (high variance, edges)
        edges = cv2.Canny(strip, 50, 150)
        edge_count = np.sum(edges > 0)
        variance = np.var(strip)

        # Geotag strips typically have moderate edge density and variance
        score = edge_count * (variance / 1000)
        if score > best_score:
            best_score = score
            best_strip = strip

    if best_strip is None:
        best_strip = gray[int(h * 0.72):, :]

    # Upscale for better OCR
    strip_resized = cv2.resize(best_strip, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    # Apply image processing for better OCR
    denoised = cv2.fastNlMeansDenoising(strip_resized, None, 10, 7, 21)

    # Increase contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)

    # Sharpen
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(enhanced, -1, kernel)

    return sharpened, best_score > 500


def crop_geotag_region(image_path: str) -> np.ndarray:
    """Crop the geotag region from the image."""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    
    h, w = img.shape[:2]
    # Crop bottom 30% where geotag usually appears
    strip = img[int(h * 0.70):, :]
    return strip


def ollama_ocr(image_path: str, ollama_url: str = "http://localhost:11434") -> Optional[str]:
    """Use Ollama with GLM model for OCR."""
    try:
        # Crop geotag region
        strip = crop_geotag_region(image_path)
        
        # Encode to base64
        _, buffer = cv2.imencode('.png', strip)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # Call Ollama
        payload = {
            "model": "glm-ocr",
            "prompt": "Extract ONLY the geotag/location text from this image. Look for location names, GPS coordinates (Lat/Long), Kerala, India, addresses, place names. Return ONLY the location-related text found, nothing else. If no geotag text found, say 'NO_GEOTAG'.",
            "images": [img_base64],
            "stream": False
        }
        
        response = requests.post(
            f"{ollama_url}/api/generate",
            json=payload,
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            text = result.get("response", "").strip()
            if text and text != "NO_GEOTAG":
                return text
    except Exception as e:
        log.warning(f"Ollama OCR failed: {e}")
    
    return None


def ocr_strip(processed: np.ndarray) -> str:
    """Run Tesseract on the preprocessed strip with multiple configs."""
    pil = Image.fromarray(processed)
    
    # Try multiple OCR configs
    configs = [
        r"--oem 3 --psm 6 -l eng",
        r"--oem 3 --psm 4 -l eng",
        r"--oem 3 --psm 3 -l eng",
        r"--oem 3 --psm 11 -l eng",
    ]
    
    best_text = ""
    for config in configs:
        text = pytesseract.image_to_string(pil, config=config)
        if len(text) > len(best_text):
            best_text = text
    
    return best_text


def extract_location_tokens(ocr_text: str) -> tuple[list[str], bool]:
    """
    GPS Map Camera format (example):
        Kazhakkoottam, Kerala, India
        Satheedevam Ladies Pg 11 A, Snra, 11 A, Kazhakuttam
        Kazhakkoottam, Kerala 695582, India
        Lat 8.570406° Long 76.872536°

    Returns: (tokens, has_geotag_indicator)
    - has_geotag_indicator is True if the OCR text contains "Kerala" or coordinates
    """
    # Remove non-ASCII characters
    text = re.sub(r"[^\x00-\x7F]+", " ", ocr_text)

    candidates: list[str] = []

    # Check for geotag indicators
    has_kerala = bool(re.search(r"\bkerala\b", text, re.IGNORECASE))
    has_coordinates = bool(re.search(r"\blat\b|\blong\b|\b°\b", text, re.IGNORECASE))
    has_geotag_indicator = has_kerala or has_coordinates

    # Noise words to filter out (OCR noise, not location names)
    noise_words = {
        "kerala", "india", "google", "latitude", "longitude", "lat", "long", 
        "camera", "north", "south", "east", "west", "gmt", "road", "street", 
        "lane", "vidya", "vidyaniketan", "college", "school", 
        "hospital", "map", "mapcamera", "ladies", "pg", "hostel", "building", 
        "complex", "junction", "tower", "telecom", "bsnl", "vodafone", "airtel", 
        "jio", "network", "election", "campaign", "vote", "party", "bjp", 
        "congress", "cpim", "manifesto", "candidate", "support", "win", 
        "victory", "democracy",
    }

    # Pass 1: Look for "Word, Kerala" or "Word, Kerala PINCODE" patterns (most reliable)
    pattern1 = re.findall(
        r"([A-Z][a-z]{3,}(?:\s+[A-Z][a-z]{2,})*)\s*,\s*Kerala",
        text,
        flags=re.IGNORECASE,
    )
    for m in pattern1:
        cleaned = m.strip().lower()
        if len(cleaned) > 3 and cleaned not in noise_words:
            candidates.append(cleaned)

    # Pass 2: Look for "Word, Kerala, India" line patterns
    pattern2 = re.findall(
        r"([A-Z][a-z]{3,}(?:[\s-][A-Z][a-z]{2,})*)\s*,\s*Kerala\s*,\s*India",
        text,
        flags=re.IGNORECASE,
    )
    for m in pattern2:
        cleaned = m.strip().lower()
        if len(cleaned) > 3 and cleaned not in noise_words:
            candidates.append(cleaned)

    # Pass 3: Extract from lines containing "Kerala" or "India" or PIN codes (6 digits)
    pincode_pattern = r"\b\d{6}\b"
    has_pincode = bool(re.search(pincode_pattern, text))
    
    if has_kerala or has_pincode:
        for line in text.split("\n"):
            if re.search(r"\bkerala\b|\bindia\b|\d{6}", line, re.IGNORECASE):
                # Match proper nouns (capitalized words)
                words = re.findall(r"\b[A-Z][a-z]{2,}\b", line)
                for w in words:
                    w_lower = w.lower()
                    if w_lower not in noise_words and len(w_lower) > 3:
                        candidates.append(w_lower)

    # Pass 4: Look for specific Kerala location patterns (town/city names)
    # Common patterns in GPS camera overlays
    location_patterns = [
        r"\b([A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,})?)\s*,\s*Kerala",
        r"Kerala\s*,\s*India",
    ]
    
    # Deduplicate while preserving order
    seen: set[str] = set()
    result: list[str] = []
    for c in candidates:
        c = c.strip()
        if c and c not in seen:
            seen.add(c)
            result.append(c)

    return result, has_geotag_indicator


# ─── Constituency matching ─────────────────────────────────────────────────────
def match_constituency(
    tokens: list[str],
    alias_map: dict[str, str],
    canonical_names: dict[str, str],
    score_cutoff: int = 75,
) -> Optional[tuple[str, str, int]]:
    """
    Returns (canonical_key, display_name, score) or None.
    Tries exact alias lookup first, then fuzzy match.
    """
    all_aliases = list(alias_map.keys())

    best_key = None
    best_score = 0
    best_match = None

    for token in tokens:
        token = token.strip().lower()

        # 1. Exact alias match
        if token in alias_map:
            key = alias_map[token]
            return (key, canonical_names[key], 100)

        # 2. Fuzzy match against alias list
        result = process.extractOne(
            token,
            all_aliases,
            scorer=fuzz.WRatio,
            score_cutoff=score_cutoff,
        )
        if result:
            matched_alias, score, _ = result
            key = alias_map[matched_alias]
            if score > best_score:
                best_score = score
                best_key = key
                best_match = matched_alias

    if best_key:
        return (best_key, canonical_names[best_key], best_score)
    return None


# ─── Core sorting logic ───────────────────────────────────────────────────────
def sort_images(
    input_folder: str,
    output_folder: str,
    constituencies_json: str,
    boundary_path: str = "shapefile/India_AC.shp",
    dry_run: bool = False,
    score_cutoff: int = 75,
) -> dict:
    input_path = Path(input_folder)
    output_path = Path(output_folder)
    data = load_constituencies(constituencies_json)
    alias_map = data["alias_map"]
    canonical_names = data["canonical_names"]

    if boundary_path and Path(boundary_path).exists():
        build_constituency_polygons(boundary_path)

    image_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
    images = [
        f for f in input_path.iterdir()
        if f.is_file() and f.suffix.lower() in image_extensions
    ]

    if not images:
        log.warning("No images found in %s", input_folder)
        return {}

    log.info("Found %d image(s) to process.", len(images))

    results = {}

    for img_path in images:
        log.info("Processing: %s", img_path.name)
        try:
            # Try Ollama OCR first
            ollama_text = ollama_ocr(str(img_path))
            cv_detected = False
            
            if ollama_text:
                log.info("  Ollama OCR: %s", ollama_text[:100])
                ocr_text = ollama_text
            else:
                # Fallback to Tesseract
                processed, cv_detected = detect_and_crop_geotag_strip(str(img_path))
                ocr_text = ocr_strip(processed)
            
            log.debug("OCR raw output:\n%s", ocr_text)

            coords = extract_coordinates(ocr_text)
            tokens, has_geotag = extract_location_tokens(ocr_text)

            # Also consider CV detection
            has_geotag = has_geotag or cv_detected
            log.info("  Extracted tokens: %s (geotag indicator: %s)", tokens, has_geotag)

            match = None
            match_method = None
            polygons = None
            
            if boundary_path and Path(boundary_path).exists():
                polygons = build_constituency_polygons(boundary_path)

            # First check if OCR tokens contain constituency name directly
            if tokens:
                for token in tokens:
                    token_clean = token.lower().replace(" ", "")
                    # Direct match for known constituencies
                    if token_clean in canonical_names:
                        match = (token_clean, canonical_names[token_clean], 100)
                        match_method = "direct"
                        log.info("  ✓ Direct match from OCR: %s", match[1])
                        break
                    # Fuzzy match
                    fuzzy = match_constituency([token], alias_map, canonical_names, score_cutoff=85)
                    if fuzzy:
                        match = fuzzy
                        match_method = "text"
                        log.info("  ✓ Text fuzzy match: %s (score=%d)", match[1], match[2])
                        break

            # Use OSM to approximate constituency from coordinates
            if coords and not match:
                log.info("  Extracted coordinates: %s", coords)
                osm_data = reverse_geocode_osm(coords[0], coords[1])
                if osm_data:
                    log.info("  OSM place: %s", osm_data.get("display_name", "")[:80])
                    place_names = extract_place_from_osm(osm_data)
                    if place_names:
                        log.info("  OSM places: %s", place_names)
                        
                        for place in place_names:
                            osm_match = match_constituency(
                                [place], alias_map, canonical_names, score_cutoff=55
                            )
                            if osm_match:
                                match = osm_match
                                match_method = "osm"
                                log.info("  ✓ OSM matched: %s (score=%d)", match[1], match[2])
                                break

            # If OCR text was detected, try to match it
            if not match and tokens:
                match = match_constituency(tokens, alias_map, canonical_names, score_cutoff=55)
                if match:
                    match_method = "text"
                    log.info("  ✓ Text match: %s (score=%d)", match[1], match[2])
                else:
                    # Try each token individually with lower threshold
                    for token in tokens:
                        single_match = match_constituency(
                            [token], alias_map, canonical_names, score_cutoff=65
                        )
                        if single_match:
                            match = single_match
                            match_method = "text"
                            log.info("  ✓ Text match (single token): %s (score=%d)", match[1], match[2])
                            break

            # If OCR text was detected but no match, skip the image (don't add to others)
            ocr_detected = bool(ocr_text.strip())
            
            # If CV detected geotag but no OCR text, try OSM with coordinates
            if not match and not ocr_detected and cv_detected and coords:
                log.info("  CV detected geotag but no OCR text, trying OSM...")
                osm_data = reverse_geocode_osm(coords[0], coords[1])
                if osm_data:
                    place_names = extract_place_from_osm(osm_data)
                    if place_names:
                        for place in place_names:
                            osm_match = match_constituency(
                                [place], alias_map, canonical_names, score_cutoff=50
                            )
                            if osm_match:
                                match = osm_match
                                match_method = "osm_cv"
                                log.info("  ✓ OSM from CV detected: %s", match[1])
                                break

            if match:
                key, display, score = match
                dest_dir = output_path / display
                log.info("  ✓ Matched via %s: %s (score=%d)", match_method, display, score)
            else:
                dest_dir = output_path / "others"
                if ocr_detected:
                    log.info("  ✗ OCR detected but no match → others/")
                elif cv_detected:
                    log.info("  ✗ CV detected but no match → others/")
                else:
                    log.info("  ✗ No geotag detected → others/")

            dest_dir.mkdir(parents=True, exist_ok=True)
            dest_file = dest_dir / img_path.name

            # Handle filename collisions
            if dest_file.exists():
                stem = img_path.stem
                suffix = img_path.suffix
                counter = 1
                while dest_file.exists():
                    dest_file = dest_dir / f"{stem}_{counter}{suffix}"
                    counter += 1

            if not dry_run:
                shutil.copy2(str(img_path), str(dest_file))
                log.info("  → Copied to %s", dest_file)
            else:
                log.info("  [DRY RUN] Would copy to %s", dest_file)

            results[img_path.name] = {
                "status": "matched" if match else "unmatched",
                "constituency": display if match else None,
                "score": match[2] if match else None,
                "match_method": match_method,
                "coordinates": coords,
                "has_geotag_indicator": has_geotag,
                "ocr_tokens": tokens,
                "destination": str(dest_file),
            }

        except Exception as e:
            log.error("  Error processing %s: %s", img_path.name, e)
            results[img_path.name] = {"status": "error", "error": str(e)}

    # Summary
    matched = sum(1 for r in results.values() if r.get("status") == "matched")
    unmatched = sum(1 for r in results.values() if r.get("status") == "unmatched")
    errors = sum(1 for r in results.values() if r.get("status") == "error")
    log.info("\n── Summary ──────────────────────────────")
    log.info("  Total:     %d", len(images))
    log.info("  Matched:   %d", matched)
    log.info("  Unmatched: %d  (→ others/)", unmatched)
    log.info("  Errors:    %d", errors)

    return results


# ─── CLI ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Sort Kerala political poster images by constituency using OCR geotag."
    )
    parser.add_argument("input_folder", help="Folder containing input images")
    parser.add_argument("output_folder", help="Folder where sorted images will be placed")
    parser.add_argument(
        "--constituencies",
        default="constituencies.json",
        help="Path to constituencies JSON file (default: constituencies.json)",
    )
    parser.add_argument(
        "--boundary",
        default="shapefile/India_AC.shp",
        help="Path to constituency boundary file (GeoJSON or Shapefile)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview what would happen without copying files",
    )
    parser.add_argument(
        "--score-cutoff",
        type=int,
        default=75,
        help="Fuzzy match score threshold 0-100 (default: 75)",
    )
    parser.add_argument(
        "--report",
        help="Save a JSON report to this path",
    )
    args = parser.parse_args()

    results = sort_images(
        args.input_folder,
        args.output_folder,
        args.constituencies,
        boundary_path=args.boundary,
        dry_run=args.dry_run,
        score_cutoff=args.score_cutoff,
    )

    if args.report:
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        log.info("Report saved to %s", args.report)


if __name__ == "__main__":
    main()