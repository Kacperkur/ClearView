"""
feature_extractor.py — Forensic feature extraction for AI image detection.

Returns a fixed 102-dimensional feature vector per image.
All config values read from config.yaml. Never raises on a valid image file —
failed sub-features return documented sentinel values.

Feature vector layout:
  [0:6]    - ELA multi-scale: mean_q75, std_q75, mean_q85, std_q85, mean_q95, std_q95
  [6:16]   - DCT AC coefficient histogram (10 bins)
  [16:26]  - LBP histogram r=1, n_points=8  (10 bins, uniform)
  [26:52]  - LBP histogram r=3, n_points=24 (26 bins, uniform)
  [52:94]  - LBP histogram r=5, n_points=40 (42 bins, uniform)
  [94:96]  - Noise residual mean, std
  [96]     - LSB entropy R channel
  [97]     - LSB entropy G channel
  [98]     - LSB entropy B channel
  [99]     - EXIF MakerNote present (0 or 1)
  [100]    - EXIF field completeness score (0.0 - 1.0)
  [101]    - Eye highlight consistency score (-1.0 if no face detected)
Total: 102 dimensions
"""

from __future__ import annotations

import io
import os
import warnings
from pathlib import Path

import cv2
import exifread
import numpy as np
import pywt
from PIL import Image, ImageChops
from scipy.fftpack import dct
from skimage.feature import local_binary_pattern

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────

def _load_config() -> dict:
    import yaml
    root = Path(__file__).resolve().parent.parent
    with open(root / "config.yaml") as f:
        return yaml.safe_load(f)


_CFG: dict | None = None


def _cfg() -> dict:
    global _CFG
    if _CFG is None:
        _CFG = _load_config()
    return _CFG


# ── ELA (indices 0-5) ─────────────────────────────────────────────────────────

def compute_ela(image_path: str, quality: int = 95) -> Image.Image:
    """Re-save at quality level and return pixel-difference image."""
    original = Image.open(image_path).convert("RGB")
    buf = io.BytesIO()
    original.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    resaved = Image.open(buf).convert("RGB")
    ela = ImageChops.difference(original, resaved)
    ela_arr = np.array(ela)
    ela_scaled = (ela_arr * 10).clip(0, 255).astype(np.uint8)
    return Image.fromarray(ela_scaled)


def _ela_features(image_path: str) -> np.ndarray:
    """
    Returns 6-dim multi-scale ELA vector:
      [mean_q75, std_q75, mean_q85, std_q85, mean_q95, std_q95]
    Sentinel: [-1, -1, -1, -1, -1, -1] on failure.
    """
    quality_levels = _cfg()["forensic_features"].get("ela_quality_levels", [75, 85, 95])
    out = []
    try:
        original = Image.open(image_path).convert("RGB")
        for q in quality_levels:
            buf = io.BytesIO()
            original.save(buf, format="JPEG", quality=q)
            buf.seek(0)
            resaved = Image.open(buf).convert("RGB")
            ela_arr = (np.array(ImageChops.difference(original, resaved)) * 10).clip(0, 255)
            out.extend([ela_arr.mean(), ela_arr.std()])
        return np.array(out, dtype=np.float32)
    except Exception:
        return np.full(len(quality_levels) * 2, -1.0, dtype=np.float32)


# ── DCT (indices 6-15) ────────────────────────────────────────────────────────

def extract_dct_coefficients(image_path: str) -> np.ndarray:
    """Extract DCT AC coefficients from 8x8 blocks. Returns (N, 8, 8) array."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (256, 256))
    coeffs = []
    for i in range(0, 256, 8):
        for j in range(0, 256, 8):
            block = img[i:i + 8, j:j + 8].astype(float)
            block_dct = dct(dct(block.T, norm="ortho").T, norm="ortho")
            coeffs.append(block_dct)
    return np.array(coeffs)


def _dct_features(image_path: str) -> np.ndarray:
    """Returns 10-bin AC coefficient histogram. Sentinel: zeros on failure."""
    try:
        coeffs = extract_dct_coefficients(image_path)
        # AC coefficients: flatten all except DC (0,0)
        ac = coeffs[:, 1:, :].ravel()
        ac = np.abs(ac)
        hist, _ = np.histogram(ac, bins=10, range=(0, 200), density=True)
        return hist.astype(np.float32)
    except Exception:
        return np.zeros(10, dtype=np.float32)


# ── LBP (indices 16-93) ───────────────────────────────────────────────────────

def compute_lbp_histogram(image_path: str, radius: int = 3,
                          n_points: int = 24) -> np.ndarray:
    """Returns (n_points + 2)-bin uniform LBP histogram for the given scale."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (224, 224))
    lbp = local_binary_pattern(img, n_points, radius, method="uniform")
    n_bins = n_points + 2
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins,
                           range=(0, n_bins), density=True)
    return hist.astype(np.float32)


def _lbp_features(image_path: str) -> np.ndarray:
    """
    Returns 78-dim multi-scale LBP vector (concatenated histograms):
      r=1 n_points=8  → 10 bins  [indices 16:26]
      r=3 n_points=24 → 26 bins  [indices 26:52]
      r=5 n_points=40 → 42 bins  [indices 52:94]
    Sentinel: zeros on failure.
    """
    scales = _cfg()["forensic_features"].get(
        "lbp_scales",
        [{"radius": 1, "n_points": 8},
         {"radius": 3, "n_points": 24},
         {"radius": 5, "n_points": 40}],
    )
    total_bins = sum(s["n_points"] + 2 for s in scales)
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (224, 224))
        parts = []
        for s in scales:
            r, n = s["radius"], s["n_points"]
            lbp = local_binary_pattern(img, n, r, method="uniform")
            n_bins = n + 2
            hist, _ = np.histogram(lbp.ravel(), bins=n_bins,
                                   range=(0, n_bins), density=True)
            parts.append(hist.astype(np.float32))
        return np.concatenate(parts)
    except Exception:
        return np.zeros(total_bins, dtype=np.float32)


# ── Noise Residual (indices 94-95) ────────────────────────────────────────────

def extract_noise_residual(image_path: str) -> np.ndarray:
    """Wavelet noise residual (detail coefficients minus approximation)."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE).astype(float)
    img = cv2.resize(img, (256, 256))
    cA, (cH, cV, cD) = pywt.dwt2(img, "db8")
    denoised = pywt.idwt2((np.zeros_like(cA), (cH, cV, cD)), "db8")
    denoised_resized = cv2.resize(denoised, (256, 256))
    residual = img - denoised_resized
    return residual


def _noise_features(image_path: str) -> np.ndarray:
    """Returns [residual_mean, residual_std]. Sentinel: [-1, -1] on failure."""
    try:
        residual = extract_noise_residual(image_path)
        return np.array([residual.mean(), residual.std()], dtype=np.float32)
    except Exception:
        return np.array([-1.0, -1.0], dtype=np.float32)


# ── LSB Entropy (indices 96-98) ───────────────────────────────────────────────

def extract_lsb_plane(image_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns LSB planes for R, G, B channels."""
    img = np.array(Image.open(image_path).convert("RGB"))
    return img[:, :, 0] & 1, img[:, :, 1] & 1, img[:, :, 2] & 1


def lsb_entropy(lsb_plane: np.ndarray) -> float:
    """Binary entropy of a LSB plane."""
    p = float(np.mean(lsb_plane))
    if p == 0 or p == 1:
        return 0.0
    return float(-p * np.log2(p) - (1 - p) * np.log2(1 - p))


def _lsb_features(image_path: str) -> np.ndarray:
    """Returns [entropy_R, entropy_G, entropy_B]. Sentinel: [-1,-1,-1] on failure."""
    try:
        r, g, b = extract_lsb_plane(image_path)
        return np.array([lsb_entropy(r), lsb_entropy(g), lsb_entropy(b)],
                        dtype=np.float32)
    except Exception:
        return np.array([-1.0, -1.0, -1.0], dtype=np.float32)


# ── EXIF (indices 99-100) ─────────────────────────────────────────────────────

_EXIF_FIELDS = [
    "MakerNote", "ThumbnailImage", "SubSecTimeOriginal",
    "LensModel", "ExposureTime", "FNumber",
]


def _exif_features(image_path: str) -> np.ndarray:
    """
    Returns [makernote_present (0/1), completeness_score (0-1)].
    Sentinel: [0, 0] on failure.
    """
    try:
        with open(image_path, "rb") as f:
            tags = exifread.process_file(f, details=True, stop_tag="UNDEF")
        fields_to_check = _cfg()["forensic_features"].get(
            "exif_fields_checked", _EXIF_FIELDS
        )
        makernote = float(any("MakerNote" in k for k in tags))
        present = sum(
            1 for field in fields_to_check
            if any(field in k for k in tags)
        )
        completeness = present / len(fields_to_check) if fields_to_check else 0.0
        return np.array([makernote, completeness], dtype=np.float32)
    except Exception:
        return np.array([0.0, 0.0], dtype=np.float32)


# ── Eye Highlight Consistency (index 101) ────────────────────────────────────

def extract_eye_regions(image_path: str):
    """Detect eye regions with Haar cascade. Returns (eyes, img)."""
    img = cv2.imread(image_path)
    if img is None:
        return np.array([]), None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cascade_path = cv2.data.haarcascades + "haarcascade_eye.xml"
    eye_cascade = cv2.CascadeClassifier(cascade_path)
    eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    return eyes, img


def _eye_consistency_score(img: np.ndarray, eyes: np.ndarray) -> float:
    """
    Normalized cross-correlation between highlight masks of two eye regions.
    Returns score in [0, 1] or -1 if fewer than 2 eyes detected.
    """
    if len(eyes) < 2:
        return -1.0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    regions = []
    for (ex, ey, ew, eh) in eyes[:2]:
        region = gray[ey:ey + eh, ex:ex + ew]
        region = cv2.resize(region, (32, 32))
        regions.append(region)
    # Highlight mask: top 10% brightest pixels
    masks = []
    for r in regions:
        thresh = np.percentile(r, 90)
        masks.append((r >= thresh).astype(float))
    # Normalized cross-correlation
    a, b = masks[0].ravel(), masks[1].ravel()
    a_norm = a - a.mean()
    b_norm = b - b.mean()
    denom = (np.linalg.norm(a_norm) * np.linalg.norm(b_norm))
    if denom < 1e-8:
        return -1.0
    return float(np.dot(a_norm, b_norm) / denom)


def _eye_features(image_path: str) -> np.ndarray:
    """Returns [eye_consistency_score]. Sentinel: -1.0 if no face/eyes detected."""
    try:
        eyes, img = extract_eye_regions(image_path)
        if img is None or len(eyes) < 2:
            return np.array([-1.0], dtype=np.float32)
        score = _eye_consistency_score(img, eyes)
        return np.array([score], dtype=np.float32)
    except Exception:
        return np.array([-1.0], dtype=np.float32)


# ── Validation ────────────────────────────────────────────────────────────────

def validate_feature_vector(vec: np.ndarray) -> None:
    """Assert vector is 102-dim and contains no NaN or Inf."""
    assert vec.shape == (102,), f"Expected shape (102,), got {vec.shape}"
    assert not np.any(np.isnan(vec)), "Feature vector contains NaN"
    assert not np.any(np.isinf(vec)), "Feature vector contains Inf"


# ── Public API ────────────────────────────────────────────────────────────────

def extract_forensic_features(image_path: str) -> np.ndarray:
    """
    Returns a fixed 102-dimensional forensic feature vector for an image.

    Never raises on a valid image file. Failed sub-features return sentinel
    values as documented in module docstring.

    Args:
        image_path: Path to any RGB image file (JPEG, PNG, etc.)

    Returns:
        np.ndarray of shape (102,) and dtype float32.
    """
    ela = _ela_features(image_path)           # [0:6]
    dct_h = _dct_features(image_path)         # [6:16]
    lbp = _lbp_features(image_path)           # [16:94]
    noise = _noise_features(image_path)       # [94:96]
    lsb = _lsb_features(image_path)           # [96:99]
    exif = _exif_features(image_path)         # [99:101]
    eye = _eye_features(image_path)           # [101]

    vec = np.concatenate([ela, dct_h, lbp, noise, lsb, exif, eye]).astype(np.float32)
    validate_feature_vector(vec)
    return vec
