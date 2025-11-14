"""Feature extraction based on segmented fryums."""

from __future__ import annotations

from typing import Dict

import cv2
import numpy as np

from .background_segmentation import SegmentationResult

FEATURE_NAMES = (
    "area_ratio",
    "bbox_ratio",
    "solidity",
    "elongation",
    "perimeter_ratio",
    "roughness",
    "symmetry",
    "lightness_mean",
    "lightness_std",
    "a_mean",
    "a_std",
    "b_mean",
    "b_std",
    "dark_fraction",
    "bright_fraction",
    "yellow_fraction",
    "red_fraction",
    "laplacian_std",
)


def extract_features(
    image: np.ndarray,
    segmentation: SegmentationResult,
    *,
    dark_threshold: int = 170,
    bright_threshold: int = 210,
    yellow_threshold: int = 150,
    red_threshold: int = 150,
    laplacian_ksize: int = 1,
) -> Dict[str, float]:
    """Computes shape, colour and texture descriptors."""

    mask = segmentation.mask
    if mask.sum() == 0:
        raise ValueError("Segmentation mask is empty.")

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    hull_perimeter = cv2.arcLength(hull, True)
    h, w = mask.shape
    x0, y0, x1, y1 = segmentation.bbox
    bbox_area = max(1, (x1 - x0) * (y1 - y0))
    cropped_mask = segmentation.cropped_mask
    symmetry = _symmetry_score(cropped_mask)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    masked_pixels = lab[mask == 255]
    L = masked_pixels[:, 0]
    a = masked_pixels[:, 1]
    b = masked_pixels[:, 2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F, ksize=laplacian_ksize)
    laplacian_std = float(np.std(laplacian[mask == 255]))

    features = {
        "area_ratio": float(area / (h * w)),
        "bbox_ratio": float(area / bbox_area),
        "solidity": float(area / hull_area) if hull_area else 0.0,
        "elongation": float((x1 - x0) / ((y1 - y0) + 1e-6)),
        "perimeter_ratio": float(perimeter / (2 * (h + w))),
        "roughness": float(hull_perimeter / (perimeter + 1e-6)),
        "symmetry": float(symmetry),
        "lightness_mean": float(L.mean()),
        "lightness_std": float(L.std()),
        "a_mean": float(a.mean()),
        "a_std": float(a.std()),
        "b_mean": float(b.mean()),
        "b_std": float(b.std()),
        "dark_fraction": float((L < dark_threshold).mean()),
        "bright_fraction": float((L > bright_threshold).mean()),
        "yellow_fraction": float((b > yellow_threshold).mean()),
        "red_fraction": float((a > red_threshold).mean()),
        "laplacian_std": laplacian_std,
    }
    return features


def _symmetry_score(mask: np.ndarray) -> float:
    """Returns a left/right symmetry score between 0 and 1."""

    h, w = mask.shape
    if w == 0 or h == 0:
        return 0.0
    mid = w // 2
    left = mask[:, :mid]
    right = mask[:, mid:]
    right = np.fliplr(right)
    min_w = min(left.shape[1], right.shape[1])
    if min_w == 0:
        return 0.0
    diff = np.abs(left[:, :min_w] - right[:, :min_w]).sum() / 255.0
    content = mask.sum() / 255.0
    if content == 0:
        return 0.0
    score = 1.0 - diff / content
    return max(0.0, min(1.0, score))
