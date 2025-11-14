"""Background removal and cropping."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np


@dataclass
class SegmentationResult:
    """Container for segmentation products."""

    mask: np.ndarray
    cropped_image: np.ndarray
    cropped_mask: np.ndarray
    bbox: Tuple[int, int, int, int]
    area_ratio: float
    raw_mask: np.ndarray
    blurred: np.ndarray


class BackgroundSegmenter:
    """Removes the greenish background using adaptive thresholding."""

    def __init__(
        self,
        blur_size: int = 5,
        morph_kernel_size: int = 11,
        median_kernel_size: int = 5,
        morph_iterations: int = 1,
        close_then_open: bool = True,
        keep_largest_object: bool = True,
        invert_threshold: int = 200,
    ) -> None:
        self.blur_size = blur_size
        self.median_kernel_size = median_kernel_size
        self.morph_iterations = morph_iterations
        self.close_then_open = close_then_open
        self.keep_largest_object = keep_largest_object
        self.invert_threshold = invert_threshold
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size))

    def segment(self, image: np.ndarray) -> SegmentationResult:
        """Generates a binary mask and returns cropped foreground patches."""

        blurred = cv2.GaussianBlur(image, (self.blur_size, self.blur_size), 0)
        lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)
        lightness = lab[:, :, 0]
        _, mask = cv2.threshold(lightness, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if mask.mean() > self.invert_threshold:
            mask = 255 - mask
        raw_mask = mask.copy()
        mask = cv2.medianBlur(mask, self.median_kernel_size)
        operations = (
            (cv2.MORPH_CLOSE, cv2.MORPH_OPEN)
            if self.close_then_open
            else (cv2.MORPH_OPEN, cv2.MORPH_CLOSE)
        )
        for op in operations:
            mask = cv2.morphologyEx(mask, op, self.kernel, iterations=self.morph_iterations)
        cleaned = self._keep_largest_object(mask) if self.keep_largest_object else mask
        h, w = cleaned.shape
        area_ratio = float(cleaned.sum() / 255) / (h * w)
        y0, y1, x0, x1 = self._bounding_box(cleaned)
        if x1 - x0 <= 1 or y1 - y0 <= 1:
            x0, y0, x1, y1 = 0, 0, w, h
        cropped_mask = cleaned[y0:y1, x0:x1]
        cropped_image = image[y0:y1, x0:x1].copy()
        foreground = cv2.bitwise_and(cropped_image, cropped_image, mask=cropped_mask)
        return SegmentationResult(
            mask=cleaned,
            cropped_image=foreground,
            cropped_mask=cropped_mask,
            bbox=(x0, y0, x1, y1),
            area_ratio=area_ratio,
            raw_mask=raw_mask,
            blurred=blurred,
        )

    @staticmethod
    def _keep_largest_object(mask: np.ndarray) -> np.ndarray:
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return np.zeros_like(mask)
        largest = max(cnts, key=cv2.contourArea)
        cleaned = np.zeros_like(mask)
        cv2.drawContours(cleaned, [largest], -1, 255, -1)
        return cleaned

    @staticmethod
    def _bounding_box(mask: np.ndarray) -> Tuple[int, int, int, int]:
        coords = np.column_stack(np.nonzero(mask))
        if coords.size == 0:
            return (0, 0, mask.shape[1], mask.shape[0])
        ys, xs = coords[:, 0], coords[:, 1]
        return (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)
