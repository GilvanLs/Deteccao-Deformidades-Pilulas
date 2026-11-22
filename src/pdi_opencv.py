"""
Implementacoes equivalentes utilizando funcoes otimizadas do OpenCV
para comparacao de desempenho e resultados contra a solucao do zero.
"""

from __future__ import annotations

import cv2
import numpy as np


def bgr_to_rgb_cv(image_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def rgb_to_gray_cv(image_rgb: np.ndarray) -> np.ndarray:
    if image_rgb.ndim == 2:
        return image_rgb.copy()
    return cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)


def rgb_to_hsv_cv(image_rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)


def gaussian_blur_cv(image: np.ndarray, ksize: int = 5, sigma: float = 1.2) -> np.ndarray:
    return cv2.GaussianBlur(image, (ksize, ksize), sigmaX=sigma, sigmaY=sigma)


def median_filter_cv(image: np.ndarray, ksize: int = 3) -> np.ndarray:
    return cv2.medianBlur(image, ksize)


def otsu_threshold_cv(gray: np.ndarray) -> tuple[int, np.ndarray]:
    thresh, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return int(thresh), binary


def canny_cv(gray: np.ndarray, low_thresh: int = 50, high_thresh: int = 150) -> np.ndarray:
    return cv2.Canny(gray, low_thresh, high_thresh)


def erode_cv(mask: np.ndarray, ksize: int = 5) -> np.ndarray:
    kernel = np.ones((ksize, ksize), dtype=np.uint8)
    return cv2.erode(mask, kernel, iterations=1)


def dilate_cv(mask: np.ndarray, ksize: int = 5) -> np.ndarray:
    kernel = np.ones((ksize, ksize), dtype=np.uint8)
    return cv2.dilate(mask, kernel, iterations=1)


def opening_cv(mask: np.ndarray, ksize: int = 5) -> np.ndarray:
    kernel = np.ones((ksize, ksize), dtype=np.uint8)
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)


def closing_cv(mask: np.ndarray, ksize: int = 5) -> np.ndarray:
    kernel = np.ones((ksize, ksize), dtype=np.uint8)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


def blackhat_cv(gray: np.ndarray, ksize: int = 11) -> np.ndarray:
    kernel = np.ones((ksize, ksize), dtype=np.uint8)
    return cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
