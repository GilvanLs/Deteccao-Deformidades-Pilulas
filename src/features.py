"""
Extracao de caracteristicas geometricas, morfologicas, de cor (HSV) e de bordas (Canny)
para identificacao de defeitos em comprimidos.
"""

from __future__ import annotations

from dataclasses import dataclass
import cv2
import numpy as np


@dataclass
class PillFeatures:
    area: float
    perimeter: float
    circularity: float
    solidity: float
    aspect_ratio: float
    morph_diff_ratio: float
    stain_ratio: float
    internal_edge_density: float
    mean_s: float
    std_s: float
    mean_v: float
    std_v: float


def keep_largest_component(mask: np.ndarray) -> np.ndarray:
    """Mantem apenas o maior componente conectado da mascara binaria."""
    binary = (mask > 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if num_labels <= 1:
        return (binary * 255).astype(np.uint8)

    # Ignora o label 0 (fundo)
    largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    cleaned = np.where(labels == largest_label, 255, 0).astype(np.uint8)
    return cleaned


def extract_pill_features(
    hsv_image: np.ndarray,
    pill_mask: np.ndarray,
    edges: np.ndarray,
    erode_fn,
    closing_fn,
) -> PillFeatures:
    """
    Extrai metricas geometricas, morfologicas, colorimetricas e de textura/bordas
    a partir da imagem HSV, mascara segmentada e mapa de bordas de Canny.
    """
    contours, _ = cv2.findContours(pill_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return PillFeatures(0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0)

    contour = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(contour))
    perimeter = float(cv2.arcLength(contour, True))
    circularity = (4.0 * np.pi * area / (perimeter**2)) if perimeter > 1e-6 else 0.0

    hull = cv2.convexHull(contour)
    hull_area = float(cv2.contourArea(hull))
    solidity = (area / hull_area) if hull_area > 1e-6 else 0.0

    (_, _), (rw, rh), _ = cv2.minAreaRect(contour)
    min_side = max(min(rw, rh), 1e-6)
    aspect_ratio = max(rw, rh) / min_side

    # Diferenca morfologica (fechamento amplo - mascara original) destaca quebras/concavidades
    closed_wide = closing_fn(pill_mask, ksize=15)
    diff_pixels = np.count_nonzero((closed_wide > 0) & (pill_mask == 0))
    morph_diff_ratio = diff_pixels / max(area, 1.0)

    # Mascara interna (erodida) para ignorar a borda externa do comprimido
    interior_mask = erode_fn(pill_mask, ksize=13) > 0
    if not np.any(interior_mask):
        interior_mask = pill_mask > 0

    interior_pixels = max(int(np.count_nonzero(interior_mask)), 1)

    # Caracteristicas de cor no espaco HSV
    s_vals = hsv_image[..., 1][interior_mask].astype(np.float64)
    v_vals = hsv_image[..., 2][interior_mask].astype(np.float64)
    median_v = float(np.median(v_vals))
    median_s = float(np.median(s_vals))

    # Pixels com saturacao anomala ou variacao brusca de brilho em relacao ao comprimido
    stain_pixels = np.count_nonzero(
        (np.abs(s_vals - median_s) > 30.0) | (s_vals > 50.0) | (np.abs(v_vals - median_v) > 35.0)
    )
    stain_ratio = stain_pixels / interior_pixels

    # Densidade de bordas internas (rachaduras/fissuras)
    internal_edges = np.count_nonzero((edges > 0) & interior_mask)
    internal_edge_density = internal_edges / interior_pixels

    return PillFeatures(
        area=area,
        perimeter=perimeter,
        circularity=circularity,
        solidity=solidity,
        aspect_ratio=aspect_ratio,
        morph_diff_ratio=morph_diff_ratio,
        stain_ratio=stain_ratio,
        internal_edge_density=internal_edge_density,
        mean_s=float(np.mean(s_vals)),
        std_s=float(np.std(s_vals)),
        mean_v=float(np.mean(v_vals)),
        std_v=float(np.std(v_vals)),
    )
