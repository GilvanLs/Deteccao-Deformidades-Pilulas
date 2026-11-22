"""
Extracao de caracteristicas geometricas, morfologicas (incluindo Black-Hat),
colorimetricas (HSV + distancia de Mahalanobis), de bordas (Canny) e de residuo
contra modelo de referencia calibrado em `train/good`.
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
    blackhat_p995: float = 0.0
    color_maha_p995: float = 0.0
    residual_p995: float = 0.0
    anomaly_area_ratio: float = 0.0


@dataclass
class ReferenceModel:
    """Modelo estatistico calibrado por divisao (fit/val) sobre imagens normais (`train/good`)."""
    color_mu: np.ndarray            # Media (4,) de [R, G, B, S] dos pixels normais
    color_inv_cov: np.ndarray       # Inversa da covariancia (4x4) para Mahalanobis
    ref_rgb_roi: np.ndarray         # Mediana dos ROIs alinhados (128x128x3)
    ref_std_roi: np.ndarray         # Desvio padrao por pixel (128x128x3)
    ref_mask_roi: np.ndarray        # Mascara media alinhada (128x128)
    hi_thresholds: dict[str, float] # Limiares superiores calibrados em validacao
    lo_solidity: float              # Limiar inferior de solidez
    ar_median: float                # Mediana da razao de aspecto
    ar_tol: float                   # Tolerancia da razao de aspecto


def keep_largest_component(mask: np.ndarray) -> np.ndarray:
    """Mantem apenas o maior componente conectado da mascara binaria."""
    binary = (mask > 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if num_labels <= 1:
        return (binary * 255).astype(np.uint8)

    largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return np.where(labels == largest_label, 255, 0).astype(np.uint8)


def extract_aligned_roi(
    rgb_filtered: np.ndarray,
    pill_mask: np.ndarray,
    roi_size: int = 128,
) -> tuple[np.ndarray, np.ndarray]:
    """Recorta e redimensiona a ROI do comprimido/capsula para (roi_size x roi_size)."""
    ys, xs = np.where(pill_mask > 0)
    if ys.size == 0:
        return (
            np.zeros((roi_size, roi_size, 3), dtype=np.float32),
            np.zeros((roi_size, roi_size), dtype=np.uint8),
        )

    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1

    rgb_roi = cv2.resize(
        rgb_filtered[y0:y1, x0:x1], (roi_size, roi_size), interpolation=cv2.INTER_AREA
    ).astype(np.float32)
    mask_roi = cv2.resize(
        pill_mask[y0:y1, x0:x1], (roi_size, roi_size), interpolation=cv2.INTER_NEAREST
    )
    return rgb_roi, mask_roi


def compute_anomaly_metrics(
    rgb_roi: np.ndarray,
    mask_roi: np.ndarray,
    ref_model: ReferenceModel,
) -> tuple[float, float, np.ndarray]:
    """Calcula mapa de residuo normalizado contra a aparencia de referencia."""
    kernel = np.ones((7, 7), dtype=np.uint8)
    valid_mask = (
        cv2.erode(((mask_roi > 0) & (ref_model.ref_mask_roi > 127)).astype(np.uint8) * 255, kernel)
        > 0
    )
    if not np.any(valid_mask):
        valid_mask = mask_roi > 0

    diff = np.abs(rgb_roi - ref_model.ref_rgb_roi)
    z_map = np.mean(diff / np.maximum(ref_model.ref_std_roi, 8.0), axis=-1)
    z_map = cv2.GaussianBlur(z_map.astype(np.float32), (3, 3), 0.8)

    valid_vals = z_map[valid_mask]
    if valid_vals.size == 0:
        return 0.0, 0.0, z_map

    residual_p995 = float(np.percentile(valid_vals, 99.5))
    anomaly_area_ratio = float(np.mean(valid_vals > 2.8))
    return residual_p995, anomaly_area_ratio, z_map


def extract_pill_features(
    rgb_filtered: np.ndarray,
    gray_filtered: np.ndarray,
    hsv_image: np.ndarray,
    pill_mask: np.ndarray,
    edges: np.ndarray,
    erode_fn,
    closing_fn,
    blackhat_fn,
    reference_model: ReferenceModel | None = None,
) -> tuple[PillFeatures, np.ndarray | None]:
    """Extrai todas as caracteristicas de PDI para o comprimido/capsula."""
    contours, _ = cv2.findContours(pill_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        empty = PillFeatures(0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        return empty, None

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

    closed_wide = closing_fn(pill_mask, ksize=15)
    diff_pixels = np.count_nonzero((closed_wide > 0) & (pill_mask == 0))
    morph_diff_ratio = diff_pixels / max(area, 1.0)

    interior_mask = erode_fn(pill_mask, ksize=15) > 0
    if not np.any(interior_mask):
        interior_mask = pill_mask > 0
    interior_pixels = max(int(np.count_nonzero(interior_mask)), 1)

    s_vals = hsv_image[..., 1][interior_mask].astype(np.float64)
    v_vals = hsv_image[..., 2][interior_mask].astype(np.float64)
    median_v = float(np.median(v_vals))
    median_s = float(np.median(s_vals))

    stain_pixels = np.count_nonzero(
        (np.abs(s_vals - median_s) > 30.0) | (s_vals > 50.0) | (np.abs(v_vals - median_v) > 35.0)
    )
    stain_ratio = stain_pixels / interior_pixels

    internal_edges = np.count_nonzero((edges > 0) & interior_mask)
    internal_edge_density = internal_edges / interior_pixels

    # Transformada morfologica Black-Hat para fissuras/riscos escuros
    bh = blackhat_fn(gray_filtered, ksize=11)
    blackhat_p995 = float(np.percentile(bh[interior_mask], 99.5))

    color_maha_p995 = 0.0
    residual_p995 = 0.0
    anomaly_area_ratio = 0.0
    anomaly_map = None

    if reference_model is not None:
        # Distancia de Mahalanobis no espaco [R, G, B, S] contra distribuicao normal
        pix_feat = np.concatenate(
            [rgb_filtered[interior_mask], hsv_image[interior_mask][:, 1:2]], axis=1
        ).astype(np.float64)
        centered = pix_feat - reference_model.color_mu
        d2 = np.sum((centered @ reference_model.color_inv_cov) * centered, axis=1)
        color_maha_p995 = float(np.percentile(d2, 99.5))

        rgb_roi, mask_roi = extract_aligned_roi(rgb_filtered, pill_mask)
        residual_p995, anomaly_area_ratio, anomaly_map = compute_anomaly_metrics(
            rgb_roi, mask_roi, reference_model
        )

    features = PillFeatures(
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
        blackhat_p995=blackhat_p995,
        color_maha_p995=color_maha_p995,
        residual_p995=residual_p995,
        anomaly_area_ratio=anomaly_area_ratio,
    )
    return features, anomaly_map
