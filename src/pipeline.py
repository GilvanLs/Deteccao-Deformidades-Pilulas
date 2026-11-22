"""
Pipeline completo de inspecao visual de comprimidos e capsulas por PDI.

Etapas:
1. Aquisicao e padronizacao de resolucao (320x320, BGR -> RGB)
2. Filtragem (Mediana + Gaussiano)
3. Conversao de espaco de cor (RGB -> Cinza e HSV)
4. Segmentacao (Otsu com deteccao automatica de polaridade de fundo + Morfologia)
5. Deteccao de bordas (Canny) e Transformada Morfologica Black-Hat
6. Extracao de caracteristicas (+ calibracao split fit/val em `train/good`)
7. Classificacao (APROVADO / REPROVADO + defeitos)
8. Visualizacao das etapas intermediarias e resultado final
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src import pdi_opencv as cv_ops
from src import pdi_scratch as sc_ops
from src.classifier import ClassificationResult, classify_pill, classify_with_reference
from src.features import (
    PillFeatures,
    ReferenceModel,
    extract_aligned_roi,
    extract_pill_features,
    keep_largest_component,
)

VALID_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


@dataclass
class PipelineOutput:
    image_path: str
    backend: str
    elapsed_ms: float
    classification: ClassificationResult
    rgb_original: np.ndarray
    rgb_filtered: np.ndarray
    hsv: np.ndarray
    pill_mask: np.ndarray
    edges: np.ndarray
    annotated_rgb: np.ndarray
    anomaly_map: np.ndarray | None = None


def _auto_foreground_mask(gray: np.ndarray, threshold: int, binary: np.ndarray) -> np.ndarray:
    """Garante objeto branco (255) e fundo preto (0) tanto em fundo claro quanto escuro."""
    border_pixels = np.concatenate([gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]])
    if float(np.mean(border_pixels)) > float(threshold):
        return (255 - binary).astype(np.uint8)
    return binary


def _preprocess_and_segment(
    image_bgr: np.ndarray,
    backend: str,
    target_size: int = 320,
):
    """Executa etapas 1 a 5 do pipeline e retorna mapas intermediarios e operadores."""
    if image_bgr.shape[0] != target_size or image_bgr.shape[1] != target_size:
        image_bgr = cv2.resize(image_bgr, (target_size, target_size), interpolation=cv2.INTER_AREA)

    if backend == "scratch":
        rgb = sc_ops.bgr_to_rgb(image_bgr)
        filtered = sc_ops.median_filter_scratch(rgb, ksize=3)
        filtered = sc_ops.gaussian_blur_scratch(filtered, ksize=5, sigma=1.0)
        gray = sc_ops.rgb_to_gray(filtered)
        hsv = sc_ops.rgb_to_hsv_scratch(filtered)
        thresh, binary = sc_ops.otsu_threshold_scratch(gray)
        binary = _auto_foreground_mask(gray, thresh, binary)
        opened = sc_ops.opening_scratch(binary, ksize=5)
        closed = sc_ops.closing_scratch(opened, ksize=7)
        pill_mask = keep_largest_component(closed)
        edges = sc_ops.canny_scratch(gray, sigma=1.0, low_ratio=0.08, high_ratio=0.22)
        erode_fn = sc_ops.erode_scratch
        closing_fn = sc_ops.closing_scratch
        blackhat_fn = sc_ops.blackhat_scratch
    else:
        rgb = cv_ops.bgr_to_rgb_cv(image_bgr)
        filtered = cv_ops.median_filter_cv(rgb, ksize=3)
        filtered = cv_ops.gaussian_blur_cv(filtered, ksize=5, sigma=1.0)
        gray = cv_ops.rgb_to_gray_cv(filtered)
        hsv = cv_ops.rgb_to_hsv_cv(filtered)
        thresh, binary = cv_ops.otsu_threshold_cv(gray)
        binary = _auto_foreground_mask(gray, thresh, binary)
        opened = cv_ops.opening_cv(binary, ksize=5)
        closed = cv_ops.closing_cv(opened, ksize=7)
        pill_mask = keep_largest_component(closed)
        edges = cv_ops.canny_cv(gray, low_thresh=40, high_thresh=120)
        erode_fn = cv_ops.erode_cv
        closing_fn = cv_ops.closing_cv
        blackhat_fn = cv_ops.blackhat_cv

    return rgb, filtered, gray, hsv, pill_mask, edges, erode_fn, closing_fn, blackhat_fn


def calibrate_reference_model(
    train_good_dir: str | Path,
    backend: str = "scratch",
    max_samples: int = 30,
) -> ReferenceModel:
    """
    Calibra o modelo de referencia em `train/good` usando divisao fit/validacao
    (2/3 para estimar distribuicao de cor e template de aparencia, 1/3 para calibrar
    limiares sem vazamento de dados).
    """
    good_dir = Path(train_good_dir)
    img_paths = sorted(
        p for p in good_dir.iterdir() if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS
    )[:max_samples]
    if len(img_paths) < 6:
        raise ValueError(f"Sao necessarias pelo menos 6 imagens em {good_dir} para calibracao.")

    split_idx = max(int(len(img_paths) * 2 / 3), 4)
    fit_paths = img_paths[:split_idx]
    val_paths = img_paths[split_idx:]

    rng = np.random.default_rng(42)
    pix_samples: list[np.ndarray] = []
    rois_rgb: list[np.ndarray] = []
    rois_mask: list[np.ndarray] = []

    for img_path in fit_paths:
        image_bgr = cv2.imread(str(img_path))
        if image_bgr is None:
            continue
        _, filtered, _, hsv, pill_mask, _, erode_fn, _, _ = _preprocess_and_segment(
            image_bgr, backend=backend
        )
        interior = erode_fn(pill_mask, ksize=15) > 0
        if not np.any(interior):
            interior = pill_mask > 0
        pix = np.concatenate([filtered[interior], hsv[interior][:, 1:2]], axis=1).astype(np.float64)
        sample_n = min(len(pix), 500)
        idx = rng.choice(len(pix), size=sample_n, replace=False)
        pix_samples.append(pix[idx])

        rgb_roi, mask_roi = extract_aligned_roi(filtered, pill_mask)
        rois_rgb.append(rgb_roi)
        rois_mask.append(mask_roi.astype(np.float32))

    all_pix = np.vstack(pix_samples)
    color_mu = np.mean(all_pix, axis=0)
    color_cov = np.cov(all_pix, rowvar=False) + np.eye(4) * 1e-3
    color_inv_cov = np.linalg.inv(color_cov)

    stack_rgb = np.stack(rois_rgb, axis=0)
    ref_rgb_roi = np.median(stack_rgb, axis=0).astype(np.float32)
    ref_std_roi = np.std(stack_rgb, axis=0).astype(np.float32)
    ref_mask_roi = np.mean(np.stack(rois_mask, axis=0), axis=0).astype(np.float32)

    base_ref = ReferenceModel(
        color_mu=color_mu,
        color_inv_cov=color_inv_cov,
        ref_rgb_roi=ref_rgb_roi,
        ref_std_roi=ref_std_roi,
        ref_mask_roi=ref_mask_roi,
        hi_thresholds={},
        lo_solidity=0.95,
        ar_median=1.0,
        ar_tol=0.1,
    )

    val_feats: list[PillFeatures] = []
    for img_path in val_paths:
        image_bgr = cv2.imread(str(img_path))
        if image_bgr is None:
            continue
        _, filtered, gray, hsv, pill_mask, edges, erode_fn, closing_fn, blackhat_fn = _preprocess_and_segment(
            image_bgr, backend=backend
        )
        f, _ = extract_pill_features(
            rgb_filtered=filtered,
            gray_filtered=gray,
            hsv_image=hsv,
            pill_mask=pill_mask,
            edges=edges,
            erode_fn=erode_fn,
            closing_fn=closing_fn,
            blackhat_fn=blackhat_fn,
            reference_model=base_ref,
        )
        val_feats.append(f)

    hi_thresholds: dict[str, float] = {}
    for key in ("color_maha_p995", "blackhat_p995", "internal_edge_density", "residual_p995", "anomaly_area_ratio"):
        vals = np.array([getattr(f, key) for f in val_feats], dtype=np.float64)
        hi_thresholds[key] = float(np.percentile(vals, 92) * 1.04)

    sol_vals = np.array([f.solidity for f in val_feats], dtype=np.float64)
    lo_solidity = float(np.min(sol_vals) - 0.015)

    ar_vals = np.array([f.aspect_ratio for f in val_feats], dtype=np.float64)
    ar_median = float(np.median(ar_vals))
    ar_tol = float(max(np.max(np.abs(ar_vals - ar_median)) * 1.3, 0.06))

    return ReferenceModel(
        color_mu=color_mu,
        color_inv_cov=color_inv_cov,
        ref_rgb_roi=ref_rgb_roi,
        ref_std_roi=ref_std_roi,
        ref_mask_roi=ref_mask_roi,
        hi_thresholds=hi_thresholds,
        lo_solidity=lo_solidity,
        ar_median=ar_median,
        ar_tol=ar_tol,
    )


def _draw_annotated_result(
    rgb_original: np.ndarray,
    pill_mask: np.ndarray,
    classification: ClassificationResult,
) -> np.ndarray:
    """Desenha contorno do objeto e rotulo de status/defeitos sobre a imagem RGB."""
    annotated = rgb_original.copy()
    contours, _ = cv2.findContours(pill_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    is_ok = classification.status == "APROVADO"
    color = (46, 204, 113) if is_ok else (231, 76, 60)

    if contours:
        cv2.drawContours(annotated, contours, -1, color, 2)

    label = classification.status
    if classification.defects:
        label = f"{label}: {', '.join(classification.defects)}"

    cv2.rectangle(annotated, (6, 6), (annotated.shape[1] - 6, 34), (20, 20, 20), -1)
    cv2.putText(
        annotated,
        label[:42],
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.44,
        color,
        1,
        cv2.LINE_AA,
    )
    return annotated


def run_pipeline(
    image_path: str | Path,
    backend: str = "scratch",
    reference_model: ReferenceModel | None = None,
) -> PipelineOutput:
    """Executa o pipeline completo em uma imagem usando o backend escolhido."""
    if backend not in ("scratch", "opencv"):
        raise ValueError("backend deve ser 'scratch' ou 'opencv'.")

    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise FileNotFoundError(f"Nao foi possivel ler a imagem: {image_path}")

    t0 = time.perf_counter()

    rgb, filtered, gray, hsv, pill_mask, edges, erode_fn, closing_fn, blackhat_fn = _preprocess_and_segment(
        image_bgr, backend=backend
    )
    features, anomaly_map = extract_pill_features(
        rgb_filtered=filtered,
        gray_filtered=gray,
        hsv_image=hsv,
        pill_mask=pill_mask,
        edges=edges,
        erode_fn=erode_fn,
        closing_fn=closing_fn,
        blackhat_fn=blackhat_fn,
        reference_model=reference_model,
    )
    if reference_model is not None:
        classification = classify_with_reference(features, reference_model)
    else:
        classification = classify_pill(features)

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    annotated = _draw_annotated_result(rgb, pill_mask, classification)

    return PipelineOutput(
        image_path=str(image_path),
        backend=backend,
        elapsed_ms=elapsed_ms,
        classification=classification,
        rgb_original=rgb,
        rgb_filtered=filtered,
        hsv=hsv,
        pill_mask=pill_mask,
        edges=edges,
        annotated_rgb=annotated,
        anomaly_map=anomaly_map,
    )


def save_visual_report(output: PipelineOutput, save_path: str | Path) -> Path:
    """Salva figura 2x3 com todas as etapas intermediarias e o resultado final."""
    fig, axes = plt.subplots(2, 3, figsize=(13, 8.5))
    fig.suptitle(
        f"Inspecao Visual ({output.backend.upper()}) - {Path(output.image_path).parent.name}/{Path(output.image_path).name} "
        f"| {output.classification.status} ({output.elapsed_ms:.1f} ms)",
        fontsize=12,
        fontweight="bold",
    )

    axes[0, 0].imshow(output.rgb_original)
    axes[0, 0].set_title("1. Imagem Original (320x320 RGB)")

    axes[0, 1].imshow(output.rgb_filtered)
    axes[0, 1].set_title("2. Filtragem (Mediana + Gaussiano)")

    if output.anomaly_map is not None:
        axes[0, 2].imshow(output.anomaly_map, cmap="inferno", vmin=0, vmax=6)
        axes[0, 2].set_title("3. Mapa de Residuo vs Referencia")
    else:
        axes[0, 2].imshow(output.hsv[..., 1], cmap="magma")
        axes[0, 2].set_title("3. Canal S (HSV - Manchas)")

    axes[1, 0].imshow(output.pill_mask, cmap="gray")
    axes[1, 0].set_title("4. Segmentacao (Otsu + Morfologia)")

    axes[1, 1].imshow(output.edges, cmap="gray")
    axes[1, 1].set_title("5. Deteccao de Bordas (Canny)")

    axes[1, 2].imshow(output.annotated_rgb)
    axes[1, 2].set_title("6. Classificacao Final")

    for ax in axes.ravel():
        ax.axis("off")

    plt.tight_layout()
    out_file = Path(save_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_file), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_file
