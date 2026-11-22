"""
Pipeline completo de inspecao visual de comprimidos por Processamento Digital de Imagens.

Etapas:
1. Aquisicao da imagem (BGR -> RGB)
2. Filtragem (Mediana + Gaussiano)
3. Conversao de espaco de cor (RGB -> Cinza e HSV)
4. Segmentacao (Otsu + Morfologia Matematica + Maior Componente)
5. Deteccao de bordas (Canny)
6. Extracao de caracteristicas
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
from src.classifier import ClassificationResult, classify_pill
from src.features import extract_pill_features, keep_largest_component


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


def _draw_annotated_result(
    rgb_original: np.ndarray,
    pill_mask: np.ndarray,
    classification: ClassificationResult,
) -> np.ndarray:
    """Desenha contorno do comprimido e rotulo de status/defeitos sobre a imagem RGB."""
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
        label,
        (12, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        color,
        1,
        cv2.LINE_AA,
    )
    return annotated


def run_pipeline(image_path: str | Path, backend: str = "scratch") -> PipelineOutput:
    """
    Executa o pipeline completo em uma imagem usando o backend escolhido
    ('scratch' para implementacao do zero com NumPy, ou 'opencv').
    """
    if backend not in ("scratch", "opencv"):
        raise ValueError("backend deve ser 'scratch' ou 'opencv'.")

    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise FileNotFoundError(f"Nao foi possivel ler a imagem: {image_path}")

    t0 = time.perf_counter()

    if backend == "scratch":
        rgb = sc_ops.bgr_to_rgb(image_bgr)
        filtered = sc_ops.median_filter_scratch(rgb, ksize=3)
        filtered = sc_ops.gaussian_blur_scratch(filtered, ksize=5, sigma=1.2)
        gray = sc_ops.rgb_to_gray(filtered)
        hsv = sc_ops.rgb_to_hsv_scratch(filtered)
        _, binary = sc_ops.otsu_threshold_scratch(gray)
        opened = sc_ops.opening_scratch(binary, ksize=5)
        closed = sc_ops.closing_scratch(opened, ksize=7)
        pill_mask = keep_largest_component(closed)
        edges = sc_ops.canny_scratch(gray, sigma=1.0, low_ratio=0.08, high_ratio=0.22)
        erode_fn = sc_ops.erode_scratch
        closing_fn = sc_ops.closing_scratch
    else:
        rgb = cv_ops.bgr_to_rgb_cv(image_bgr)
        filtered = cv_ops.median_filter_cv(rgb, ksize=3)
        filtered = cv_ops.gaussian_blur_cv(filtered, ksize=5, sigma=1.2)
        gray = cv_ops.rgb_to_gray_cv(filtered)
        hsv = cv_ops.rgb_to_hsv_cv(filtered)
        _, binary = cv_ops.otsu_threshold_cv(gray)
        opened = cv_ops.opening_cv(binary, ksize=5)
        closed = cv_ops.closing_cv(opened, ksize=7)
        pill_mask = keep_largest_component(closed)
        edges = cv_ops.canny_cv(gray, low_thresh=50, high_thresh=150)
        erode_fn = cv_ops.erode_cv
        closing_fn = cv_ops.closing_cv

    features = extract_pill_features(
        hsv_image=hsv,
        pill_mask=pill_mask,
        edges=edges,
        erode_fn=erode_fn,
        closing_fn=closing_fn,
    )
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
    )


def save_visual_report(output: PipelineOutput, save_path: str | Path) -> Path:
    """Salva figura 2x3 com todas as etapas intermediarias e o resultado final."""
    fig, axes = plt.subplots(2, 3, figsize=(13, 8.5))
    fig.suptitle(
        f"Inspecao Visual ({output.backend.upper()}) - {Path(output.image_path).name} "
        f"| {output.classification.status} ({output.elapsed_ms:.1f} ms)",
        fontsize=12,
        fontweight="bold",
    )

    axes[0, 0].imshow(output.rgb_original)
    axes[0, 0].set_title("1. Imagem Original (RGB)")

    axes[0, 1].imshow(output.rgb_filtered)
    axes[0, 1].set_title("2. Filtragem (Mediana + Gaussiano)")

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
