"""
Implementacoes do zero (apenas NumPy) de algoritmos classicos de
Processamento Digital de Imagens (PDI).

Inclui:
- Conversoes de espaco de cor (BGR -> RGB, RGB -> Cinza, RGB -> HSV)
- Convolucao 2D, Filtro Gaussiano e Filtro da Mediana
- Limiarizacao automatica de Otsu
- Gradientes de Sobel e Detector de Bordas de Canny (NMS + Histerese)
- Morfologia Matematica binaria (Erosao, Dilatacao, Abertura, Fechamento)
"""

from __future__ import annotations

from collections import deque
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def bgr_to_rgb(image_bgr: np.ndarray) -> np.ndarray:
    """Converte imagem BGR (padrao OpenCV) para RGB invertendo os canais."""
    return image_bgr[..., ::-1].copy()


def rgb_to_gray(image_rgb: np.ndarray) -> np.ndarray:
    """Converte imagem RGB para escala de cinza usando luminancia ITU-R BT.601."""
    if image_rgb.ndim == 2:
        return image_rgb.copy()
    r = image_rgb[..., 0].astype(np.float64)
    g = image_rgb[..., 1].astype(np.float64)
    b = image_rgb[..., 2].astype(np.float64)
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    return np.clip(np.round(gray), 0, 255).astype(np.uint8)


def rgb_to_hsv_scratch(image_rgb: np.ndarray) -> np.ndarray:
    """
    Converte imagem RGB (uint8) para HSV na escala OpenCV:
    - H (Matiz): [0, 179]
    - S (Saturacao): [0, 255]
    - V (Valor/Brilho): [0, 255]
    """
    rgb = image_rgb.astype(np.float64) / 255.0
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]

    c_max = np.max(rgb, axis=-1)
    c_min = np.min(rgb, axis=-1)
    delta = c_max - c_min

    h = np.zeros_like(c_max)
    non_zero = delta > 1e-12

    mask_r = non_zero & (c_max == r)
    mask_g = non_zero & (c_max == g)
    mask_b = non_zero & (c_max == b)

    h[mask_r] = 60.0 * (((g[mask_r] - b[mask_r]) / delta[mask_r]) % 6.0)
    h[mask_g] = 60.0 * (((b[mask_g] - r[mask_g]) / delta[mask_g]) + 2.0)
    h[mask_b] = 60.0 * (((r[mask_b] - g[mask_b]) / delta[mask_b]) + 4.0)
    h = np.where(h < 0, h + 360.0, h)

    s = np.zeros_like(c_max)
    valid_v = c_max > 1e-12
    s[valid_v] = delta[valid_v] / c_max[valid_v]

    v = c_max

    h_u8 = np.clip(np.round(h / 2.0), 0, 179).astype(np.uint8)
    s_u8 = np.clip(np.round(s * 255.0), 0, 255).astype(np.uint8)
    v_u8 = np.clip(np.round(v * 255.0), 0, 255).astype(np.uint8)
    return np.stack([h_u8, s_u8, v_u8], axis=-1)


def gaussian_kernel(ksize: int = 5, sigma: float = 1.0) -> np.ndarray:
    """Gera um kernel Gaussiano 2D normalizado."""
    if ksize % 2 == 0 or ksize < 1:
        raise ValueError("ksize deve ser um inteiro impar positivo.")
    radius = ksize // 2
    ax = np.arange(-radius, radius + 1, dtype=np.float64)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    kernel /= np.sum(kernel)
    return kernel


def convolve2d_scratch(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Aplica convolucao 2D em imagem 2D (cinza) ou 3D (canais independentes)
    utilizando preenchimento por reflexao (reflect padding) e janelas vetorizadas.
    """
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2

    if image.ndim == 2:
        padded = np.pad(image.astype(np.float64), ((pad_h, pad_h), (pad_w, pad_w)), mode="reflect")
        windows = sliding_window_view(padded, (kh, kw))
        return np.einsum("ijab,ab->ij", windows, kernel)

    channels = [convolve2d_scratch(image[..., c], kernel) for c in range(image.shape[2])]
    return np.stack(channels, axis=-1)


def gaussian_blur_scratch(image: np.ndarray, ksize: int = 5, sigma: float = 1.2) -> np.ndarray:
    """Suavizacao Gaussiana implementada do zero com NumPy."""
    kernel = gaussian_kernel(ksize=ksize, sigma=sigma)
    blurred = convolve2d_scratch(image, kernel)
    return np.clip(np.round(blurred), 0, 255).astype(np.uint8)


def median_filter_scratch(image: np.ndarray, ksize: int = 3) -> np.ndarray:
    """
    Filtro da mediana implementado do zero com NumPy para remocao de ruido impulsivo
    preservando bordas.
    """
    if ksize % 2 == 0 or ksize < 1:
        raise ValueError("ksize deve ser um inteiro impar positivo.")
    pad = ksize // 2

    if image.ndim == 2:
        padded = np.pad(image, ((pad, pad), (pad, pad)), mode="reflect")
        windows = sliding_window_view(padded, (ksize, ksize))
        return np.median(windows, axis=(-2, -1)).astype(np.uint8)

    channels = [median_filter_scratch(image[..., c], ksize=ksize) for c in range(image.shape[2])]
    return np.stack(channels, axis=-1)


def otsu_threshold_scratch(gray: np.ndarray) -> tuple[int, np.ndarray]:
    """
    Calcula o limiar otimo de Otsu maximizando a variancia entre classes
    e retorna (limiar, mascara_binaria_0_255).
    """
    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    total = gray.size
    prob = hist / total

    omega = np.cumsum(prob)
    mu = np.cumsum(prob * np.arange(256))
    mu_total = mu[-1]

    denom = omega * (1.0 - omega)
    valid = denom > 1e-12
    between_var = np.zeros(256, dtype=np.float64)
    between_var[valid] = ((mu_total * omega[valid] - mu[valid]) ** 2) / denom[valid]

    threshold = int(np.argmax(between_var))
    binary = np.where(gray > threshold, 255, 0).astype(np.uint8)
    return threshold, binary


def sobel_gradients_scratch(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Calcula magnitude e direcao (em radianos) do gradiente usando operadores de Sobel."""
    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
    ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float64)

    gx = convolve2d_scratch(gray, kx)
    gy = convolve2d_scratch(gray, ky)

    magnitude = np.hypot(gx, gy)
    angle = np.arctan2(gy, gx)
    return magnitude, angle


def _non_max_suppression(magnitude: np.ndarray, angle: np.ndarray) -> np.ndarray:
    """Supressao nao-maxima vetorizada para afinar bordas na direcao do gradiente."""
    suppressed = np.zeros_like(magnitude, dtype=np.float64)
    deg = np.rad2deg(angle) % 180.0

    center = magnitude[1:-1, 1:-1]
    d = deg[1:-1, 1:-1]

    # 0 graus (horizontal)
    m0 = ((0 <= d) & (d < 22.5)) | ((157.5 <= d) & (d <= 180.0))
    ok0 = m0 & (center >= magnitude[1:-1, 2:]) & (center >= magnitude[1:-1, :-2])

    # 45 graus (diagonal /)
    m45 = (22.5 <= d) & (d < 67.5)
    ok45 = m45 & (center >= magnitude[2:, :-2]) & (center >= magnitude[:-2, 2:])

    # 90 graus (vertical)
    m90 = (67.5 <= d) & (d < 112.5)
    ok90 = m90 & (center >= magnitude[2:, 1:-1]) & (center >= magnitude[:-2, 1:-1])

    # 135 graus (diagonal \)
    m135 = (112.5 <= d) & (d < 157.5)
    ok135 = m135 & (center >= magnitude[:-2, :-2]) & (center >= magnitude[2:, 2:])

    keep = ok0 | ok45 | ok90 | ok135
    suppressed[1:-1, 1:-1] = np.where(keep, center, 0.0)
    return suppressed


def _hysteresis_thresholding(
    nms: np.ndarray, low_ratio: float = 0.08, high_ratio: float = 0.22
) -> np.ndarray:
    """Limiarizacao dupla e rastreamento de bordas por histerese (BFS)."""
    max_val = float(np.max(nms))
    if max_val < 1e-12:
        return np.zeros_like(nms, dtype=np.uint8)

    high_thresh = max_val * high_ratio
    low_thresh = max_val * low_ratio

    strong = nms >= high_thresh
    weak = (nms >= low_thresh) & (~strong)

    edges = np.zeros_like(nms, dtype=np.uint8)
    edges[strong] = 255

    h, w = nms.shape
    queue = deque(zip(*np.where(strong)))
    neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    while queue:
        y, x = queue.popleft()
        for dy, dx in neighbors:
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and weak[ny, nx] and edges[ny, nx] == 0:
                edges[ny, nx] = 255
                queue.append((ny, nx))

    return edges


def canny_scratch(
    gray: np.ndarray,
    sigma: float = 1.0,
    low_ratio: float = 0.08,
    high_ratio: float = 0.22,
) -> np.ndarray:
    """
    Detector de bordas de Canny completo implementado do zero:
    1. Suavizacao Gaussiana
    2. Gradiente de Sobel
    3. Supressao Nao-Maxima (NMS)
    4. Limiarizacao dupla + Histerese
    """
    smooth = gaussian_blur_scratch(gray, ksize=5, sigma=sigma)
    magnitude, angle = sobel_gradients_scratch(smooth)
    nms = _non_max_suppression(magnitude, angle)
    return _hysteresis_thresholding(nms, low_ratio=low_ratio, high_ratio=high_ratio)


def _pad_binary(mask: np.ndarray, ksize: int, pad_val: int) -> np.ndarray:
    pad = ksize // 2
    return np.pad(mask, ((pad, pad), (pad, pad)), mode="constant", constant_values=pad_val)


def erode_scratch(mask: np.ndarray, ksize: int = 5) -> np.ndarray:
    """Erosao morfologica binaria (elemento estruturante quadrado ksize x ksize)."""
    binary = (mask > 0).astype(np.uint8) * 255
    padded = _pad_binary(binary, ksize, pad_val=255)
    windows = sliding_window_view(padded, (ksize, ksize))
    return np.min(windows, axis=(-2, -1)).astype(np.uint8)


def dilate_scratch(mask: np.ndarray, ksize: int = 5) -> np.ndarray:
    """Dilatacao morfologica binaria (elemento estruturante quadrado ksize x ksize)."""
    binary = (mask > 0).astype(np.uint8) * 255
    padded = _pad_binary(binary, ksize, pad_val=0)
    windows = sliding_window_view(padded, (ksize, ksize))
    return np.max(windows, axis=(-2, -1)).astype(np.uint8)


def opening_scratch(mask: np.ndarray, ksize: int = 5) -> np.ndarray:
    """Abertura morfologica (erosao seguida de dilatacao) para remover pequenos ruidos."""
    return dilate_scratch(erode_scratch(mask, ksize=ksize), ksize=ksize)


def closing_scratch(mask: np.ndarray, ksize: int = 5) -> np.ndarray:
    """Fechamento morfologico (dilatacao seguida de erosao) para fechar lacunas."""
    return erode_scratch(dilate_scratch(mask, ksize=ksize), ksize=ksize)
