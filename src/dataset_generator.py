"""
Gerador de base de dados sintetica e reprodutivel de comprimidos farmaceuticos
para testes controlados de inspecao visual por PDI.

Gera 60 imagens (320x320, PNG) divididas em 5 cenarios:
- aprovado (20 imagens): comprimidos integros com iluminacao e ruido realista
- rachadura (10 imagens): fissuras/quebras na superficie
- mancha (10 imagens): alteracoes de cor e manchas superficiais
- fragmentacao (10 imagens): bordas quebradas ou partes ausentes
- deformacao (10 imagens): formato ou dimensoes fora do padrao
"""

from __future__ import annotations

from pathlib import Path
import cv2
import numpy as np


def _make_background(h: int, w: int, rng: np.random.Generator) -> np.ndarray:
    """Cria fundo escuro texturizado simulando esteira de inspecao."""
    base = np.full((h, w, 3), (28, 30, 34), dtype=np.float64)
    texture = rng.normal(0.0, 3.0, size=(h, w, 1))
    # Gradiente sutil de iluminacao
    yy, xx = np.mgrid[0:h, 0:w]
    vignette = 1.0 - 0.15 * (((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2)
    bg = base * vignette[..., None] + texture
    return np.clip(bg, 0, 255).astype(np.uint8)


def _create_pill_base(
    h: int,
    w: int,
    center: tuple[int, int],
    axes: tuple[int, int],
    angle: float,
    base_color_bgr: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Desenha um comprimido com sombreamento 3D suave e retorna (imagem_bgr, mascara)."""
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(mask, center, axes, angle, 0, 360, 255, -1)

    # Sombreamento radial usando transformada de distancia
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    max_dist = max(float(np.max(dist)), 1.0)
    shading = 0.72 + 0.33 * np.sqrt(dist / max_dist)

    pill = np.zeros((h, w, 3), dtype=np.float64)
    for c in range(3):
        pill[..., c] = base_color_bgr[c] * shading

    return np.clip(pill, 0, 255).astype(np.uint8), mask


def _add_sensor_noise(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Adiciona ruido Gaussiano leve e ruido impulsivo (sal e pimenta)."""
    noisy = image.astype(np.float64) + rng.normal(0.0, 2.8, size=image.shape)
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)

    # Pequena fracao de ruido impulsivo para demonstrar eficacia do filtro da mediana
    h, w, _ = noisy.shape
    num_salt = int(0.0015 * h * w)
    ys = rng.integers(0, h, size=num_salt)
    xs = rng.integers(0, w, size=num_salt)
    noisy[ys, xs] = 245

    ys = rng.integers(0, h, size=num_salt)
    xs = rng.integers(0, w, size=num_salt)
    noisy[ys, xs] = 15
    return noisy


def _apply_crack(pill: np.ndarray, mask: np.ndarray, center: tuple[int, int], rng: np.random.Generator) -> np.ndarray:
    """Desenha uma rachadura irregular na superficie do comprimido."""
    out = pill.copy()
    cx, cy = center
    angle = rng.uniform(0, np.pi)
    length = rng.integers(45, 75)

    pts = []
    num_steps = 12
    for step in range(num_steps):
        t = (step / (num_steps - 1)) - 0.5
        px = int(cx + t * length * np.cos(angle) + rng.integers(-4, 5))
        py = int(cy + t * length * np.sin(angle) + rng.integers(-4, 5))
        pts.append((px, py))

    crack_layer = np.zeros_like(out)
    for i in range(len(pts) - 1):
        cv2.line(crack_layer, pts[i], pts[i + 1], (1, 1, 1), thickness=int(rng.integers(2, 4)))

    crack_mask = (crack_layer[..., 0] > 0) & (mask > 0)
    out[crack_mask] = (out[crack_mask] * 0.25).astype(np.uint8)
    return out


def _apply_stain(pill: np.ndarray, mask: np.ndarray, center: tuple[int, int], rng: np.random.Generator) -> np.ndarray:
    """Aplica manchas e alteracoes de coloracao na superficie do comprimido."""
    out = pill.astype(np.float64)
    h, w = mask.shape
    yy, xx = np.mgrid[0:h, 0:w]

    num_spots = int(rng.integers(2, 4))
    stain_colors = [
        np.array([45, 75, 135], dtype=np.float64),   # mancha amarelada/amarronzada (BGR)
        np.array([55, 55, 65], dtype=np.float64),    # contaminacao escura
        np.array([70, 120, 175], dtype=np.float64),  # oxidacao alaranjada
    ]
    stain_color = stain_colors[int(rng.integers(0, len(stain_colors)))]

    for _ in range(num_spots):
        sx = center[0] + int(rng.integers(-35, 36))
        sy = center[1] + int(rng.integers(-25, 26))
        radius = float(rng.integers(14, 26))
        dist2 = ((xx - sx) ** 2 + (yy - sy) ** 2) / (radius**2)
        alpha = np.exp(-dist2 * 1.4) * 0.78
        alpha = np.where(mask > 0, alpha, 0.0)
        out = out * (1.0 - alpha[..., None]) + stain_color * alpha[..., None]

    return np.clip(out, 0, 255).astype(np.uint8)


def _apply_fragmentation(
    pill: np.ndarray,
    mask: np.ndarray,
    center: tuple[int, int],
    axes: tuple[int, int],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Remove pedacos irregulares da borda simulando fragmentacao/quebra."""
    new_mask = mask.copy()
    theta = rng.uniform(0, 2 * np.pi)
    bx = int(center[0] + axes[0] * 0.88 * np.cos(theta))
    by = int(center[1] + axes[1] * 0.88 * np.sin(theta))

    # Poligono irregular na borda para simular lasca quebrada
    num_verts = 7
    poly = []
    for k in range(num_verts):
        ang = 2 * np.pi * k / num_verts
        rad = rng.integers(18, 32)
        poly.append([int(bx + rad * np.cos(ang)), int(by + rad * np.sin(ang))])
    pts = np.array(poly, dtype=np.int32)
    cv2.fillPoly(new_mask, [pts], 0)

    # Segundo pequeno lascado opcional
    if rng.random() > 0.4:
        theta2 = theta + rng.uniform(0.5, 1.4)
        bx2 = int(center[0] + axes[0] * 0.9 * np.cos(theta2))
        by2 = int(center[1] + axes[1] * 0.9 * np.sin(theta2))
        cv2.circle(new_mask, (bx2, by2), int(rng.integers(12, 20)), 0, -1)

    return pill, new_mask


def generate_synthetic_dataset(output_dir: str | Path, seed: int = 42) -> Path:
    """
    Gera a base sintetica completa em `output_dir` e retorna o Path raiz.
    """
    root = Path(output_dir)
    rng = np.random.default_rng(seed)

    counts = {
        "aprovado": 20,
        "rachadura": 10,
        "mancha": 10,
        "fragmentacao": 10,
        "deformacao": 10,
    }

    h, w = 320, 320
    standard_color = (218, 224, 232)  # comprimido branco/acinzentado claro (BGR)

    for category, total in counts.items():
        cat_dir = root / category
        cat_dir.mkdir(parents=True, exist_ok=True)

        for idx in range(1, total + 1):
            bg = _make_background(h, w, rng)
            cx = w // 2 + int(rng.integers(-10, 11))
            cy = h // 2 + int(rng.integers(-10, 11))
            angle = float(rng.uniform(-15.0, 15.0))

            if category == "deformacao":
                # Formato alongado ou achatado fora do padrao circular
                if rng.random() > 0.5:
                    axes = (int(rng.integers(95, 115)), int(rng.integers(45, 58)))
                else:
                    axes = (int(rng.integers(48, 60)), int(rng.integers(95, 115)))
            else:
                r = int(rng.integers(74, 80))
                axes = (r, r)

            pill, mask = _create_pill_base(h, w, (cx, cy), axes, angle, standard_color)

            if category == "rachadura":
                pill = _apply_crack(pill, mask, (cx, cy), rng)
            elif category == "mancha":
                pill = _apply_stain(pill, mask, (cx, cy), rng)
            elif category == "fragmentacao":
                pill, mask = _apply_fragmentation(pill, mask, (cx, cy), axes, rng)

            # Suaviza levemente a transicao da borda da mascara
            alpha = cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (3, 3), 0.6)[..., None]
            composed = (pill.astype(np.float32) * alpha + bg.astype(np.float32) * (1.0 - alpha))
            final_img = _add_sensor_noise(np.clip(composed, 0, 255).astype(np.uint8), rng)

            out_path = cat_dir / f"{category}_{idx:02d}.png"
            cv2.imwrite(str(out_path), final_img)

    return root
