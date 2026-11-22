"""
Modulo de avaliacao quantitativa do sistema de inspecao:
- Acuracia, Precisao, Recall, F1-Score
- Matriz de confusao
- Taxa de acerto por categoria de defeito
- Tempo medio de execucao por imagem
- Graficos comparativos (Matplotlib)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.pipeline import (
    PipelineOutput,
    calibrate_reference_model,
    run_pipeline,
    save_visual_report,
)

VALID_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
NORMAL_FOLDERS = {"aprovado", "good", "normal", "ok"}


@dataclass
class EvaluationSummary:
    backend: str
    total_images: int
    tp: int
    fp: int
    tn: int
    fn: int
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    avg_time_ms: float
    per_category: dict[str, dict[str, float]]


def collect_dataset_samples(data_dir: str | Path) -> list[tuple[Path, str, str]]:
    """
    Percorre subpastas de `data_dir` e retorna lista de (caminho_imagem, categoria, rotulo_binario).
    Pastas em NORMAL_FOLDERS recebem rotulo 'APROVADO'; demais recebem 'REPROVADO'.
    """
    root = Path(data_dir)
    if not root.exists():
        raise FileNotFoundError(f"Diretorio de dados nao encontrado: {root}")

    samples: list[tuple[Path, str, str]] = []
    for subdir in sorted(p for p in root.iterdir() if p.is_dir()):
        category = subdir.name.lower()
        binary_label = "APROVADO" if category in NORMAL_FOLDERS else "REPROVADO"
        for img_path in sorted(subdir.iterdir()):
            if img_path.is_file() and img_path.suffix.lower() in VALID_EXTENSIONS:
                samples.append((img_path, category, binary_label))

    if not samples:
        raise ValueError(f"Nenhuma imagem encontrada nas subpastas de {root}")
    return samples


def evaluate_dataset(
    data_dir: str | Path,
    backend: str = "scratch",
    reports_dir: str | Path | None = None,
    save_one_per_category: bool = True,
    train_good_dir: str | Path | None = None,
) -> EvaluationSummary:
    """Executa o pipeline em todo o dataset e calcula as metricas de classificacao."""
    samples = collect_dataset_samples(data_dir)
    ref_model = None
    if train_good_dir is not None and Path(train_good_dir).exists():
        ref_model = calibrate_reference_model(train_good_dir, backend=backend)

    tp = fp = tn = fn = 0
    times_ms: list[float] = []
    cat_stats: dict[str, list[int]] = {}
    saved_cats: set[str] = set()

    for img_path, category, true_label in samples:
        output: PipelineOutput = run_pipeline(img_path, backend=backend, reference_model=ref_model)
        pred_label = output.classification.status
        times_ms.append(output.elapsed_ms)

        if true_label == "REPROVADO" and pred_label == "REPROVADO":
            tp += 1
        elif true_label == "APROVADO" and pred_label == "REPROVADO":
            fp += 1
        elif true_label == "APROVADO" and pred_label == "APROVADO":
            tn += 1
        else:
            fn += 1

        if category not in cat_stats:
            cat_stats[category] = [0, 0]
        cat_stats[category][1] += 1
        if pred_label == true_label:
            cat_stats[category][0] += 1

        if reports_dir is not None and save_one_per_category and category not in saved_cats:
            out_file = Path(reports_dir) / f"etapas_{backend}_{category}.png"
            save_visual_report(output, out_file)
            saved_cats.add(category)

    total = len(samples)
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    avg_time = float(np.mean(times_ms)) if times_ms else 0.0

    per_category = {
        cat: {
            "acertos": vals[0],
            "total": vals[1],
            "taxa": vals[0] / vals[1] if vals[1] else 0.0,
        }
        for cat, vals in cat_stats.items()
    }

    return EvaluationSummary(
        backend=backend,
        total_images=total,
        tp=tp,
        fp=fp,
        tn=tn,
        fn=fn,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1_score=f1,
        avg_time_ms=avg_time,
        per_category=per_category,
    )


def save_metrics_chart(
    summaries: list[EvaluationSummary],
    save_path: str | Path,
    title_prefix: str = "",
) -> Path:
    """Gera imagem com matriz de confusao e comparativo de metricas."""
    n = len(summaries)
    fig, axes = plt.subplots(1, n + 1, figsize=(5.5 * (n + 1), 4.8))
    if title_prefix:
        fig.suptitle(title_prefix, fontsize=13, fontweight="bold")
    if n == 1:
        axes = [axes[0], axes[1]]

    for idx, s in enumerate(summaries):
        cm = np.array([[s.tn, s.fp], [s.fn, s.tp]])
        ax = axes[idx]
        im = ax.imshow(cm, cmap="Blues")
        ax.set_title(f"Matriz de Confusao ({s.backend.upper()})")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["APROVADO", "REPROVADO"])
        ax.set_yticklabels(["APROVADO", "REPROVADO"])
        ax.set_xlabel("Predito")
        ax.set_ylabel("Real")

        for i in range(2):
            for j in range(2):
                color = "white" if cm[i, j] > cm.max() / 2 else "black"
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=color, fontsize=13, fontweight="bold")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax_bar = axes[-1]
    metric_names = ["Acuracia", "Precisao", "Recall", "F1-Score"]
    x = np.arange(len(metric_names))
    width = 0.35 if n > 1 else 0.5

    for idx, s in enumerate(summaries):
        vals = [s.accuracy * 100, s.precision * 100, s.recall * 100, s.f1_score * 100]
        offset = (idx - (n - 1) / 2) * width
        bars = ax_bar.bar(x + offset, vals, width, label=f"{s.backend.upper()} ({s.avg_time_ms:.1f} ms/img)")
        for bar in bars:
            h = bar.get_height()
            ax_bar.annotate(
                f"{h:.1f}%",
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                fontsize=8.5,
            )

    ax_bar.set_ylim(0, 115)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(metric_names)
    ax_bar.set_ylabel("Percentual (%)")
    ax_bar.set_title("Metricas de Desempenho")
    ax_bar.legend(loc="lower right", fontsize=8.5)
    ax_bar.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    out_file = Path(save_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_file), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_file
