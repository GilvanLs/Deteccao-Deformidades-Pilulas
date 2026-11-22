"""
Ponto de entrada CLI do sistema de inspecao visual de comprimidos.

Exemplos de uso:
    python main.py
    python main.py --mode scratch
    python main.py --mode opencv --data-dir data/synthetic
    python main.py --image data/synthetic/rachadura/rachadura_01.png --mode scratch
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.dataset_generator import generate_synthetic_dataset
from src.metrics import EvaluationSummary, evaluate_dataset, save_metrics_chart
from src.pipeline import run_pipeline, save_visual_report


def _print_summary(s: EvaluationSummary) -> None:
    print(f"\n=== RESULTADOS ({s.backend.upper()}) ===")
    print(f"Total de imagens : {s.total_images}")
    print(f"Matriz Confusao  : TP={s.tp}  FP={s.fp}  TN={s.tn}  FN={s.fn}")
    print(f"Acuracia         : {s.accuracy * 100:.2f}%")
    print(f"Precisao         : {s.precision * 100:.2f}%")
    print(f"Recall           : {s.recall * 100:.2f}%")
    print(f"F1-Score         : {s.f1_score * 100:.2f}%")
    print(f"Tempo medio/img  : {s.avg_time_ms:.2f} ms")
    print("Taxa por categoria:")
    for cat, stats in s.per_category.items():
        print(f"  - {cat:<14}: {int(stats['acertos'])}/{int(stats['total'])} ({stats['taxa'] * 100:.1f}%)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sistema de Inspecao Visual de Comprimidos por PDI"
    )
    parser.add_argument(
        "--mode",
        choices=["scratch", "opencv", "compare"],
        default="compare",
        help="Backend de processamento: 'scratch' (do zero com NumPy), 'opencv', ou 'compare' (ambos).",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/synthetic",
        help="Diretorio contendo subpastas por classe (ex.: aprovado/, rachadura/, mancha/, ...).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Diretorio onde serao salvos os relatorios visuais e graficos de metricas.",
    )
    parser.add_argument(
        "--generate-dataset",
        action="store_true",
        help="Forca a regeneracao da base sintetica em --data-dir.",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Caminho para inspecionar uma unica imagem e gerar seu relatorio visual.",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Modo imagem unica
    if args.image:
        backends = ["scratch", "opencv"] if args.mode == "compare" else [args.mode]
        for b in backends:
            res = run_pipeline(args.image, backend=b)
            report_path = out_dir / f"inspecao_{b}_{Path(args.image).stem}.png"
            save_visual_report(res, report_path)
            print(
                f"[{b.upper()}] {res.classification.status} | "
                f"Defeitos: {res.classification.defects or ['nenhum']} | "
                f"Tempo: {res.elapsed_ms:.1f} ms | Relatorio: {report_path}"
            )
        return

    # Modo avaliacao em lote
    data_dir = Path(args.data_dir)
    if args.generate_dataset or not data_dir.exists():
        print(f"Gerando base sintetica em '{data_dir}'...")
        generate_synthetic_dataset(data_dir)

    backends = ["scratch", "opencv"] if args.mode == "compare" else [args.mode]
    summaries: list[EvaluationSummary] = []

    for b in backends:
        summary = evaluate_dataset(
            data_dir=data_dir,
            backend=b,
            reports_dir=out_dir / "etapas",
            save_one_per_category=True,
        )
        _print_summary(summary)
        summaries.append(summary)

    chart_path = save_metrics_chart(summaries, out_dir / "metricas_comparacao.png")
    print(f"\nGrafico de metricas salvo em: {chart_path}")
    print(f"Imagens intermediarias salvas em: {out_dir / 'etapas'}")


if __name__ == "__main__":
    main()
