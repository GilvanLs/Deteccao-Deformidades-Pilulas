"""
Ponto de entrada CLI do sistema de inspecao visual de comprimidos e capsulas
utilizando as bases reais MVTec AD (`pill` e `capsule` - foersben/mvtec-ad).

Exemplos de uso:
    python main.py                                    # Avalia MVTec AD pill + capsule (compare)
    python main.py --dataset pill --mode compare      # Apenas MVTec AD pill
    python main.py --dataset capsule --mode compare   # Apenas MVTec AD capsule
    python main.py --image data/mvtec/pill/test/crack/000.png --train-good data/mvtec/pill/train/good
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.download_mvtec import download_mvtec_category
from src.metrics import EvaluationSummary, evaluate_dataset, save_metrics_chart
from src.pipeline import calibrate_reference_model, run_pipeline, save_visual_report


def _print_summary(dataset_name: str, s: EvaluationSummary) -> None:
    print(f"\n=== RESULTADOS [{dataset_name.upper()}] ({s.backend.upper()}) ===")
    print(f"Total de imagens : {s.total_images}")
    print(f"Matriz Confusao  : TP={s.tp}  FP={s.fp}  TN={s.tn}  FN={s.fn}")
    print(f"Acuracia         : {s.accuracy * 100:.2f}%")
    print(f"Precisao         : {s.precision * 100:.2f}%")
    print(f"Recall           : {s.recall * 100:.2f}%")
    print(f"F1-Score         : {s.f1_score * 100:.2f}%")
    print(f"Tempo medio/img  : {s.avg_time_ms:.2f} ms")
    print("Taxa por categoria:")
    for cat, stats in s.per_category.items():
        print(f"  - {cat:<16}: {int(stats['acertos'])}/{int(stats['total'])} ({stats['taxa'] * 100:.1f}%)")


def _run_dataset_eval(
    name: str,
    test_dir: Path,
    train_good_dir: Path | None,
    backends: list[str],
    out_dir: Path,
) -> None:
    ds_out = out_dir / name
    summaries: list[EvaluationSummary] = []

    for b in backends:
        summary = evaluate_dataset(
            data_dir=test_dir,
            backend=b,
            reports_dir=ds_out / "etapas",
            save_one_per_category=True,
            train_good_dir=train_good_dir,
        )
        _print_summary(name, summary)
        summaries.append(summary)

    chart_path = save_metrics_chart(
        summaries,
        ds_out / "metricas_comparacao.png",
        title_prefix=f"Avaliacao - {name.upper()}",
    )
    print(f"Grafico salvo em : {chart_path}")
    print(f"Etapas salvas em : {ds_out / 'etapas'}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sistema de Inspecao Visual de Comprimidos e Capsulas (MVTec AD)"
    )
    parser.add_argument(
        "--dataset",
        choices=["mvtec-all", "pill", "capsule", "custom"],
        default="mvtec-all",
        help="Base de dados a avaliar (padrao: mvtec-all = MVTec AD pill + capsule).",
    )
    parser.add_argument(
        "--mode",
        choices=["scratch", "opencv", "compare"],
        default="compare",
        help="Backend: 'scratch' (do zero com NumPy), 'opencv', ou 'compare' (ambos).",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Diretorio customizado de teste (quando --dataset custom).",
    )
    parser.add_argument(
        "--train-good",
        type=str,
        default=None,
        help="Diretorio opcional de imagens normais para calibracao (ex.: data/mvtec/pill/train/good).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Diretorio de saida para graficos e paineis visuais.",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Inspeciona uma unica imagem e salva seu relatorio visual.",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    backends = ["scratch", "opencv"] if args.mode == "compare" else [args.mode]

    if args.image:
        for b in backends:
            ref = calibrate_reference_model(args.train_good, backend=b) if args.train_good else None
            res = run_pipeline(args.image, backend=b, reference_model=ref)
            report_path = out_dir / f"inspecao_{b}_{Path(args.image).stem}.png"
            save_visual_report(res, report_path)
            print(
                f"[{b.upper()}] {res.classification.status} | "
                f"Defeitos: {res.classification.defects or ['nenhum']} | "
                f"Tempo: {res.elapsed_ms:.1f} ms | Relatorio: {report_path}"
            )
        return

    if args.dataset in ("mvtec-all", "pill", "capsule"):
        cats = ("pill", "capsule") if args.dataset == "mvtec-all" else (args.dataset,)
        for cat in cats:
            cat_root = Path("data/mvtec") / cat
            if not (cat_root / "test").exists():
                download_mvtec_category(cat, output_root="data/mvtec")
            _run_dataset_eval(
                name=f"mvtec_{cat}",
                test_dir=cat_root / "test",
                train_good_dir=cat_root / "train" / "good",
                backends=backends,
                out_dir=out_dir,
            )
    elif args.dataset == "custom":
        if not args.data_dir:
            raise ValueError("Informe --data-dir ao usar --dataset custom.")
        _run_dataset_eval(
            name="custom",
            test_dir=Path(args.data_dir),
            train_good_dir=Path(args.train_good) if args.train_good else None,
            backends=backends,
            out_dir=out_dir,
        )


if __name__ == "__main__":
    main()
