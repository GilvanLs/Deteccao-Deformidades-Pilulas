"""
Script para download automatico das categorias 'pill' e 'capsule' do dataset
MVTec AD hospedado no Hugging Face (foersben/mvtec-ad).

Baixa as pastas `train/good` (para calibracao de referencia) e `test/*`
(para avaliacao quantitativa e visual) mantendo a estrutura original.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import urllib.parse
import urllib.request

REPO_ID = "foersben/mvtec-ad"
REVISION = "14f50140550afeccd97ad31d4cf68fbf9bb94e37"
API_BASE = f"https://huggingface.co/api/datasets/{REPO_ID}/tree/{REVISION}"
RAW_BASE = f"https://huggingface.co/datasets/{REPO_ID}/resolve/{REVISION}"
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}


def _list_tree(subpath: str) -> list[dict]:
    encoded = urllib.parse.quote(subpath, safe="/")
    url = f"{API_BASE}/{encoded}"
    req = urllib.request.Request(url, headers={"User-Agent": "pill-defect-inspection/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _collect_files_recursive(subpath: str) -> list[str]:
    entries = _list_tree(subpath)
    files: list[str] = []
    for entry in entries:
        entry_type = entry.get("type")
        entry_path = entry.get("path", "")
        if entry_type == "directory":
            files.extend(_collect_files_recursive(entry_path))
        elif entry_type == "file" and Path(entry_path).suffix.lower() in IMAGE_EXTS:
            files.append(entry_path)
    return sorted(files)


def _download_one(remote_path: str, local_path: Path) -> bool:
    if local_path.exists() and local_path.stat().st_size > 0:
        return False
    local_path.parent.mkdir(parents=True, exist_ok=True)
    encoded = urllib.parse.quote(remote_path, safe="/")
    url = f"{RAW_BASE}/{encoded}"
    req = urllib.request.Request(url, headers={"User-Agent": "pill-defect-inspection/1.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = resp.read()
    local_path.write_bytes(data)
    return True


def download_mvtec_category(
    category: str,
    output_root: str | Path = "data/mvtec",
    max_train_good: int = 30,
    max_workers: int = 8,
) -> Path:
    """
    Baixa imagens de `train/good` (limitado a `max_train_good` para calibracao rapida)
    e todas as imagens de `test/*` da categoria (`pill` ou `capsule`).
    """
    out_root = Path(output_root)
    print(f"\n[MVTec AD] Listando arquivos de '{category}' no Hugging Face...")

    train_files = _collect_files_recursive(f"{category}/train/good")
    if max_train_good > 0:
        train_files = train_files[:max_train_good]

    test_files = _collect_files_recursive(f"{category}/test")
    all_files = train_files + test_files
    print(
        f"[MVTec AD] '{category}': {len(train_files)} imagens train/good + "
        f"{len(test_files)} imagens test ({len(all_files)} total)."
    )

    downloaded = 0
    skipped = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_download_one, rpath, out_root / rpath): rpath
            for rpath in all_files
        }
        for future in as_completed(futures):
            was_downloaded = future.result()
            if was_downloaded:
                downloaded += 1
            else:
                skipped += 1

    print(
        f"[MVTec AD] '{category}' concluido: {downloaded} baixadas, {skipped} ja existentes "
        f"em '{out_root / category}'."
    )
    return out_root / category


def download_all(
    categories: tuple[str, ...] = ("pill", "capsule"),
    output_root: str | Path = "data/mvtec",
    max_train_good: int = 30,
) -> Path:
    root = Path(output_root)
    for cat in categories:
        download_mvtec_category(cat, output_root=root, max_train_good=max_train_good)
    return root


if __name__ == "__main__":
    download_all()
