"""
precompute_features.py — Pre-compute forensic feature vectors for all images.

Run once before training to avoid re-extracting features every epoch.

Usage:
    python src/precompute_features.py --data_dir data/cifake-224 --output data/processed/features_cache_v2.npz

The output .npz contains:
    paths    : (N,)    str array of absolute image paths
    features : (N,102) float32 feature matrix
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Iterator

import numpy as np
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

# Make src importable when run as a script from the project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.feature_extractor import extract_forensic_features  # noqa: E402

console = Console()


def _iter_images(data_dir: str) -> Iterator[str]:
    """Yield all image paths under data_dir (recursive)."""
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    for root, _, files in os.walk(data_dir):
        for fname in sorted(files):
            if Path(fname).suffix.lower() in valid_exts:
                yield os.path.join(root, fname)


def precompute(data_dir: str, output_path: str) -> None:
    os.makedirs(Path(output_path).parent, exist_ok=True)

    all_paths = list(_iter_images(data_dir))
    if not all_paths:
        console.print(f"[red]No images found under {data_dir}[/red]")
        sys.exit(1)

    console.print(f"[cyan]Found {len(all_paths):,} images in {data_dir}[/cyan]")

    paths_out: list[str] = []
    features_out: list[np.ndarray] = []
    failures: list[tuple[str, str]] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Extracting features...", total=len(all_paths))

        for img_path in all_paths:
            t0 = time.perf_counter()
            try:
                feat = extract_forensic_features(img_path)
                elapsed = time.perf_counter() - t0
                if elapsed > 0.5:
                    console.print(
                        f"[yellow]SLOW ({elapsed:.2f}s): {img_path}[/yellow]"
                    )
                paths_out.append(img_path)
                features_out.append(feat)
            except Exception:
                tb = traceback.format_exc().strip().split("\n")[-1]
                failures.append((img_path, tb))
            finally:
                progress.advance(task)

    # Save
    np.savez_compressed(
        output_path,
        paths=np.array(paths_out, dtype=object),
        features=np.array(features_out, dtype=np.float32),
    )

    # Summary
    table = Table(title="Pre-computation Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Total images", f"{len(all_paths):,}")
    table.add_row("Succeeded", f"{len(paths_out):,}")
    table.add_row("Failed", f"{len(failures):,}")
    table.add_row("Output", output_path)
    console.print(table)

    if failures:
        console.print(f"\n[red]Failures ({len(failures)}):[/red]")
        for path, reason in failures[:20]:
            console.print(f"  {path}: {reason}")
        if len(failures) > 20:
            console.print(f"  ... and {len(failures) - 20} more")


def main():
    parser = argparse.ArgumentParser(description="Pre-compute forensic features.")
    parser.add_argument(
        "--data_dir",
        default=None,
        help="Root data directory to scan recursively. "
             "Defaults to datasets.cifake.path from config.yaml.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output .npz path. Defaults to forensic_features.cache_path in config.yaml.",
    )
    args = parser.parse_args()

    # Read defaults from config
    import yaml
    root = Path(__file__).resolve().parent.parent
    with open(root / "config.yaml") as f:
        cfg = yaml.safe_load(f)

    data_dir = args.data_dir or cfg["datasets"]["cifake"]["path"]
    output_path = args.output or cfg["forensic_features"]["cache_path"]

    precompute(data_dir, output_path)


if __name__ == "__main__":
    main()
