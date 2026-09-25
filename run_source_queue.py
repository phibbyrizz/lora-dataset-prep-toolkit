#!/usr/bin/env python3
"""Resumable multi-folder launcher for build_lora_dataset.py."""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".avif"
}


@dataclass
class QueueResult:
    folder: str
    status: str
    details: str


def direct_images(folder: Path) -> list[Path]:
    return sorted(
        (
            item
            for item in folder.iterdir()
            if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS
        ),
        key=lambda item: item.name.casefold(),
    )


def is_ignored_folder(folder: Path) -> bool:
    name = folder.name.casefold()
    return (
        name.startswith("_")
        or name.endswith("_metadata")
        or name.endswith(" metadata")
        or name in {"metadata", "reports", "tools"}
    )


def derive_trigger(folder_name: str) -> str:
    name = re.sub(
        r"(?i)(?:[_\-\s]+(?:source|dataset))$", "", folder_name
    ).strip(" _-")
    aliases: list[str] = []
    for part in re.split(r"__+", name):
        alias = re.sub(r"[_]+", " ", part)
        alias = re.sub(r"\s+", " ", alias).strip(" ,")
        if alias:
            aliases.append(alias.lower())
    return ", ".join(aliases)


def discover_source_folders(source_root: Path) -> tuple[list[Path], list[QueueResult]]:
    eligible: list[Path] = []
    ignored: list[QueueResult] = []

    for folder in sorted(source_root.iterdir(), key=lambda item: item.name.casefold()):
        if not folder.is_dir():
            continue
        if is_ignored_folder(folder):
            ignored.append(QueueResult(folder.name, "ignored", "metadata/output/system folder"))
            continue
        image_count = len(direct_images(folder))
        if image_count == 0:
            ignored.append(QueueResult(folder.name, "ignored", "no supported images directly inside folder"))
            continue
        eligible.append(folder)

    return eligible, ignored


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Process every eligible LoRA source folder as a resumable queue."
    )
    parser.add_argument("source_root", type=Path, help="DATASETS/Source folder")
    parser.add_argument(
        "--builder",
        type=Path,
        default=Path(__file__).with_name("build_lora_dataset.py"),
        help="Path to build_lora_dataset.py",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Maximum number of unfinished folders to attempt; 0 means all",
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="Show the discovered queue without processing anything",
    )
    return parser.parse_args()


def write_summary(output_root: Path, results: list[QueueResult]) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = output_root / f"queue_summary_{stamp}.csv"
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["folder", "status", "details"])
        for result in results:
            writer.writerow([result.folder, result.status, result.details])
    return path


def main() -> int:
    args = parse_args()
    source_root = args.source_root.expanduser().resolve()
    builder = args.builder.expanduser().resolve()

    if not source_root.is_dir():
        print(f"ERROR: Source root not found: {source_root}", file=sys.stderr)
        return 2
    if not builder.is_file():
        print(f"ERROR: Builder not found: {builder}", file=sys.stderr)
        return 2
    if args.limit < 0:
        print("ERROR: --limit cannot be negative.", file=sys.stderr)
        return 2

    output_root = source_root / "_pipeline_output"
    folders, ignored = discover_source_folders(source_root)
    results: list[QueueResult] = list(ignored)

    completed: list[Path] = []
    unfinished: list[Path] = []
    blocked: list[Path] = []

    for folder in folders:
        final_zip = output_root / f"{folder.name}_TRAINING_READY.zip"
        final_dir = output_root / f"{folder.name}_dataset"
        building_dir = output_root / f"{folder.name}_BUILDING"

        if final_zip.is_file():
            completed.append(folder)
        elif final_dir.exists() or building_dir.exists():
            blocked.append(folder)
        else:
            unfinished.append(folder)

    selected = unfinished[: args.limit or None]

    print("=" * 68)
    print(" In-House LoRA Dataset Pipeline v0.1.4 — Resumable Queue")
    print("=" * 68)
    print(f"Source root       : {source_root}")
    print(f"Eligible folders  : {len(folders)}")
    print(f"Completed ZIPs    : {len(completed)}")
    print(f"Blocked partials  : {len(blocked)}")
    print(f"Ready to process  : {len(unfinished)}")
    print(f"Selected this run : {len(selected)}")
    print()

    for folder in completed:
        print(f"[SKIP COMPLETE] {folder.name}")
        results.append(QueueResult(folder.name, "skipped_complete", "training ZIP already exists"))

    for folder in blocked:
        print(f"[BLOCKED]       {folder.name} — existing dataset or BUILDING folder; review manually")
        results.append(QueueResult(folder.name, "blocked_partial", "existing dataset or BUILDING folder without ZIP"))

    if args.list_only:
        for folder in selected:
            trigger = derive_trigger(folder.name)
            print(
                f"[WOULD PROCESS] {folder.name} "
                f"({len(direct_images(folder))} source images)"
            )
            print(f"                Trigger: {trigger}")
            results.append(
                QueueResult(
                    folder.name,
                    "listed",
                    f"{len(direct_images(folder))} source images; trigger={trigger}",
                )
            )
        summary = write_summary(output_root, results)
        print(f"\nList-only summary: {summary}")
        return 0

    failures = 0
    for index, folder in enumerate(selected, start=1):
        print()
        print("-" * 68)
        print(f"[{index}/{len(selected)}] Processing {folder.name}")
        print("-" * 68)

        command = [sys.executable, str(builder), str(folder)]
        completed_process = subprocess.run(command, check=False)

        final_zip = output_root / f"{folder.name}_TRAINING_READY.zip"
        if completed_process.returncode == 0 and final_zip.is_file():
            print(f"[SUCCESS] {folder.name}")
            results.append(QueueResult(folder.name, "success", str(final_zip)))
        else:
            failures += 1
            print(f"[FAILED]  {folder.name} (exit code {completed_process.returncode})")
            results.append(
                QueueResult(
                    folder.name,
                    "failed",
                    f"builder exit code {completed_process.returncode}",
                )
            )

    unselected = unfinished[len(selected):]
    for folder in unselected:
        results.append(QueueResult(folder.name, "not_run", "outside this run's limit"))

    summary = write_summary(output_root, results)

    print()
    print("=" * 68)
    print(" QUEUE FINISHED")
    print("=" * 68)
    print(f"Attempted : {len(selected)}")
    print(f"Succeeded : {len(selected) - failures}")
    print(f"Failed    : {failures}")
    print(f"Remaining : {len(unselected)}")
    print(f"Summary   : {summary}")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
