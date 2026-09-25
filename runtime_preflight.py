#!/usr/bin/env python3
"""Runtime dependency/asset preflight for LoRA Dataset Prep Toolkit v0.1.4.

Temporary model downloads deliberately retain the .onnx suffix because OpenCV
uses the filename extension when selecting the network importer.
"""
from __future__ import annotations
import argparse, hashlib, shutil, sys, urllib.request
from pathlib import Path

YUNET_NAME = "face_detection_yunet_2023mar.onnx"
YUNET_URLS = [
    "https://huggingface.co/opencv/opencv_zoo/resolve/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx?download=true",
    "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
]
YUNET_SHA256 = "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"
JOYCAPTION_REPO = "fancyfeast/llama-joycaption-beta-one-hf-llava"
MIN_FREE_GB_WARNING = 25


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _validate_yunet(path: Path) -> tuple[bool, str]:
    if not path.exists():
        return False, "missing"
    size = path.stat().st_size
    if size != 232589:
        return False, f"unexpected size {size} bytes"
    got = sha256(path)
    if got.lower() != YUNET_SHA256.lower():
        return False, f"checksum mismatch ({got})"
    try:
        import cv2
        detector = cv2.FaceDetectorYN.create(str(path), "", (320, 320), 0.9, 0.3, 5000)
        if detector is None:
            return False, "OpenCV could not load the model"
    except Exception as e:
        return False, f"OpenCV load test failed: {e}"
    return True, "ok"


def download_yunet(dest: Path) -> None:
    tmp = dest.with_name(dest.stem + ".download" + dest.suffix)
    tmp.unlink(missing_ok=True)
    print("YuNet face detector is missing or invalid; downloading official OpenCV Zoo model...")
    errors = []
    for url in YUNET_URLS:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "LoRA-Dataset-Prep-Toolkit/0.1.4"})
            with urllib.request.urlopen(req, timeout=90) as r, tmp.open("wb") as f:
                shutil.copyfileobj(r, f)
            ok, reason = _validate_yunet(tmp)
            if not ok:
                raise RuntimeError(reason)
            tmp.replace(dest)
            print(f"YuNet ready: {dest}")
            return
        except Exception as e:
            errors.append(f"{url}: {e}")
            tmp.unlink(missing_ok=True)
    raise RuntimeError("All official YuNet download sources failed. " + " | ".join(errors))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ensure-joycaption", action="store_true", help="Fully cache JoyCaption before processing")
    args = ap.parse_args()
    root = Path(__file__).resolve().parent

    try:
        import cv2
        import torch
        import transformers  # noqa: F401
        import accelerate  # noqa: F401
        import bitsandbytes  # noqa: F401
        from PIL import Image  # noqa: F401
        from huggingface_hub import snapshot_download
    except Exception as e:
        print(f"ERROR: Python dependency check failed: {e}", file=sys.stderr)
        print("Run SETUP_WINDOWS.bat and try again.", file=sys.stderr)
        return 2

    if not hasattr(cv2, "FaceDetectorYN"):
        print("ERROR: OpenCV FaceDetectorYN is unavailable. Run SETUP_WINDOWS.bat again.", file=sys.stderr)
        return 2
    if not torch.cuda.is_available():
        print("ERROR: CUDA is unavailable in the toolkit environment.", file=sys.stderr)
        print("Run SETUP_WINDOWS.bat and verify the NVIDIA driver/PyTorch installation.", file=sys.stderr)
        return 2

    gpu = torch.cuda.get_device_name(0)
    total_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"GPU ready: {gpu} ({total_gb:.1f} GB VRAM)")
    if total_gb < 8:
        print("WARNING: Less than 8 GB VRAM detected. The validated 4-bit JoyCaption path may run out of memory.")

    yunet = root / YUNET_NAME
    ok, reason = _validate_yunet(yunet)
    if not ok:
        if yunet.exists():
            print(f"Existing YuNet model is invalid ({reason}); replacing it automatically...")
            yunet.unlink(missing_ok=True)
        try:
            download_yunet(yunet)
        except Exception as e:
            print(f"ERROR: Could not obtain a verified YuNet model: {e}", file=sys.stderr)
            print("Check internet access and rerun SETUP_WINDOWS.bat.", file=sys.stderr)
            return 3
    else:
        print(f"YuNet ready: {yunet}")

    free_gb = shutil.disk_usage(Path.home()).free / (1024**3)
    print(f"Free space on user-profile drive: {free_gb:.1f} GB")
    if free_gb < MIN_FREE_GB_WARNING:
        print(f"WARNING: Under {MIN_FREE_GB_WARNING} GB free. First-run JoyCaption caching is large and may fail for lack of disk space.")

    if args.ensure_joycaption:
        print("Checking JoyCaption model cache. First use may require a large Hugging Face download...")
        try:
            snapshot_download(repo_id=JOYCAPTION_REPO)
        except Exception as e:
            print(f"ERROR: JoyCaption could not be fully cached: {e}", file=sys.stderr)
            print("Check internet access, Hugging Face availability, and free disk space, then retry.", file=sys.stderr)
            return 4
        print("JoyCaption model cache ready.")

    print("Runtime preflight passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
