# LoRA Dataset Prep Toolkit v0.1.4

A Windows-first preparation pipeline for turning a **manually curated folder of character/identity images** into a validated, training-ready LoRA dataset.

It preserves your originals, creates conservative face crops, captions originals and crops with JoyCaption, injects your trigger token, removes stable identity descriptions that should be learned by the trigger, validates every image/caption pair, and creates a training ZIP plus diagnostic reports.

> **Important:** This release contains the **dataset preparation pipeline only**. The separate image-search Dataset Collector is intentionally not included because its original authorship/license has not yet been established with enough confidence for redistribution.

## What it does

For each source folder:

1. Copies source images without modifying them.
2. Generates conservative YuNet face crops when appropriate.
3. Captions every original and crop with JoyCaption.
4. Adds the subject trigger to each caption.
5. Removes generic subject labels and stable identity traits that should be learned from the trigger.
6. Conservatively removes exact duplicate concepts and clear generated contradictions such as conflicting framing or expression labels.
7. Rejects/records captions that violate the validation rules.
8. Refuses to produce a final ZIP if any image lacks a valid caption.
9. Produces a training-ready image/text folder, ZIP, build summary, caption validation report, and diagnostics.

## Captioning model

The pipeline uses:

`fancyfeast/llama-joycaption-beta-one-hf-llava`

The model is downloaded on first use through Hugging Face/Transformers and is **not bundled in this repository**. The exact custom instruction used for captioning is documented in `CAPTIONING_PROMPT.md`.

The captioning policy deliberately describes **variables** such as framing, pose, expression, hairstyle, clothing, accessories, makeup, background, lighting and camera angle while suppressing **stable identity concepts** such as eye color, skin tone, facial structure/proportions and body build. The trigger token is intended to carry the identity.

## Requirements

- Windows 10/11
- Python 3.10-3.12 recommended
- NVIDIA GPU with a working CUDA-enabled PyTorch installation
- Enough disk space for the JoyCaption model cache and generated datasets
- Internet access on first JoyCaption model download

JoyCaption's native model is large. GPU/VRAM limitations may require a different model-loading strategy; this release preserves the loading behavior used by the validated v0.1.4 workflow rather than silently changing model precision.

## Installation

### 1. Download

Download the repository ZIP or clone the repository into a normal writable folder.

Do **not** put a Python virtual environment or model weights into the GitHub repository.

### 2. Install Python

Install Python 3.10, 3.11 or 3.12 and ensure `python` is available from Command Prompt.

### 3. Run setup

Double-click `SETUP_WINDOWS.bat`. It creates a private `.prep_venv`, installs the CUDA 12.8 PyTorch build used by this release, installs the remaining Python dependencies, verifies CUDA/GPU access, and downloads/verifies the small YuNet face-detector model from the official OpenCV Zoo source.

Setup is complete only after the final import, GPU, and runtime-asset checks pass. You do not need to activate the virtual environment manually.

## Runtime models and first-run download

Face cropping uses OpenCV YuNet. `SETUP_WINDOWS.bat` downloads `face_detection_yunet_2023mar.onnx` directly from the official OpenCV Zoo source when missing and verifies its SHA-256 checksum. The weight is intentionally not committed to this repository.

JoyCaption is much larger and is not bundled. Before dataset processing starts, the launcher runs a preflight that verifies the environment and fully caches `fancyfeast/llama-joycaption-beta-one-hf-llava` through Hugging Face. The first run can therefore take a while and requires substantial free disk space and internet access. Later runs reuse the Hugging Face cache.

The validated pipeline loads JoyCaption in 4-bit NF4 mode to reduce VRAM use. Upstream JoyCaption documentation states that native BF16 loading is about 17 GB VRAM; 4-bit loading is specifically used here to make the workflow practical on lower-VRAM NVIDIA GPUs. Hardware varies, so very low-VRAM GPUs can still run out of memory.

## Preparing source folders

Each subject should have a folder containing the curated source images directly inside it.

Example:

```text
Source/
  John_Smith/
    image01.jpg
    image02.jpg
    ...
  Jane_Doe/
    image01.jpg
    ...
```

Supported image types include JPG/JPEG, PNG, WEBP, BMP, TIFF and TIF.

### Trigger naming

**The trigger matters:** it is inserted at the beginning of every generated caption. The single-dataset launcher always shows the inferred trigger and lets you override it before processing. Batch mode is unattended, so it always uses the folder-derived trigger.

Single underscores become spaces and the derived trigger is lowercased:

```text
John_Smith -> john smith
```

Use a double underscore for multiple aliases:

```text
John_Smith__Johnny -> john smith, johnny
```

A manual trigger can also be supplied from the command line:

```text
python build_lora_dataset.py "C:\path\to\subject" --trigger "custom trigger"
```

## Run one dataset

Double-click `RUN_ONE_DATASET.bat` and select one curated subject folder in the Windows folder picker. You can also drag a subject folder onto the BAT as a shortcut.

Before touching the dataset, the launcher checks the private environment, CUDA/GPU, YuNet integrity, disk space, and JoyCaption cache. On first use it downloads JoyCaption completely before processing begins. The launcher derives a trigger from the folder name, shows it before processing, and asks you to press Enter to accept it or type a different trigger. The confirmed trigger is inserted at the beginning of every caption.

If validation fails, the failed build is **not** packaged as training-ready. Read the generated reports before retrying.

## Run a folder queue

Double-click:

`RUN_CURRENT_BATCH.bat`

If the script cannot locate your `DATASETS\Source` directory automatically, it opens a folder picker.

The launcher offers:

1. Process the first five unfinished folders as a stress test.
2. Process every unfinished folder.
3. Preview the first five folders and derived triggers without processing.

The queue skips subjects that already have a completed training-ready ZIP. It blocks rather than overwrites ambiguous partial outputs. One failed subject does not stop later subjects.

## Output

Queue output is written beneath:

```text
Source/_pipeline_output/
```

Per-subject output includes the prepared image/text pairs, training-ready ZIP and diagnostic/validation reports. Queue runs also produce timestamped CSV summaries.

## Validation philosophy

The pipeline has a hard final gate: **every training image must have one meaningful current caption**. Captions are post-processed to prevent stable identity descriptions from competing with the trigger token. A conservative consistency pass also removes exact duplicate comma-separated clauses and, when JoyCaption emits conflicting framing labels, keeps the first explicit framing clause. It deliberately does not rewrite subjective details such as expression, clothing, or pose. Rejected captions are retained in reports so failures are inspectable rather than silently accepted.

Automated captioning is not infallible. Manually review important datasets and captions before training.

## Dataset guidance

This tool prepares files; it cannot make a weak source dataset strong. For identity LoRAs, prioritize clear, varied, correctly identified images. Avoid filling a dataset with near-duplicates or a single framing/setting. Keep enough variation in pose, crop, expression, clothing, lighting and background that the subject identity is the stable common factor.

## Privacy and responsible use

- Face detection/cropping is performed locally.
- JoyCaption runs locally after its model files have been downloaded.
- Do not train on material you do not have the right or appropriate permission to use.
- You are responsible for complying with applicable privacy, publicity, copyright, platform and model-license requirements.
- Do not use the toolkit for impersonation, fraud, harassment, or exploitative content.

## Troubleshooting

**`CUDA GPU not available`**  
Your Python environment does not have a working CUDA-enabled PyTorch installation. Verify `torch.cuda.is_available()` inside the same environment used by the launcher.

**Missing package error**  
Run `SETUP_WINDOWS.bat` again. If you maintain your own environment, install `requirements.txt` without replacing a known-good CUDA PyTorch build with a CPU build.

**JoyCaption download/loading failure**  
The launcher intentionally downloads/caches the model before dataset processing. Confirm internet access, Hugging Face availability, and free disk space. Retry the launcher after correcting the problem; Hugging Face normally resumes/reuses cached files.

**CUDA out of memory while loading/captioning**  
Close ComfyUI, games, browsers using GPU acceleration, and other GPU-heavy applications, then retry. This release uses 4-bit JoyCaption, but very low-VRAM GPUs may still be insufficient.

**YuNet download/checksum failure**  
Rerun `SETUP_WINDOWS.bat` with internet access. Setup obtains YuNet from official OpenCV Zoo mirrors, validates the exact 232,589-byte file, verifies SHA-256, and performs an OpenCV load test. Invalid or partial files are replaced automatically.

**Queue says a subject is blocked**  
A matching `_dataset` or `_BUILDING` output already exists. Inspect the previous reports before deleting or retrying anything. This is an overwrite-safety feature.

**Training ZIP is not created**  
The validation gate found one or more missing/invalid captions. Inspect `caption_validation.csv` and rejected-caption reports.

## Version history

### v0.1.4
- Explicit multiple trigger aliases using double underscores.
- Windows one-click environment setup and CUDA verification.
- Automatic YuNet download with checksum verification.
- Runtime preflight before dataset processing.
- First-run JoyCaption cache verification/download before source processing.
- Folder picker for single-dataset and queue launchers.
- Removed private-machine path assumptions from public launchers.
- Single-dataset launcher now previews the inferred trigger and allows an override before processing.
- Conservative caption consistency cleanup removes exact duplicate clauses and conflicting extra framing labels.
- Suppresses the harmless Hugging Face Windows symlink-cache warning in the launcher.

### v0.1.3
- Resumable all-folder queue around the validated per-folder builder.

### v0.1.2
- Identity-caption cleanup including removal of eyebrow-only identity descriptions.

### v0.1.x
- Original/crop preservation, YuNet crops, JoyCaption captioning, trigger injection, identity cleanup, hard validation gate, ZIP/report output.

## Licensing and attribution

The preparation scripts in this repository are released under the MIT License. See `LICENSE`.

Third-party models and packages are **not relicensed** by this repository. See `THIRD_PARTY_NOTICES.md` and the upstream terms for JoyCaption, Llama, OpenCV, PyTorch and other dependencies.

### Dataset Collector status

A separate `LoRA Dataset Collector v3.3.1` exists in the private working toolkit. Its source contains no embedded author, copyright, upstream URL or license notice. Searches for distinctive code identifiers did not establish a public upstream, and comparison against the inspected LoRA Dataset Studio Windows package did not show distinctive identifier matches. Because provenance is unresolved, **the Collector is not included in this public release**. This avoids claiming ownership or redistributing code without confirmed permission.

## Contributing

Bug reports and improvements are welcome. When reporting a failure, include the toolkit version, Python/PyTorch versions, GPU, the error text and the relevant generated report. Do not post private images, access tokens, personal file paths or other sensitive information.


## Running one dataset on Windows

Double-click `RUN_ONE_DATASET.bat`. A Windows folder picker opens; select one curated subject folder and the pipeline will process it. As an optional shortcut, you can drag a subject folder onto `RUN_ONE_DATASET.bat`. Canceling the picker exits without changing anything. Run `SETUP_WINDOWS.bat` first.
