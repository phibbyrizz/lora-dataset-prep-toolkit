# Third-party components

This repository does **not** bundle third-party model weights.

## JoyCaption

Captioning uses the Hugging Face model:

`fancyfeast/llama-joycaption-beta-one-hf-llava`

The model is downloaded by Hugging Face/Transformers when first used. The model repository includes its own Llama license and use-policy files. Those terms apply to the model independently of this repository's MIT license.

Upstream project: `fpgaminer/joycaption`

## OpenCV YuNet

Face-crop detection uses OpenCV's YuNet face detector. The pipeline can use `face_detection_yunet_2023mar.onnx` when placed in a discoverable tools location. The model originates from OpenCV Zoo and is subject to its upstream license terms.

## Python dependencies

Python packages installed from `requirements.txt`, plus PyTorch, retain their respective upstream licenses.

## LoRA Dataset Studio

LoRA Dataset Studio by `perfectgf` was inspected during provenance review because it had previously been evaluated in the same workflow. No distinctive identifiers from this prep pipeline or the separate Dataset Collector were found in the inspected LoRA Dataset Studio Windows package. **No LoRA Dataset Studio source code is included in this release.**

## OpenCV Zoo YuNet
The toolkit downloads `face_detection_yunet_2023mar.onnx` from the official OpenCV Zoo repository during setup when it is missing. OpenCV Zoo states that all files in the YuNet model directory are licensed under the MIT License. The model is verified against the checksum expected by this release and is not stored in this Git repository.
