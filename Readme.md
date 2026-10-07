# Atlas-Aware Facade Style Transfer for 3D Building Models

**"A Facade-Based Pipeline for Structure-Preserving Style Transfer of Atlas Textures for Building Models"**

This repository provides a demo implementation of an atlas-aware, facade-based texture style-transfer pipeline for existing textured 3D building models.

The pipeline converts fragmented texture atlases into facade-aligned images, applies structure-preserving diffusion-based stylization, and reconstructs the stylized facades into the original texture-atlas layout while retaining the original building geometry, UV coordinates, and material assignments.

## Overview

<p align="center">
  <img src="figures/Propose_pipline.png" width="900">
</p>

The pipeline modifies the texture appearance only and does not require remeshing of the source building asset.

## Demo overview
<p align="center">
  <img src="figures/Demo_overview.png" width="900">
</p>

The demo supports five scenario-oriented facade appearance presets:

- Fire damage
- Corrosion and rust
- Traffic-induced soiling
- Post-earthquake cracking
- Facade peeling

(A custom reference image and text prompt can also be used to specify the target appearance)

## Tested environment:
- Ubuntu 22.04
- NVIDIA RTX 4090 24 GB
- Python 3.10
- PyTorch 2.6.0 + CUDA 12.4

## Installation

After cloning this repo, please run the command below.

```bash
cd Style_transfer_for_Building_model

python3 -m venv venv
source venv/bin/activate

pip install --upgrade pip

pip install torch==2.6.0 torchvision==0.21.0 \
    --index-url https://download.pytorch.org/whl/cu124

# no weights are downloaded
bash scripts/setup_engine.sh
pip install -r requirements.txt
```

## Model Weights

scripts/download_models.sh downloads these files automatically.
To download them by hand instead, get each file from its Source and save it under engine/ComfyUI/<Destination>.

The following pretrained model weights are required

| Component | Source | Destination under `engine/ComfyUI/` |
|---|---|---|
| Juggernaut XL XI | [Civitai](https://civitai.com/api/download/models/782002) | `models/checkpoints/juggernautXL_juggXIByRundiffusion.safetensors` |
| T2I-Adapter LineArt SDXL | [Hugging Face](https://huggingface.co/TencentARC/t2i-adapter-lineart-sdxl-1.0) | `models/controlnet/t2i-adapter-lineart-sdxl-1.0.fp16.safetensors` |
| IP-Adapter Plus SDXL | [Hugging Face](https://huggingface.co/h94/IP-Adapter) | `models/ipadapter/ip-adapter-plus_sdxl_vit-h.safetensors` |
| CLIP Vision ViT-H/14 | [Hugging Face](https://huggingface.co/h94/IP-Adapter) | `models/clip_vision/CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors` |
| LineArt Annotator | [Hugging Face](https://huggingface.co/lllyasviel/Annotators) | `custom_nodes/comfyui_controlnet_aux/ckpts/lllyasviel/Annotators/` |
| SAM 2.1 Hiera-Small | [Hugging Face](https://huggingface.co/facebook/sam2.1-hiera-small) | `models/sam2/sam2.1_hiera_small.pt` |

The required model files can be downloaded automatically:

```bash
bash scripts/download_models.sh
```

The downloaded files can be verified using:

```bash
bash scripts/download_models.sh --check
```

## Thanks to
We thank the authors of [StyleCity3D](https://github.com/chenyingshu/stylecity3d) for providing the public 3D scene data. Example buildings used in our public demo were extracted and prepared from the Tokyo (Shibuya) and Los Angeles scenes released with StyleCity3D.