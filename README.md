# depth2room

Depth-conditioned video generation for room furnishing using [VACE](https://github.com/ali-vilab/VACE) (Wan2.1).

Given structural depth maps of empty rooms (walls, floor, ceiling) from [CAD-Estate](https://github.com/google-research/cad-estate), generates photorealistic videos of furnished interiors.

## Installation

```bash
# Clone
git clone https://github.com/thomaskiefer/depth2room.git
cd depth2room

# Install core + training dependencies
pip install -e ".[training]"

# For data preparation (requires cad_estate clone):
pip install -e ".[data]"

# For inference:
pip install -e ".[inference]"

# For evaluation metrics (LPIPS, SSIM):
pip install -e ".[eval]"

# Everything:
pip install -e ".[all]"
```

### External Dependencies

**CAD-Estate** (data preparation only):
```bash
git clone https://github.com/google-research/cad-estate.git
export CAD_ESTATE_SRC=/path/to/cad-estate/src
```

**VACE native inference** (optional, for `infer_depth2rgb.py`):
```bash
git clone https://github.com/Wan-Video/Wan2.1.git
export VACE_ROOT=/path/to/Wan2.1/vace
```

## Pipeline

1. **Data preparation** (`depth2room.data`): Render depth maps from CAD-Estate room structures, generate captions, create training metadata
2. **Training** (`depth2room.training`): Fine-tune VACE Wan2.1 with depth conditioning + validity masks
3. **Inference** (`depth2room.inference`): Generate furnished room videos from depth inputs

## Training

```bash
# Full fine-tuning
bash scripts/train_full.sh

# LoRA fine-tuning
bash scripts/train_lora.sh
```

## Project Structure

```
src/depth2room/
├── data/                  # Data preparation (CAD-Estate rendering, captions)
├── training/              # Training pipeline (dataset, VACE unit, trainer)
├── inference/             # Inference & evaluation
└── utils/                 # Visualization utilities
scripts/                   # Shell launch scripts
```
