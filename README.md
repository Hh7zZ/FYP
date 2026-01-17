<div align="center">

# Generalizing Zero-Shot Robotic Grasping through One-Shot Open-Vocabulary Affordance Learning

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

</div>

## 📖 Abstract

As robotic applications expand from structured industrial environments to unstructured domestic service domains, robots are required not only to recognize object categories but also to comprehend object affordance to facilitate effective interaction. Addressing the challenges inherent in traditional robotic grasping methods—specifically their heavy reliance on massive annotated datasets and confinement to closed-set vocabularies—this paper proposes a **"One-Shot Open Affordance Learning"** framework.

This framework integrates the **DINOv3** model, employed for extracting fine-grained visual features, with the **CLIP** model enhanced by **Context Optimization (CoOp)**. Through the introduction of a **CLS-Guided Transformer Decoder**, the system achieves a precise mapping from abstract semantic instructions to pixel-level affordance regions. Unlike methods requiring millions of training iterations, the proposed approach achieves lightweight training using **only one annotated example** in each base category and demonstrates robust zero-shot inference capabilities.

## 📊 Results & Visualization

The figure below demonstrates the model's capability to generalize affordance learning to both seen and unseen categories.

<div align="center">
  <img src="assets/affordance_visualization.png" alt="Affordance Prediction Visualization" width="100%">
  <br>
  <em>Figure 1: Qualitative results showing input images (top row) and generated affordance heatmaps (bottom row). The model accurately identifies regions for actions like "Cut", "Hold", "Open" and "Carry" across various objects.</em>
</div>

<br>

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/Hh7zZ/FYP.git
cd your-repo-name

# Install dependencies
pip install -r requirements.txt

```
### 📥 Pre-trained Models

This project utilizes **DINOv3** as the visual backbone. Please download the specific pre-trained weights from the official repository and place them in the `weights` directory.

| Model Variant | Parameters | Pretraining Dataset | Download Link |
| :--- | :--- | :--- | :--- |
| **ViT-B/16 distilled** | **86M** | **LVD-1689M** | https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/|

**Setup Steps:**

1. Create a directory for weights:
   ```bash
   mkdir -p weights
2. Download the model (ViT-B/16 distilled, 86M) from the link above.

3. Rename the file to dinov3_vitb16.pth (optional, based on your config) and save it to the weights/ folder.
Example structure
.
├── weights/
│   └── dinov3_vitb16.pth  <-- Place the downloaded file here
├── train.py
└── ...

## 🚀 Usage

### Training
To train the model using the One-Shot setting:

```bash
python train.py
```

### Inference
To infer the affordance area in the objects:

```bash
python demo.py \
  --img "path_to_image" \
  --model_file "path_to_trained_weights"
```
