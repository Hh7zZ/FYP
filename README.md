<div align="center">

# Generalizing Zero-Shot Robotic Grasping through One-Shot Open-Vocabulary Affordance Learning

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

</div>

## 📖 Abstract

As robotic applications expand from structured industrial environments to unstructured domestic service domains, robots are required not only to recognize object categories but also to comprehend object affordance to facilitate effective interaction. Addressing the challenges inherent in traditional robotic grasping methods—specifically their heavy reliance on massive annotated datasets and confinement to closed-set vocabularies—this paper proposes a **"One-Shot Open Affordance Learning"** framework.

This framework integrates the **DINOv3** model, employed for extracting fine-grained visual features, with the **CLIP** model enhanced by **Context Optimization (CoOp)**. Through the introduction of a **CLS-Guided Transformer Decoder**, the system achieves a precise mapping from abstract semantic instructions to pixel-level affordance regions. Unlike methods requiring millions of training iterations, the proposed approach achieves lightweight training using **only one annotated example** in each base category and demonstrates robust zero-shot inference capabilities.
