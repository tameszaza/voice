# mGANs: Multi-Generator Adversarial Networks for Synthetic Speech Detection

mGANs (Multi-Generator Adversarial Networks) is a research-focused project aimed at enhancing the detection of synthetic speech using Generative Adversarial Networks with multiple generators. This model is designed to address the challenges of generalizing synthetic speech detection across diverse datasets.

---

## Features
- **Multiple Generators:** Utilizes multiple generators to create a diverse dataset for training, improving generalization.
- **Enhanced Detection Accuracy:** Focuses on distinguishing AI-generated speech from real human speech.
- **Dataset Augmentation:** Leverages GAN-generated data to overcome limitations of existing datasets.
- **Scalable Architecture:** Optimized for use on high-performance computing clusters.

---

## Research Objectives
1. Improve synthetic speech detection accuracy on generalized datasets.
2. Evaluate the performance of mGANs in comparison to traditional GANs.
3. Provide insights into the use of adversarial networks for security applications in voice synthesis.

---

## Requirements

### Hardware
- **GPU:** Recommended for faster training.
- **High-Performance Cluster:** Supported for large-scale experiments.

### Software
- Python 3.8 or later
- TensorFlow/PyTorch (Latest stable version)
- Libraries: 
  - numpy
  - pandas
  - matplotlib
  - librosa
  - scikit-learn

Install dependencies with:
```bash
pip install -r requirements.txt
