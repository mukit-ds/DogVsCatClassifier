# 🐱🐶 Cat vs. Dog Image Classifier — CNN with Transfer Learning (VGG16)

A binary image classifier that distinguishes between cats and dogs using **Convolutional Neural Networks (CNNs)** and **transfer learning** with the pre-trained VGG16 architecture. Achieves ~95.6% validation accuracy in just 5 epochs.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Results](#results)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Training Configuration](#training-configuration)
- [License](#license)

---

## Overview

This project leverages **transfer learning** with VGG16 (pre-trained on ImageNet) to build a robust cat vs. dog classifier. Rather than training a deep CNN from scratch — which would require a large dataset and significant compute — we freeze the earlier convolutional blocks and fine-tune only the last block (`block5`) to adapt the network to our binary classification task.

**Why transfer learning?**
- Dramatically reduces training time and required data
- Leverages rich visual features already learned from 1.2M ImageNet images
- Achieves high accuracy with minimal overfitting

---

## Results

| Metric | Value |
|---|---|
| Architecture | VGG16 + Custom Classification Head |
| Fine-tuned from | `block5_conv1` |
| Input Size | 150 × 150 × 3 |
| Optimizer | RMSprop (lr = 1e-5) |
| Loss Function | Binary Crossentropy |
| Epochs | 5 |
| Training Accuracy | ~95.4% |
| Validation Accuracy | ~95.6% |

---

## Project Structure

```
CatDogClassifier/
│
├── CatDogCNNClassifier.ipynb   # Main notebook (training + inference)
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

---

## Getting Started

### Prerequisites

- Python 3.10+
- A [Kaggle account](https://www.kaggle.com) with an API key (`kaggle.json`)
- GPU runtime recommended (Google Colab or local CUDA setup)

### Installation

**1. Clone the repository**
```bash
git clone https://github.com/kazirafi17/DogVsCatClassifier.git
cd DogVsCatClassifier
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Set up Kaggle credentials**

Download your `kaggle.json` API key from [kaggle.com/settings](https://www.kaggle.com/settings) and place it in the project directory. The notebook will configure it automatically.

### Running the Notebook

Open `CatDogCNNClassifier.ipynb` in Jupyter or Google Colab and run the cells sequentially. The notebook will:

1. Download and extract the dataset from Kaggle
2. Preprocess and augment the training images
3. Build and fine-tune the VGG16-based model
4. Plot training/validation accuracy and loss curves
5. Run inference on new images

---

## Model Architecture

```
Input (150 × 150 × 3)
        │
   ┌────▼─────┐
   │  VGG16   │  ← Pre-trained on ImageNet
   │          │     blocks 1–4: frozen
   │          │     block5:     fine-tuned
   └────┬─────┘
        │
     Flatten
        │
   Dense(256, ReLU)
        │
   Dense(64, ReLU)
        │
   Dense(1, Sigmoid)
        │
   Output: P(Dog)
```

A sigmoid output > 0.5 is classified as **Dog**, otherwise **Cat**.

---

## Dataset

**Dogs vs. Cats** from Kaggle — 25,000 labeled images (12,500 cats, 12,500 dogs) split into `train/` and `test/` directories.

```
kaggle datasets download -d salader/dogs-vs-cats
```

Images vary in resolution and are resized to **150 × 150** during preprocessing.

### Data Augmentation (training only)

| Technique | Value |
|---|---|
| Rescale | 1/255 |
| Shear Range | 0.2 |
| Zoom Range | 0.2 |
| Horizontal Flip | Enabled |

---

## Training Configuration

```python
IMG_SIZE   = (150, 150)
BATCH_SIZE = 32
EPOCHS     = 5
OPTIMIZER  = RMSprop(learning_rate=1e-5)
LOSS       = "binary_crossentropy"
SEED       = 42
```

The low learning rate (`1e-5`) is intentional — it prevents the fine-tuning step from overwriting the pre-trained weights too aggressively.

---

## License

This project is open-source and available under the [MIT License](https://opensource.org/licenses/MIT).
