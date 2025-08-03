
# Detecting AI-Generated Images using Deep Learning

This project focuses on detecting AI-generated images using Resnet-18 on the CIFAKE dataset. With the recent advancements in generative models, the ability to detect images as fake becomes critical. This binary classifier is designed to determine whether an image is real (from CIFAR-10) or fake (generated using Stable Diffusion conditioned on CIFAR-10 images).

---

## Dataset

- **Name**: [CIFAKE Dataset](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images)
- **Size**: 120,000 total images (60k real, 60k fake)
- **Format**: 32x32 RGB images
- **Real**: Images from CIFAR-10  
- **Fake**: Generated using Stable Diffusion

A subset of this sample was used to reduce computational time.

---

## Project Structure

```bash
├── Code.ipynb               # Main notebook containing code
├── Presentation Slides.pptx # Project presentation slides
├── README.md                # Project documentation
├── assets/                  # Folder for visualizations
└── Report.docx              # Report detailing methodology and results
```

---

## Model Architecture

- **Base Model**: ResNet-18 (trained from scratch)
- **Loss Function**: Binary Cross-Entropy Loss (BCE)
- **Optimizer**: Adam
- **Activation Function**: Sigmoid (binary classification)
- **Regularization**:
  - Early stopping (patience = 3)
  - Data augmentation (horizontal flip)

---

## Training Details

- **Train / Test Split**: 80 / 20
- **Train / Val Split**: 80 / 20
- **Batch Size**: 16 (chosen using hyperparameter tuning)
- **Epochs**: ~14 (early stopping applied)
- **Learning Rate**: 0.0005 (chosen using hyperparameter tuning)

---

## Evaluation Metrics on Test

| Metric      | Value     |
|-------------|-----------|
| Accuracy    | ~95%      |
| Precision   | ~94%      |
| Recall      | ~96%      |
| F1 Score    | ~95%      |


---

## Model Interpretability

Grad-CAM (Gradient-weighted Class Activation Mapping) was used to visualize which parts of the image influenced the model's decision:

- Real Images: Focus is on main object in photo and more of the image is used for decision making  
- Fake Images: Focus is on smooth/blurred regions in small areas of the photo, mainly background inconsistencies


---

## Limitations & Improvements

### Limitations
- Poor generalization to non-CIFAR fake images or high-res real images
- Susceptible to domain shift and resolution mismatch

### Potential Improvements
- Include AI-generated images from other models (e.g., Midjourney, DALLE)
- Train on higher-resolution or multi-resolution datasets

---

## Getting Started

### Prerequisites
- Python ≥ 3.10
- PyTorch
- torchvision
- matplotlib
- seaborn
- numpy
- sklearn
- OpenCV (for Grad-CAM)


### Run the Notebook
Open and run `Code.ipynb` in Google Colab. The dataset will directly load from Kaggle.

---

## References

- [CIFAKE Dataset on Kaggle](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images)
- [CIFAKE: Image Classification and Explainable Identification of AI-Generated Synthetic Images](https://www.researchgate.net/publication/377538637_CIFAKE_Image_Classification_and_Explainable_Identification_of_AI-Generated_Synthetic_Images)
- [Grad-CAM Adaptation](https://github.com/jacobgil/pytorch-grad-cam)

---

## Author

**Yaqin Albirawi**  
Fanshawe College – Deep Learning with Pytorch Capstone Project  
August 2025

---
