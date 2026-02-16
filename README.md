# 🌿 Virtual Plant Care Assistance  
### 🍃 Intelligent Plant Leaf Disease Detection using Image Processing & Machine Learning

> An AI-powered virtual assistant that analyzes plant leaf images to detect diseases and help farmers/gardeners take early action.

---

## 🚀 Project Overview

**Virtual Plant Care Assistance** is a classical Machine Learning–based system that detects whether a plant leaf is **Healthy or Diseased** using image processing and handcrafted feature extraction techniques.

Instead of Deep Learning, this project focuses on:

✅ Image Preprocessing  
✅ Feature Engineering  
✅ Traditional ML Classifiers  
✅ Random Forest for final prediction  

The goal is to provide a **lightweight, explainable, and beginner-friendly AI solution** for agriculture.

---

## ✨ Key Highlights

🌱 Image Segmentation using HSV color space  
📊 Global Feature Extraction (Color, Texture, Shape)  
🧠 Multiple ML models comparison  
🌲 Random Forest Classifier (Final Model)  
📈 ~97% Accuracy  
💾 Features stored using HDF5  
🧪 End-to-end ML pipeline in Python  

---

## 🖼 Dataset

Dataset taken from **PlantVillage (Apple Leaves)**:

- Healthy Leaves  
- Diseased Leaves (Apple Scab, Black Rot, Cedar Apple Rust)

Structure:

virtual-plant-care-assistance/
│
├── image_classification/
│ ├── dataset/
│ │ ├── train/
│ │ │ ├── healthy/
│ │ │ └── diseased/
│ │ └── test/
│ │
│ └── output/
│ ├── train_data.h5
│ └── train_labels.h5
│
├── utils/
│ └── test.py
│
├── full_pipeline.py
├── test_image.py
├── testing.ipynb
├── requirements.txt
└── README.md

Each image is resized and processed before feature extraction.

---

## 🔬 Image Properties

| Property | Value |
|----------|------|
| Format | JPG |
| Size | 256 × 256 |
| Bit Depth | 24 |
| Resolution | 96 DPI |

---

## 🧩 Workflow

### 1️⃣ Image Loading  
Leaf images are read and resized.

---

### 2️⃣ Color Conversion  
BGR → RGB → HSV  

HSV helps separate color from intensity, improving segmentation.

---

### 3️⃣ Image Segmentation  
Green & brown regions are extracted to isolate leaf area from background.

---

### 4️⃣ Feature Extraction  

Three global descriptors are used:

### 🎨 Color  
- HSV Color Histogram  

### 🧱 Texture  
- Haralick Features  

### 📐 Shape  
- Hu Moments  

All features are concatenated into a single vector.

---

### 5️⃣ Feature Scaling  

MinMaxScaler → Values normalized between 0 and 1.

---

### 6️⃣ Feature Storage  

Saved using **HDF5** format:
image_classification/output/
├── train_data.h5
└── train_labels.h5


---

### 7️⃣ Machine Learning Models

The following classifiers are evaluated:

- Logistic Regression  
- Linear Discriminant Analysis  
- K-Nearest Neighbors  
- Decision Tree  
- Random Forest  
- Naive Bayes  
- Support Vector Machine  

10-fold cross validation is applied.

---

### 🏆 Best Model

✅ **Random Forest Classifier**

Achieved approximately **97% accuracy**.

---

## ⚙️ How to Run

# Step 0 — Go to project folder
```bash
cd Plant-Disease-Detection-master
```
# Step 1 — Create virtual environment (Python 3.7 recommended)
```bash
py -3.7 -m venv plantenv
```
# Step 2 — Activate virtual environment
```
plantenv\Scripts\activate
```
# Step 3 — Upgrade pip
```
python -m pip install --upgrade pip
```
# Step 4 — Install project requirements
```
pip install -r requirements.txt
```
# Step 5 — Install remaining libraries
```
pip install opencv-python scikit-learn mahotas h5py seaborn matplotlib joblib
```
# Step 6 — Verify installation
```
python -c "import cv2,sklearn,mahotas,h5py; print('ALL OK')"
```
# Step 7 — (Optional) Test image loading
```
python test_image.py
```
# Step 8 — Run full machine learning pipeline
```
python full_pipeline.py
```
