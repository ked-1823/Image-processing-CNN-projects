# 🐾 Cats vs Dogs Image Classification

## 📌 Project Overview
This project is part of my **ML journey**, where I built an **end-to-end image classification pipeline** using Convolutional Neural Networks (CNNs).  
The goal is to classify images of cats and dogs with high accuracy using **transfer learning** and **data augmentation** techniques.

---

## 📂 Dataset
- **Source:** Kaggle "Cats vs Dogs" mini dataset  
- **Total Images:** 2000+ images (split into training & validation)  
- **Image Size:** Resized to 128x128 pixels  
- **Data Augmentation:** Rotation, flipping, zoom, shear, width/height shifts to improve model generalization

---

## 🧩 Data Preprocessing
- Images are **rescaled to [0,1]** to normalize pixel values.
- **ImageDataGenerator** is used to:
  - Split the dataset into **training (70%)** and **validation (30%)**
  - Apply real-time **data augmentation** including flips, rotations, zooms, and shifts
- Augmentation ensures the model sees **diverse variations** of the same image, reducing overfitting.

---

## 🧱 Model Architecture
- **Base Model:** MobileNetV2 (pre-trained on ImageNet)  
  - `include_top=False` → Removes the original classifier  
  - `trainable=False` → Freeze base layers to retain pre-learned features
- **Custom Layers:**
  - Global Average Pooling to reduce spatial dimensions
  - Dropout (0.5) to prevent overfitting
  - Dense layer with **2 neurons** and softmax activation for cat vs dog classification
- **Compilation:**
  - Optimizer: Adam  
  - Loss: Categorical Crossentropy  
  - Metric: Accuracy

---

## 🏋️ Training
- **Epochs:** Up to 50  
- **Early Stopping:** Monitors validation loss with `patience=5` to avoid overfitting
- **Training and Validation Generators:** Feed data in batches for efficient GPU usage
- The model learns **dataset-specific features** while leveraging the pre-trained MobileNetV2 features.

---

## 📊 Evaluation
- **Validation Accuracy & Loss** are measured after training to assess generalization.
- Accuracy/loss curves are plotted to visualize:
  - How well the model learned during training
  - Overfitting or underfitting trends

---

## 🖼️ Visualizations
- **Training vs Validation Accuracy & Loss:** Check learning curves to evaluate performance
- **Augmented Images:** Visualizing a batch of images after augmentation to verify transformations

---

## 💾 Saving the Model
- The trained model is saved as `cat_vs_dog.keras` for:
  - Future inference  
  - Deployment in applications  
  - Resuming training without starting over

---

## ✅ Key Learnings
- Data augmentation significantly improves model generalization  
- Transfer learning with MobileNetV2 reduces training time and improves accuracy  
- Visualizing training data and learning curves helps identify overfitting early

---

example : <img width="848" height="443" alt="Screenshot 2025-10-03 225125" src="https://github.com/user-attachments/assets/33250406-3c82-4a25-9291-ed62db059e1e" />


## 🚀 Deployment with Streamlit
You can deploy this Cats vs Dogs classifier as a **web application** using Streamlit.

### Steps to Run the App:
1. Install Streamlit if not already installed:  
   ```bash
   pip install streamlit

