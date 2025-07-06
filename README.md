# Adaptive-Lightweight-YOLOv5-Military-Target-Detection
A lightweight YOLOv5-based system for detecting and classifying military targets in low-visibility conditions like fog, smoke, and night vision scenarios.
# Adaptive Lightweight YOLOv5-Based Network for Military Target Detection

This repository contains a deep learning-based object detection system designed to classify and detect military targets—such as weapons, personnel, and equipment—under low-visibility conditions like fog, smoke, and poor lighting. Built using a lightweight version of YOLOv5, the system is optimized for real-time performance on resource-constrained devices such as drones and edge hardware.

## 📌 Key Highlights
- 🔍 Target detection in foggy, smoky, and low-light environments
- ⚙️ Lightweight YOLOv5 model optimized for speed and efficiency
- 🖼️ Image preprocessing pipeline: grayscale conversion, denoising, thresholding, sharpening
- 🧠 CNN-based classification with feature extraction using HOG
- 🎯 Real-time classification of military targets into risk categories (e.g., low, medium, high)

## 📁 Project Structure
- `model.py` – Training and inference using YOLOv5
- `gui.py` – Tkinter GUI for image upload and live detection
- `dataset/` – Military dataset with foggy, low-light conditions
- `utils/` – Helper scripts for preprocessing, augmentation, and filtering
- `requirements.txt` – Dependencies for training and running the application

## 🛠 Technologies Used
- Python, OpenCV, TensorFlow/Keras, YOLOv5, Tkinter
- Scikit-learn for preprocessing and evaluation
- Flask (if using web app)

## 🔧 Features
- Image preprocessing: grayscale conversion, denoising (median filter), sharpening (high-pass)
- Object classification using CNN + SoftMax activation
- Real-time visualization with GUI
- Easy deployment on laptops, drones, or edge devices

## 🚀 Getting Started
```bash
git clone https://github.com/your-username/adaptive-yolov5-military-detection.git
cd adaptive-yolov5-military-detection
pip install -r requirements.txt
python gui.py
