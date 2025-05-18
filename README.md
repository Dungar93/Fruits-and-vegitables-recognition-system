# Fruits and Vegetables Recognition System

This project is a deep learning-based image classification system that identifies different types of fruits and vegetables from input images. It uses a Convolutional Neural Network (CNN) model trained on labeled image data to predict the correct category of the input object. The system can serve as the foundation for automated sorting in agriculture, smart retail checkouts, or food quality analysis systems.

---

## Project Objectives

The main objectives of this project are:

- To develop a robust and scalable deep learning model capable of classifying fruits and vegetables from real-world images.
- To design a simple pipeline for preprocessing image data, training a CNN model, and evaluating its performance.
- To create a practical use-case prototype that can be extended into a real-time recognition system with a camera interface.

---

## Problem Statement

Manual classification of fruits and vegetables is time-consuming, prone to human error, and inefficient at scale. This project aims to automate this classification using a machine learning model trained on visual features extracted from images of various fruit and vegetable types.

---

## Approach

### 1. Data Collection

- A labeled dataset of fruit and vegetable images is used. The dataset includes images categorized into multiple classes (e.g., apple, banana, carrot, potato, etc.).
- Each image is resized and normalized to maintain consistency and improve training stability.

### 2. Data Preprocessing

- Images are resized (e.g., 64x64 or 128x128) to reduce computational load.
- Normalization and data augmentation (optional) are applied to improve model generalization.

### 3. Model Architecture

- A CNN (Convolutional Neural Network) is implemented using frameworks like Keras or TensorFlow.
- The architecture consists of multiple convolutional layers, pooling layers, dropout layers, and fully connected (dense) layers.
- Softmax activation is used in the output layer for multi-class classification.

### 4. Training

- The model is trained on the training set using cross-entropy loss and an optimizer like Adam or SGD.
- Model evaluation is done using a validation set or test set.

### 5. Prediction

- After training, the model is able to classify new, unseen images into their correct categories.

---

## Installation & How to Run

### Prerequisites

- Python 3.7 or higher
- TensorFlow or Keras
- NumPy, Matplotlib, OpenCV (for image preprocessing and visualization)

### Clone the repository

```bash
git clone https://github.com/Dungar93/Fruits-and-vegitables-recognition-system.git
cd Fruits-and-vegitables-recognition-system
