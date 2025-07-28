# 🐾 Cat vs Dog Image Classification Using Transfer Learning (2025)
Binary image classification project using transfer learning with MobileNetV2

## 📌 Project Overview
This project focuses on binary image classification using deep learning and transfer learning techniques. The objective is to classify images as either Cat or Dog using a pre-trained MobileNetV2 model. The goal is to demonstrate how transfer learning can achieve high accuracy even with a limited dataset, while also integrating essential computer vision techniques like image preprocessing and evaluation metrics.

## ⚙️ Model & Approach
The model architecture is based on MobileNetV2 from TensorFlow Keras applications, pre-trained on ImageNet. This base model was frozen during training, and custom dense layers were added on top for binary classification.

🔧 Model Components:

✅ Pretrained MobileNetV2 (feature extractor)

✅ GlobalAveragePooling2D

✅ Dense layer with ReLU activation

✅ Dropout (to reduce overfitting)

✅ Dense output layer with softmax activation (2 classes)

📦 Loss Function: sparse_categorical_crossentropy          
🚀 Optimizer: Adam                     
🎯 Metrics: Accuracy                      

## 📈 Evaluation Metrics
The model achieved outstanding performance across all standard classification metrics. When evaluated on the official test set consisting of 5,000 images (2,500 cats and 2,500 dogs), it reached an overall accuracy of 97%, with balanced precision, recall, and F1-scores of 0.97 for both classes. 
Furthermore, to assess its generalization ability, the model was also tested on unseen external images fethched from the web .

## 📊 Dataset

The dataset consists of a total of 25,000 labeled images of cats and dogs:

train/cats: 10,000 images

train/dogs: 10,000 images

test/cats: 2,500 images

test/dogs: 2,500 images

In this project, a subset of 5,000 images (2,500 per class) from the test folder was used as the final test set. 
The training data (20,000 images total) was further split into:

Training set: 4,000 images (stratified sample)

Validation set: 1,000 images

All images were resized to 224x224, normalized, and augmented during training to improve generalization.

## 📁Downloading Dataset
Kaggle Dataset URL: https://www.kaggle.com/c/dogs-vs-cats


## 👩‍💻 Contact
📌 Project by: Nazanin Mahmoudy, 2025                
📧 Email: Nazaninmahmoudy@gmail.com                    
🔗 GitHub: https://github.com/Nazaninmahmoudi                    
🔗 Kaggle: https://www.kaggle.com/nazaninmahmoudy                          

