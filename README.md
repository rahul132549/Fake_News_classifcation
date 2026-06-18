# Fake-News-Detection
This project implements a Fake News Detection system using an Attention-Augmented Residual Convolutional Neural Network (FNAARCNN). The model classifies news articles as Fake or Real using TF-IDF feature extraction and deep learning techniques.


##TITIE 
# FNAARCNN: Fake News Detection Using an Attention-Augmented Residual Convolutional Neural Network
## Dataset
Dataset Source: Kaggle Fake News Detection Dataset

Classes:
* Fake News
* Real News

Files:
* fake.csv
* true.csv

---
## Project Structure

```text
├── dataset
│   ├── fake.csv
│   └── true.csv
│
├── models
│   └── aarcnn_fake_news_model.keras
│
├── vectorizers
│   ├── tfidf_vectorizer.pkl
│   └── label_encoder.pkl
│
├── notebooks
│   └── FNAARCNN_Fake_News_Detection.ipynb
│
├── results
│   ├── confusion_matrix.png
│   ├── accuracy_curve.png
│   └── loss_curve.png
│
├── prediction
│   └── sample_prediction.py
│
└── README.md
```

---

## Features

* Fake News Classification
* Text Cleaning and Preprocessing
* TF-IDF Feature Extraction
* Deep Learning-Based Detection
* Model Evaluation and Visualization
* Real-Time News Prediction

---

## Workflow

1. Dataset Loading
2. Text Preprocessing
3. Stopword Removal
4. TF-IDF Vectorization
5. Train-Test Split
6. FNAARCNN Model Training
7. Model Evaluation
8. News Prediction

---

## Model Architecture

* Conv1D (128 Filters)

* MaxPooling1D

* Dropout (0.3)

* Conv1D (64 Filters)

* MaxPooling1D

* Dropout (0.3)

* Flatten Layer

* Dense Layer (64 Units)

* Dropout (0.3)

* Sigmoid Output Layer

---

## Requirements

```bash
pip install tensorflow
pip install pandas
pip install numpy
pip install matplotlib
pip install seaborn
pip install scikit-learn
pip install nltk
pip install kagglehub
pip install joblib
```

---

## Training

Run the notebook or Python script to:

* Load the dataset
* Clean and preprocess news articles
* Generate TF-IDF features
* Train the FNAARCNN model
* Evaluate model performance
* Save trained model and preprocessing objects

Saved Files:

```text
aarcnn_fake_news_model.keras
tfidf_vectorizer.pkl
label_encoder.pkl
```
## Results

| Metric     | Value  |
| ---------- | ------ |
| Accuracy   | 99.31% |
| Precision  | 99.12% |
| Recall     | 99.44% |
| F1-Score   | 99.28% |
| ROC-AUC    | 99.97% |
| Error Rate | 0.69%  |

## Prediction
Input:
```text
News Article Text
```
Output:
```text
Fake
```
or
```text
Real
```
## Applications
* Fake News Detection
* News Verification Systems
* Social Media Monitoring
* Content Moderation
* Misinformation Detection
* Digital Journalism

---
## Technologies Used
* Python
* TensorFlow / Keras
* Scikit-Learn
* NLTK
* Pandas
* NumPy
* Matplotlib
* Seaborn

```
```
