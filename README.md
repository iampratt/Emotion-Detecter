# Emotion Detection using Text Mining & NLP (with CNN)

This project presents an end-to-end pipeline for **emotion detection from text** using Natural Language Processing (NLP) techniques and deep learning with a **Convolutional Neural Network (CNN)**. The solution classifies textual data (sentences or phrases) into discrete emotions such as joy, sadness, anger, fear, love, and surprise.

## Table of Contents

- [Project Overview](#project-overview)
- [Objectives](#objectives)
- [Dataset](#dataset)
- [Tools & Libraries](#tools--libraries)
- [Methodology](#methodology)
- [Usage](#usage)
- [Results](#results)
- [License](#license)

---

## Project Overview

This notebook demonstrates how CNNs, typically used for image processing, can be applied to NLP tasks by treating textual data as a sequence suitable for convolutional operations. The workflow includes preprocessing, embedding, model building, training, and evaluation for emotion classification from text.

## Objectives

- Preprocess text (tokenization, padding).
- Use an Embedding layer to convert words into dense vectors.
- Build a CNN-based model to capture local patterns in text.
- Train and evaluate the model on an emotion-labeled dataset.
- Visualize performance and analyze metrics such as accuracy, precision, recall, and F1-score.

## Dataset

- **Source:** [Kaggle - Emotions Dataset for NLP](https://www.kaggle.com/datasets/praveengovi/emotions-dataset-for-nlp)
- **Format:** CSV files with:
  - `text`: input sentence/phrase
  - `emotion`: target label (emotion category)

Sample rows:
```
text,emotion
"I didn't feel humiliated",sadness
"I am feeling grouchy",anger
...
```

## Tools & Libraries

- Python
- TensorFlow / Keras
- NumPy, Pandas
- Matplotlib
- scikit-learn

## Methodology

1. **Data Loading & Exploration**
   - Read and concatenate training/validation/test splits.
   - Analyze emotion distribution, sentence lengths, and frequent words.

2. **Preprocessing**
   - Tokenize text and pad sequences for uniform input length.
   - Encode emotion labels.

3. **Model Architecture**
   - Embedding layer to transform tokens to dense vectors.
   - 1D Convolutional layers to extract local textual patterns.
   - Global max pooling, dense layers, dropout for regularization.
   - Output layer with softmax activation for emotion classification.

4. **Training & Evaluation**
   - Train the model on the train set.
   - Evaluate with metrics (accuracy, precision, recall, F1-score) on the test set.
   - Visualize loss/accuracy curves and confusion matrix.

## Usage

1. **Clone the repository:**
   ```bash
   git clone https://github.com/iampratt/Emotion-Detecter.git
   cd Emotion-Detecter
   ```

2. **Install requirements:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the notebook:**
   - Launch Jupyter Notebook or JupyterLab.
   - Open `Emotion_Detection_using_Text_Mining_&_NLP.ipynb`.
   - Run all cells to reproduce the pipeline and results.

## Results

- The notebook includes visualizations for class distribution, training progress, and performance metrics.
- The trained CNN model achieves competitive accuracy on the test set, demonstrating CNN's effectiveness for text-based emotion classification.

## License

This project is licensed under the MIT License.

---

**References**
- [Kaggle: Emotions Dataset for NLP](https://www.kaggle.com/datasets/praveengovi/emotions-dataset-for-nlp)
- [Text Classification with CNNs](https://arxiv.org/abs/1408.5882)
