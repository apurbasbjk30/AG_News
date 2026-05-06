# AG News Text Classification using DistilBERT

## Project Overview

This project implements a complete Natural Language Processing (NLP) pipeline for news text classification using the AG News dataset. The model was developed using PyTorch and Hugging Face Transformers.

The objective of this project was to understand and implement the complete transformer-based NLP workflow, including:

- Data preprocessing
- Tokenization
- Transformer fine-tuning
- Model training
- Evaluation metrics
- Experiment tracking concepts
- Version control using Git and GitHub

---

# Dataset

## Dataset Used
AG News Classification Dataset

The dataset contains four news categories:

1. World  
2. Sports  
3. Business  
4. Sci/Tech  

Each sample contains:
- Title
- Description
- Label

## Dataset Source
https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset

---

# Technologies Used

- Python
- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets
- Scikit-learn
- Pandas
- Matplotlib
- TensorBoard
- Git and GitHub

---

# Project Structure

```plaintext
AG_News/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│
├── outputs/
│
├── src/
│   ├── 01_preprocessing.ipynb
│   └── 02_training.ipynb
│
├── README.md
├── requirements.txt
├── .gitignore
└── report.pdf
````

---

# Data Preprocessing

The preprocessing pipeline included:

1. Loading CSV data using Pandas
2. Combining title and description columns
3. Removing unnecessary columns
4. Cleaning formatting inconsistencies
5. Converting the dataset into Hugging Face Dataset format

The title and description columns were merged because both contained important contextual information about the news articles. Combining them helped the transformer model capture richer semantic meaning from the text.

---

# Tokenization

Tokenization was performed using the pretrained DistilBERT tokenizer.

The tokenizer generated:

* `input_ids`
* `attention_mask`

## input_ids

These are numerical token representations of text generated from the DistilBERT vocabulary. Transformer models cannot process raw text directly, so the tokenizer converts words and subwords into numerical IDs.

## attention_mask

This identifies valid tokens and padded tokens. Tokens with value `1` represent actual text, while `0` represents padded positions. This helps the transformer ignore unnecessary padded regions during attention computation.

Padding and truncation were applied with a maximum sequence length of 128 tokens.

---

# Model Used

## Model

`distilbert-base-uncased`

DistilBERT was selected because it:

* trains faster than BERT
* requires lower computational resources
* performs well on NLP classification tasks
* is suitable for CPU-based experimentation

The pretrained model was fine-tuned for multi-class news classification.

---

# Training Pipeline

The training pipeline was implemented using the Hugging Face Trainer API.

## Training Workflow

1. Tokenized dataset preparation
2. Train-test split
3. Model initialization
4. Training configuration using `TrainingArguments`
5. Fine-tuning using `Trainer`
6. Model evaluation

## Training Configuration

* Epochs: 1
* Batch Size: 8
* Framework: PyTorch + Hugging Face Trainer

A smaller subset of the dataset was used during experimentation to reduce CPU training time.

---

# Evaluation Results

| Metric          | Value |
| --------------- | ----- |
| Accuracy        | 0.896 |
| F1 Score        | 0.895 |
| Precision       | 0.897 |
| Recall          | 0.896 |
| Training Loss   | 0.377 |
| Validation Loss | 0.379 |

---

# Result Analysis

The DistilBERT model achieved approximately 90% classification accuracy on the AG News dataset subset.

The training and validation losses decreased consistently, indicating stable learning behavior and good generalization performance.

The similarity between training loss and validation loss suggests that the model did not significantly overfit during training.

Balanced precision, recall, and F1-score values indicate stable classification performance across all categories.

The project also demonstrated the effectiveness of transfer learning in NLP tasks, where pretrained transformer models can achieve strong performance even with relatively smaller datasets and limited training epochs.

---

# Visualizations

The project includes:

* Loss curve visualization
* Evaluation metric plots
* Confusion matrix
* Class distribution visualization

These visualizations were used to better understand model behavior and training performance.

---

# Challenges Faced

Some implementation challenges encountered during the project included:

* Dependency and library version conflicts
* Hugging Face Trainer compatibility issues
* Dataset formatting problems
* CPU-based slow training
* Tokenization and dataset conversion errors

These issues were resolved through debugging and proper environment configuration.

---

# How to Run

## 1. Install Dependencies

```bash
pip install -r requirements.txt
```

## 2. Run Preprocessing Notebook

```plaintext
src/01_preprocessing.ipynb
```

## 3. Run Training Notebook

```plaintext
src/02_training.ipynb
```

---

# Future Improvements

Possible future improvements include:

* Training on the full dataset
* GPU acceleration
* Hyperparameter tuning
* Comparing DistilBERT with BERT
* Learning rate scheduling
* Advanced experiment tracking using Weights & Biases
* Improved visualization and error analysis

---

# Author

Apurba Das
B.Tech Computer Science and Engineering
Central University of Jharkhand
