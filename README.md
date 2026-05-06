# AG News Text Classification using DistilBERT

## 1. Introduction

The objective of this project was to build a complete Natural Language Processing (NLP) pipeline for text classification using the AG News dataset. The project focused on understanding the end-to-end workflow of transformer-based NLP systems using PyTorch and Hugging Face Transformers.

The AG News dataset contains news articles categorized into four different classes:

1. World
2. Sports
3. Business
4. Sci/Tech

The main goal of this assignment was not only to train a model successfully, but also to understand preprocessing, tokenization, model fine-tuning, evaluation metrics, and training behavior.

---

# 2. Technologies Used

The following tools and libraries were used during the project:

* Python
* PyTorch
* Hugging Face Transformers
* Hugging Face Datasets
* Scikit-learn
* Pandas
* TensorBoard
* Git and GitHub

---

# 3. Dataset Description

The AG News dataset is a benchmark dataset for topic classification.

Dataset characteristics:

* 4 news categories
* Title and description for each article
* CSV format
* Balanced class distribution

Each dataset sample contains:

* Label
* News title
* News description

The title and description columns were combined into a single text field before preprocessing.

---

# 4. Data Preprocessing

The preprocessing stage was performed carefully to make the raw dataset suitable for transformer-based training.

The original AG News dataset contained three columns:

* Class Index
* Title
* Description

During preprocessing, the following steps were performed:

1. The dataset was loaded using Pandas.
2. The title and description columns were merged into a single text column.
3. Unnecessary columns were removed.
4. The dataset was converted into Hugging Face Dataset format.

The title and description columns were combined because both contained important contextual information about the news article. Combining them helped the model capture more semantic meaning from the text.

The original separate columns were removed after merging to avoid redundancy and reduce unnecessary input complexity.

Basic cleaning was also performed on the dataset. Escape characters and formatting inconsistencies from the CSV file were handled automatically during loading and preprocessing. The final dataset contained clean text samples along with their corresponding labels.

After preprocessing, the dataset was tokenized using the DistilBERT tokenizer.

---

# 5. Tokenization

Tokenization was performed using the pretrained DistilBERT tokenizer from Hugging Face.

The tokenizer converted raw text into numerical representations that could be processed by the transformer model.

During tokenization, the following outputs were generated:

* input_ids
* attention_mask

The input_ids represent the numerical token IDs corresponding to words or subwords from the DistilBERT vocabulary. Since transformer models cannot directly process raw text, the tokenizer converts each token into a numerical representation which is later passed through embedding layers inside the model.

The attention_mask helps the model identify which tokens are actual text and which tokens are padding values. Tokens with value 1 represent valid input tokens, while tokens with value 0 represent padded positions added to maintain fixed sequence length. This allows the transformer to ignore unnecessary padded regions during attention computation.

Padding and truncation were applied to ensure consistent input length across all samples.

Maximum sequence length used:

* 128 tokens

This tokenized representation was later used as the direct input to the DistilBERT classification model during training.

---

# 6. Model Selection

The model used for this project was DistilBERT.

Model:

* distilbert-base-uncased

DistilBERT is a lightweight and faster version of BERT. It provides high accuracy while requiring less computational resources.

Reasons for selecting DistilBERT:

* Faster training
* Lower memory consumption
* Suitable for CPU-based training
* Good performance on text classification tasks

The classification layer was fine-tuned on the AG News dataset.

---

# 7. Training Pipeline

The training pipeline was implemented using Hugging Face Trainer API.

Training steps:

1. Tokenized dataset preparation
2. Dataset splitting into training and evaluation sets
3. Model initialization
4. Training configuration using TrainingArguments
5. Fine-tuning using Trainer
6. Evaluation using classification metrics

Training was performed for:

* 1 epoch

A smaller subset of the dataset was used during experimentation to reduce CPU training time.

---

# 8. Evaluation Metrics

The following evaluation metrics were used:

* Accuracy
* Precision
* Recall
* F1-score
* Training loss
* Validation loss

Final model performance:

| Metric          | Value |
| --------------- | ----- |
| Accuracy        | 0.896 |
| F1 Score        | 0.895 |
| Precision       | 0.897 |
| Recall          | 0.896 |
| Training Loss   | 0.377 |
| Validation Loss | 0.379 |

---

# 9. Result Analysis

The model achieved approximately 90% classification accuracy after fine-tuning DistilBERT on the AG News dataset subset.

The complete pipeline was implemented and tested successfully, starting from raw CSV data loading to final model evaluation. Through this implementation, the full workflow of a transformer-based NLP classification system was understood practically.

The training loss and validation loss decreased consistently during training, indicating that the model was learning meaningful semantic patterns from the news articles.

The similarity between training loss and validation loss suggests that the model generalized well and did not significantly overfit on the training subset.

The balanced precision, recall, and F1-score values indicate stable performance across all four news categories.

The experiment also demonstrated the effectiveness of transfer learning in NLP. Even with a relatively small dataset subset and only one training epoch, the pretrained DistilBERT model achieved strong classification performance because the model already contained pretrained language representations learned from large-scale text corpora.

The project also provided practical understanding of:

* transformer tokenization
* attention masks
* dataset formatting
* fine-tuning workflow
* training behavior
* evaluation metrics
* experiment tracking
* version control using Git and GitHub

---

# 10. Challenges Faced

Several practical issues were encountered during implementation:

* Dependency and library version conflicts
* Hugging Face Trainer compatibility issues
* Dataset formatting problems
* CPU-based slow training
* Tokenization and dataset conversion errors

These issues were resolved through debugging, proper dataset formatting, and environment setup corrections.

---

# 11. Conclusion

This project successfully implemented a complete transformer-based NLP text classification pipeline using PyTorch and Hugging Face Transformers.

The project helped in understanding:

* NLP preprocessing
* Tokenization
* Transformer fine-tuning
* Dataset handling
* Training and evaluation workflows
* Experiment tracking concepts
* GitHub-based version control

The final DistilBERT classifier achieved strong performance on the AG News dataset and demonstrated the effectiveness of pretrained transformer models for text classification tasks.

---

# 12. Future Improvements

Possible future improvements include:

* Training on the full dataset
* Using GPU acceleration
* Hyperparameter tuning
* Comparing DistilBERT with BERT
* Implementing learning rate scheduling
* Adding confusion matrix visualization
* Using Weights & Biases for advanced experiment tracking
