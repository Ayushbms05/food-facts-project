# Model Experiment Results

This document records the experiments conducted for the Food Health Prediction project. It includes traditional Machine Learning (ML) and Deep Learning (DL) models so that their performance can be compared.

---

# Experiment 1: TF-IDF + Logistic Regression

## 1. Experiment Information

- **Category:** Traditional Machine Learning (ML)
- **Model:** TF-IDF + Logistic Regression
- **Status:** Completed
- **Purpose:** Establish a baseline performance against which future Deep Learning models can be compared.
- **Prediction task:** Predict a food product's NOVA group (1, 2, 3 or 4) using its ingredient text.

## 2. Dataset Information

The experiment used the prepared English-language dataset containing 10,000 products.

| Dataset    | Number of products |
| ---------- | -----------------: |
| Training   |              7,018 |
| Validation |              1,481 |
| Testing    |              1,501 |
| **Total**  |         **10,000** |

## 3. Model Configuration

**TF-IDF Vectorizer:**

- Converts ingredient text into numerical features.
- Uses single words and two-word combinations (unigrams and bigrams).
- Maximum features: 100,000.
- Minimum document frequency: 2.

**Logistic Regression:**

- Maximum iterations: 1,000.
- Random state: 42.

## 4. Overall Performance

| Metric         | Validation |   Test |
| -------------- | ---------: | -----: |
| Accuracy       |     82.58% | 82.01% |
| Macro F1-score |     0.8282 | 0.8188 |

## 5. Test Classification Report

| NOVA Group | Precision | Recall | F1-score | Support |
| ---------- | --------: | -----: | -------: | ------: |
| NOVA 1     |      0.86 |   0.94 |     0.90 |     383 |
| NOVA 2     |      0.92 |   0.62 |     0.74 |     373 |
| NOVA 3     |      0.64 |   0.84 |     0.72 |     365 |
| NOVA 4     |      0.95 |   0.88 |     0.91 |     380 |

## 6. Confusion Matrix

Rows represent actual NOVA groups, while columns represent predicted NOVA groups.

| Actual / Predicted | NOVA 1 | NOVA 2 | NOVA 3 | NOVA 4 |
| ------------------ | -----: | -----: | -----: | -----: |
| NOVA 1             |    360 |      4 |     14 |      5 |
| NOVA 2             |     16 |    230 |    124 |      3 |
| NOVA 3             |     36 |     13 |    305 |     11 |
| NOVA 4             |      7 |      2 |     35 |    336 |

## 7. Key Observations

1. The model achieved **82.01% test accuracy**.
2. NOVA 1 and NOVA 4 achieved the strongest F1-scores.
3. NOVA 2 was frequently misclassified as NOVA 3.
4. Out of 373 actual NOVA 2 products, 230 were correctly classified.
5. Validation and test performance were relatively close.

## 8. Conclusion

The TF-IDF + Logistic Regression model provides our first benchmark for the project.

Although it is a traditional Machine Learning model and not a Deep Learning model, its results will help us evaluate whether more advanced architectures improve prediction performance.

---

# Experiment 2: 1D CNN

## 1. Experiment Information

- **Category:** Deep Learning (DL)
- **Model:** 1D Convolutional Neural Network (CNN)
- **Framework:** PyTorch
- **Status:** Completed
- **Device:** Apple MPS (Apple Silicon GPU)
- **Purpose:** Learn patterns and relationships in ingredient text using convolutional layers.
- **Prediction task:** Predict a food product's NOVA group (1, 2, 3 or 4) using its ingredient text.

## 2. Dataset Information

The experiment used the same prepared English-language dataset and the same training, validation and testing splits as Experiment 1.

| Dataset    | Number of products |
| ---------- | -----------------: |
| Training   |              7,018 |
| Validation |              1,481 |
| Testing    |              1,501 |
| **Total**  |         **10,000** |

Vocabulary size: 3,695 words.

Maximum sequence length: 256 tokens.

## 3. Model Configuration

**Text Processing:**

- Converts ingredient text into tokens.
- Builds a vocabulary using training data.
- Converts words into numerical token IDs.
- Uses an embedding layer to represent words as vectors.
- Embedding dimension: 128.
- Maximum vocabulary size: 30,000.

**1D CNN Architecture:**

- Three convolutional layers with kernel sizes 3, 5 and 7.
- 128 output channels per convolution.
- ReLU activation function.
- Global max pooling to extract important features.
- Dropout rate: 0.4.
- Fully connected output layer with four classes.

**Training Configuration:**

- Optimizer: AdamW.
- Learning rate: 0.001.
- Batch size: 64.
- Maximum epochs: 12.
- Early stopping patience: 3.
- Random seed: 42.
- Best model selected using validation macro F1-score.

## 4. Overall Performance

| Metric         | Validation |   Test |
| -------------- | ---------: | -----: |
| Accuracy       |     89.06% | 88.94% |
| Macro F1-score |     0.8904 | 0.8889 |

## 5. Test Classification Report

| NOVA Group | Precision | Recall | F1-score | Support |
| ---------- | --------: | -----: | -------: | ------: |
| NOVA 1     |      0.89 |   0.87 |     0.88 |     383 |
| NOVA 2     |      0.87 |   0.91 |     0.89 |     373 |
| NOVA 3     |      0.86 |   0.84 |     0.85 |     365 |
| NOVA 4     |      0.94 |   0.93 |     0.94 |     380 |

## 6. Confusion Matrix

Rows represent actual NOVA groups, while columns represent predicted NOVA groups.

| Actual / Predicted | NOVA 1 | NOVA 2 | NOVA 3 | NOVA 4 |
| ------------------ | -----: | -----: | -----: | -----: |
| NOVA 1             |    335 |     28 |     14 |      6 |
| NOVA 2             |     18 |    340 |     12 |      3 |
| NOVA 3             |     22 |     23 |    305 |     15 |
| NOVA 4             |      2 |      1 |     22 |    355 |

## 7. Key Observations

1. The model achieved **88.94% test accuracy**.
2. NOVA 4 achieved the strongest F1-score of 0.94.
3. NOVA 2 achieved an F1-score of 0.89, showing a substantial improvement over the Logistic Regression baseline.
4. NOVA 3 remained the most challenging class, with an F1-score of 0.85.
5. The CNN achieved 1,335 correct predictions out of 1,501 test products.
6. Training accuracy reached 97.45%, while the best validation accuracy was 89.06%, suggesting some overfitting.
7. Early stopping helped select the best-performing validation checkpoint.

## 8. Conclusion

The 1D CNN outperformed the TF-IDF + Logistic Regression baseline on the same test dataset.

Test accuracy increased from **82.01% to 88.94%**, while macro F1-score increased from **0.8188 to 0.8889**.

The CNN demonstrated that learning word and phrase patterns through convolutional layers can improve NOVA classification performance.

However, these results are based on the current balanced English-language dataset. Further evaluation on naturally distributed and external product data is required.

---

# 3. Model Performance Comparison

| Model                        | Category       | Test Accuracy | Test Macro F1 |
| ---------------------------- | -------------- | ------------: | ------------: |
| TF-IDF + Logistic Regression | Traditional ML |        82.01% |        0.8188 |
| 1D CNN                       | Deep Learning  |    **88.94%** |    **0.8889** |

---

# Future Experiments

| Experiment | Model                        | Category       | Status    |
| ---------- | ---------------------------- | -------------- | --------- |
| 1          | TF-IDF + Logistic Regression | Traditional ML | Completed |
| 2          | 1D CNN                       | Deep Learning  | Completed |
| 3          | LSTM                         | Deep Learning  | Pending   |
| 4          | Transformer                  | Deep Learning  | Pending   |

_Future experiment results will be added to this document after each model has been trained and evaluated._

# Experiment 3: Bidirectional LSTM

## 1. Experiment Information

- **Category:** Deep Learning (DL)
- **Model:** Bidirectional Long Short-Term Memory (LSTM)
- **Framework:** PyTorch
- **Status:** Completed
- **Device:** Apple MPS (Apple Silicon GPU)
- **Purpose:** Learn sequential patterns and contextual relationships in ingredient text by processing the text in both forward and backward directions.
- **Prediction task:** Predict a food product's NOVA group (1, 2, 3 or 4) using its ingredient text.

## 2. Dataset Information

The experiment used the same prepared English-language dataset and the same training, validation and testing splits as the previous experiments.

| Dataset    | Number of products |
| ---------- | -----------------: |
| Training   |              7,018 |
| Validation |              1,481 |
| Testing    |              1,501 |
| **Total**  |         **10,000** |

Vocabulary size: 3,695 words.

Maximum sequence length: 256 tokens.

## 3. Model Configuration

**Text Processing:**

- Converts ingredient text into tokens.
- Builds a vocabulary using training data.
- Converts words into numerical token IDs.
- Uses an embedding layer to represent words as vectors.
- Embedding dimension: 128.
- Maximum vocabulary size: 30,000.

**Bidirectional LSTM Architecture:**

- Forward LSTM processes tokens from beginning to end.
- Backward LSTM processes tokens in reverse order.
- Hidden size: 128 per direction.
- Combines the representations from both directions.
- Dropout rate: 0.4.
- Fully connected output layer with four classes.

**Training Configuration:**

- Optimizer: AdamW.
- Learning rate: 0.001.
- Batch size: 32.
- Maximum epochs: 12.
- Early stopping patience: 3.
- Random seed: 42.
- Best model selected using validation macro F1-score.

## 4. Overall Performance

| Metric         | Validation |   Test |
| -------------- | ---------: | -----: |
| Accuracy       |     87.64% | 89.81% |
| Macro F1-score |     0.8761 | 0.8977 |

## 5. Test Classification Report

| NOVA Group | Precision | Recall | F1-score | Support |
| ---------- | --------: | -----: | -------: | ------: |
| NOVA 1     |      0.89 |   0.93 |     0.91 |     383 |
| NOVA 2     |      0.94 |   0.89 |     0.91 |     373 |
| NOVA 3     |      0.83 |   0.84 |     0.84 |     365 |
| NOVA 4     |      0.94 |   0.92 |     0.93 |     380 |

## 6. Confusion Matrix

Rows represent actual NOVA groups, while columns represent predicted NOVA groups.

| Actual / Predicted | NOVA 1 | NOVA 2 | NOVA 3 | NOVA 4 |
| ------------------ | -----: | -----: | -----: | -----: |
| NOVA 1             |    358 |      5 |     16 |      4 |
| NOVA 2             |     19 |    333 |     20 |      1 |
| NOVA 3             |     23 |     17 |    307 |     18 |
| NOVA 4             |      3 |      0 |     27 |    350 |

## 7. Key Observations

1. The model achieved **89.81% test accuracy**.
2. NOVA 1 and NOVA 2 both achieved an F1-score of 0.91.
3. NOVA 4 achieved an F1-score of 0.93.
4. NOVA 3 remained the most challenging class, with an F1-score of 0.84.
5. The LSTM achieved 1,348 correct predictions out of 1,501 test products.
6. Training accuracy reached 97.89%, while the best validation accuracy was 87.64%, suggesting some overfitting.
7. Early stopping selected the checkpoint with the best validation macro F1-score.

## 8. Conclusion

The Bidirectional LSTM achieved the best performance among the three models evaluated so far.

It achieved **89.81% test accuracy** and a **0.8977 macro F1-score**, outperforming both TF-IDF + Logistic Regression and the 1D CNN on the same test split.

The results indicate that sequential processing of ingredient text can be useful for NOVA classification. However, NOVA 3 remains comparatively challenging, and further evaluation on naturally distributed and external product data is required.

---

# 4. Updated Model Performance Comparison

| Model                        | Category       | Test Accuracy | Test Macro F1 |
| ---------------------------- | -------------- | ------------: | ------------: |
| TF-IDF + Logistic Regression | Traditional ML |        82.01% |        0.8188 |
| 1D CNN                       | Deep Learning  |        88.94% |        0.8889 |
| Bidirectional LSTM           | Deep Learning  |    **89.81%** |    **0.8977** |

---

# Future Experiments

| Experiment | Model                        | Category       | Status    |
| ---------- | ---------------------------- | -------------- | --------- |
| 1          | TF-IDF + Logistic Regression | Traditional ML | Completed |
| 2          | 1D CNN                       | Deep Learning  | Completed |
| 3          | Bidirectional LSTM           | Deep Learning  | Completed |
| 4          | Transformer                  | Deep Learning  | Pending   |

_Future experiment results will be added to this document after each model has been trained and evaluated._
