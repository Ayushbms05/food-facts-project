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

Future CNN, LSTM and Transformer models will be evaluated using the same training, validation and test splits.

---

# Future Experiments

| Experiment | Model                        | Category       | Status    |
| ---------- | ---------------------------- | -------------- | --------- |
| 1          | TF-IDF + Logistic Regression | Traditional ML | Completed |
| 2          | 1D CNN                       | Deep Learning  | Pending   |
| 3          | LSTM                         | Deep Learning  | Pending   |
| 4          | Transformer                  | Deep Learning  | Pending   |

_Future experiment results will be added to this document after each model has been trained and evaluated._
