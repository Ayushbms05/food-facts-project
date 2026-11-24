# MODEL EXPERIMENT RESULTS

## Food Health Prediction Project — Machine Learning and Deep Learning

**Last Updated:** 24 November 2026
**Task:** Predict a food product's NOVA processing group (1–4) using ingredient text.
**Primary Dataset:** Open Food Facts-derived `food.parquet`
**Initial Language Scope:** English
**Current Leading Model:** DistilBERT

---

# 1. EXPERIMENT OVERVIEW

The objective of these experiments was to compare traditional Machine Learning and Deep Learning approaches for predicting the NOVA processing group of food products from their ingredient descriptions.

Four models were implemented and evaluated:

1. TF-IDF + Logistic Regression
2. 1D Convolutional Neural Network (CNN)
3. Bidirectional Long Short-Term Memory (BiLSTM)
4. DistilBERT Transformer

All four experiments used the same training, validation and test partitions to make their initial performance comparison consistent.

After selecting the best-performing model, an additional natural-distribution evaluation was conducted to investigate its performance under a more realistic class distribution.

## 1.1 Dataset summary

The initial experiments used a balanced sample of 10,000 English-language products.

| Partition  | Number of products |
| ---------- | -----------------: |
| Training   |              7,018 |
| Validation |              1,481 |
| Testing    |              1,501 |
| **Total**  |         **10,000** |

Class distribution in the balanced sample:

| NOVA group | Training | Validation | Test |
| ---------- | -------: | ---------: | ---: |
| NOVA 1     |    1,742 |        375 |  383 |
| NOVA 2     |    1,758 |        369 |  373 |
| NOVA 3     |    1,767 |        368 |  365 |
| NOVA 4     |    1,751 |        369 |  380 |

Normalized ingredient descriptions were grouped during splitting to prevent identical normalized descriptions from appearing in different partitions.

The reported overlap checks were zero across training, validation and test partitions.

**Important:** These are balanced experimental partitions, not a representation of the natural food-product population.

---

# 2. EXPERIMENT 1 — TF-IDF + LOGISTIC REGRESSION

## 2.1 Experiment information

| Property        | Details                             |
| --------------- | ----------------------------------- |
| Category        | Traditional Machine Learning        |
| Model           | TF-IDF + Logistic Regression        |
| Status          | Completed                           |
| Purpose         | Establish a baseline for comparison |
| Prediction task | Four-class NOVA classification      |

Script:

```text
src/deep_learning/models/train_baseline.py
```

## 2.2 Model configuration

**TF-IDF vectorizer**

- Converts ingredient text into numerical features.
- Uses unigrams and bigrams.
- Maximum features: 100,000.
- Minimum document frequency: 2.
- Lowercase text enabled.
- Sublinear term frequency enabled.

**Logistic Regression**

- Maximum iterations: 1,000.
- Random state: 42.

## 2.3 Overall performance

| Metric         | Validation |   Test |
| -------------- | ---------: | -----: |
| Accuracy       |     82.58% | 82.01% |
| Macro F1-score |     0.8282 | 0.8188 |

## 2.4 Test classification report

| NOVA Group | Precision | Recall | F1-score | Support |
| ---------- | --------: | -----: | -------: | ------: |
| NOVA 1     |      0.86 |   0.94 |     0.90 |     383 |
| NOVA 2     |      0.92 |   0.62 |     0.74 |     373 |
| NOVA 3     |      0.64 |   0.84 |     0.72 |     365 |
| NOVA 4     |      0.95 |   0.88 |     0.91 |     380 |

## 2.5 Confusion matrix

Rows represent actual classes and columns represent predicted classes.

| Actual / Predicted | NOVA 1 | NOVA 2 | NOVA 3 | NOVA 4 |
| ------------------ | -----: | -----: | -----: | -----: |
| NOVA 1             |    360 |      4 |     14 |      5 |
| NOVA 2             |     16 |    230 |    124 |      3 |
| NOVA 3             |     36 |     13 |    305 |     11 |
| NOVA 4             |      7 |      2 |     35 |    336 |

## 2.6 Key observations

1. The model achieved 82.01% test accuracy.
2. NOVA 1 and NOVA 4 achieved the strongest F1-scores.
3. NOVA 2 was frequently misclassified as NOVA 3.
4. Only 230 of 373 actual NOVA 2 products were correctly classified.
5. Validation and test performance were relatively close.

## 2.7 Conclusion

TF-IDF + Logistic Regression established the traditional Machine Learning baseline.

It demonstrated that ingredient text contains useful patterns for predicting NOVA groups, but its performance left room for improvement.

This baseline provided a reference for evaluating the more advanced neural models.

---

# 3. EXPERIMENT 2 — 1D CONVOLUTIONAL NEURAL NETWORK

## 3.1 Experiment information

| Property  | Details                                                 |
| --------- | ------------------------------------------------------- |
| Category  | Deep Learning                                           |
| Model     | 1D CNN                                                  |
| Framework | PyTorch                                                 |
| Device    | Apple MPS                                               |
| Status    | Completed                                               |
| Purpose   | Learn local word and phrase patterns in ingredient text |

Script:

```text
src/deep_learning/models/train_cnn.py
```

## 3.2 Dataset information

The same balanced dataset and train/validation/test partitions were used.

| Property                |      Value |
| ----------------------- | ---------: |
| Training products       |      7,018 |
| Validation products     |      1,481 |
| Test products           |      1,501 |
| Vocabulary size         |      3,695 |
| Maximum sequence length | 256 tokens |

## 3.3 Architecture

The CNN architecture consists of:

1. Text tokenization.
2. Vocabulary construction.
3. Embedding layer.
4. Three parallel 1D convolution branches.
5. ReLU activation.
6. Global max pooling.
7. Dropout.
8. Fully connected output layer with four NOVA classes.

The convolution branches use different kernel sizes to learn patterns of different lengths.

## 3.4 Model configuration

| Setting                 |   Value |
| ----------------------- | ------: |
| Embedding dimension     |     128 |
| Maximum vocabulary size |  30,000 |
| Convolution channels    |     128 |
| Kernel sizes            | 3, 5, 7 |
| Dropout                 |     0.4 |
| Batch size              |      64 |
| Learning rate           |   0.001 |
| Optimizer               |   AdamW |
| Maximum epochs          |      12 |
| Early stopping patience |       3 |
| Random seed             |      42 |

The best model was selected using validation macro F1-score.

## 3.5 Overall performance

| Metric         | Validation |   Test |
| -------------- | ---------: | -----: |
| Accuracy       |     89.06% | 88.94% |
| Macro F1-score |     0.8904 | 0.8889 |

## 3.6 Test classification report

| NOVA Group | Precision | Recall | F1-score | Support |
| ---------- | --------: | -----: | -------: | ------: |
| NOVA 1     |      0.89 |   0.87 |     0.88 |     383 |
| NOVA 2     |      0.87 |   0.91 |     0.89 |     373 |
| NOVA 3     |      0.86 |   0.84 |     0.85 |     365 |
| NOVA 4     |      0.94 |   0.93 |     0.94 |     380 |

## 3.7 Confusion matrix

| Actual / Predicted | NOVA 1 | NOVA 2 | NOVA 3 | NOVA 4 |
| ------------------ | -----: | -----: | -----: | -----: |
| NOVA 1             |    335 |     28 |     14 |      6 |
| NOVA 2             |     18 |    340 |     12 |      3 |
| NOVA 3             |     22 |     23 |    305 |     15 |
| NOVA 4             |      2 |      1 |     22 |    355 |

## 3.8 Key observations

1. The model achieved 88.94% test accuracy.
2. It correctly classified 1,335 of 1,501 test products.
3. NOVA 4 achieved the strongest F1-score of 0.94.
4. NOVA 2 achieved an F1-score of 0.89.
5. NOVA 3 remained comparatively challenging, with an F1-score of 0.85.
6. Training accuracy reached 97.45%, while validation accuracy was 89.06%, suggesting some overfitting.
7. Early stopping helped select the best validation checkpoint.

## 3.9 Conclusion

The 1D CNN outperformed the Logistic Regression baseline.

Test accuracy improved from 82.01% to 88.94%, while macro F1 improved from 0.8188 to 0.8889.

The experiment demonstrated that convolutional layers can learn useful local patterns in ingredient descriptions.

---

# 4. EXPERIMENT 3 — BIDIRECTIONAL LSTM

## 4.1 Experiment information

| Property  | Details                                                          |
| --------- | ---------------------------------------------------------------- |
| Category  | Deep Learning                                                    |
| Model     | Bidirectional LSTM                                               |
| Framework | PyTorch                                                          |
| Device    | Apple MPS                                                        |
| Status    | Completed                                                        |
| Purpose   | Learn sequential and contextual relationships in ingredient text |

Script:

```text
src/deep_learning/models/train_lstm.py
```

## 4.2 Dataset information

The same 10,000-product balanced dataset and partitions were used.

| Property                |      Value |
| ----------------------- | ---------: |
| Training products       |      7,018 |
| Validation products     |      1,481 |
| Test products           |      1,501 |
| Vocabulary size         |      3,695 |
| Maximum sequence length | 256 tokens |

## 4.3 Architecture

The BiLSTM architecture consists of:

1. Text tokenization.
2. Vocabulary construction.
3. Embedding layer.
4. Forward LSTM.
5. Backward LSTM.
6. Combination of forward and backward representations.
7. Dropout.
8. Fully connected four-class output layer.

The forward LSTM processes ingredient tokens from beginning to end.

The backward LSTM processes the same sequence in reverse.

Combining both representations allows the model to learn context from both directions.

## 4.4 Model configuration

| Setting                   |  Value |
| ------------------------- | -----: |
| Embedding dimension       |    128 |
| Hidden size per direction |    128 |
| Maximum vocabulary size   | 30,000 |
| Dropout                   |    0.4 |
| Batch size                |     32 |
| Learning rate             |  0.001 |
| Optimizer                 |  AdamW |
| Maximum epochs            |     12 |
| Early stopping patience   |      3 |
| Random seed               |     42 |

The best model was selected using validation macro F1-score.

## 4.5 Overall performance

| Metric         | Validation |   Test |
| -------------- | ---------: | -----: |
| Accuracy       |     87.64% | 89.81% |
| Macro F1-score |     0.8761 | 0.8977 |

## 4.6 Test classification report

| NOVA Group | Precision | Recall | F1-score | Support |
| ---------- | --------: | -----: | -------: | ------: |
| NOVA 1     |      0.89 |   0.93 |     0.91 |     383 |
| NOVA 2     |      0.94 |   0.89 |     0.91 |     373 |
| NOVA 3     |      0.83 |   0.84 |     0.84 |     365 |
| NOVA 4     |      0.94 |   0.92 |     0.93 |     380 |

## 4.7 Confusion matrix

| Actual / Predicted | NOVA 1 | NOVA 2 | NOVA 3 | NOVA 4 |
| ------------------ | -----: | -----: | -----: | -----: |
| NOVA 1             |    358 |      5 |     16 |      4 |
| NOVA 2             |     19 |    333 |     20 |      1 |
| NOVA 3             |     23 |     17 |    307 |     18 |
| NOVA 4             |      3 |      0 |     27 |    350 |

## 4.8 Key observations

1. The model achieved 89.81% test accuracy.
2. It correctly classified 1,348 of 1,501 test products.
3. NOVA 1 and NOVA 2 both achieved F1-scores of 0.91.
4. NOVA 4 achieved an F1-score of 0.93.
5. NOVA 3 remained the most challenging class, with an F1-score of 0.84.
6. Training accuracy reached 97.89%, suggesting a gap between training and validation performance.
7. Early stopping selected the checkpoint with the best validation macro F1-score.

## 4.9 Conclusion

The Bidirectional LSTM outperformed the Logistic Regression and CNN models on the balanced test set.

Its test accuracy was 89.81%, and its macro F1-score was 0.8977.

The experiment demonstrated that processing ingredient sequences in both directions can be useful for contextual text classification.

---

# 5. EXPERIMENT 4 — DISTILBERT TRANSFORMER

## 5.1 Experiment information

| Property  | Details                                                                    |
| --------- | -------------------------------------------------------------------------- |
| Category  | Deep Learning                                                              |
| Model     | DistilBERT                                                                 |
| Framework | PyTorch and Hugging Face Transformers                                      |
| Device    | Apple MPS                                                                  |
| Status    | Completed                                                                  |
| Purpose   | Use pretrained contextual language representations for NOVA classification |

Script:

```text
src/deep_learning/models/train_transformer.py
```

Pretrained model:

```text
distilbert/distilbert-base-uncased
```

## 5.2 Architecture

The DistilBERT architecture consists of:

1. DistilBERT tokenizer.
2. Token IDs and attention masks.
3. Pretrained DistilBERT encoder.
4. Classification head.
5. Four output classes representing NOVA 1–4.

DistilBERT is a smaller pretrained Transformer derived from BERT.

It uses self-attention to learn contextual relationships between words.

The pretrained language model was fine-tuned for the four-class classification task.

## 5.3 Dataset information

The same balanced dataset and partitions were used.

| Partition  |   Products |
| ---------- | ---------: |
| Training   |      7,018 |
| Validation |      1,481 |
| Test       |      1,501 |
| **Total**  | **10,000** |

## 5.4 Model configuration

| Setting                     |                   Value |
| --------------------------- | ----------------------: |
| Pretrained model            | DistilBERT base uncased |
| Maximum token length        |                     256 |
| Batch size                  |                       8 |
| Gradient accumulation steps |                       2 |
| Effective batch size        |                      16 |
| Learning rate               |                 0.00002 |
| Optimizer                   |                   AdamW |
| Weight decay                |                    0.01 |
| Maximum epochs              |                       3 |
| Early stopping patience     |                       2 |
| Random seed                 |                      42 |

The best checkpoint was selected using validation macro F1-score.

## 5.5 Training progress

| Epoch | Training accuracy | Training macro F1 | Validation accuracy | Validation macro F1 |
| ----- | ----------------: | ----------------: | ------------------: | ------------------: |
| 1     |            80.01% |            0.8003 |              90.07% |              0.9003 |
| 2     |            90.71% |            0.9072 |              90.75% |              0.9071 |
| 3     |            93.10% |            0.9312 |              90.75% |              0.9078 |

Epoch 3 achieved the highest validation macro F1-score and its checkpoint was selected for final testing.

## 5.6 Overall performance

| Metric         | Validation |   Test |
| -------------- | ---------: | -----: |
| Accuracy       |     90.75% | 91.34% |
| Macro F1-score |     0.9078 | 0.9132 |

## 5.7 Test classification report

| NOVA Group | Precision | Recall | F1-score | Support |
| ---------- | --------: | -----: | -------: | ------: |
| NOVA 1     |      0.92 |   0.94 |     0.93 |     383 |
| NOVA 2     |      0.87 |   0.93 |     0.90 |     373 |
| NOVA 3     |      0.87 |   0.85 |     0.86 |     365 |
| NOVA 4     |      0.99 |   0.92 |     0.96 |     380 |

## 5.8 Confusion matrix

| Actual / Predicted | NOVA 1 | NOVA 2 | NOVA 3 | NOVA 4 |
| ------------------ | -----: | -----: | -----: | -----: |
| NOVA 1             |    360 |     15 |      8 |      0 |
| NOVA 2             |      9 |    348 |     16 |      0 |
| NOVA 3             |     19 |     32 |    312 |      2 |
| NOVA 4             |      2 |      6 |     21 |    351 |

## 5.9 Key observations

1. The model achieved 91.34% test accuracy.
2. Test macro F1-score was 0.9132.
3. It correctly classified 1,371 of 1,501 test products.
4. NOVA 4 achieved precision of 0.99 and an F1-score of 0.96.
5. NOVA 1 achieved an F1-score of 0.93.
6. NOVA 2 achieved an F1-score of 0.90.
7. NOVA 3 remained the most challenging class, with an F1-score of 0.86.
8. The third epoch achieved the best validation macro F1-score.

## 5.10 Conclusion

DistilBERT achieved the strongest overall performance among all four models.

It outperformed the traditional ML baseline, CNN and BiLSTM on the same balanced test set.

The experiment demonstrated the value of pretrained Transformer representations for contextual ingredient-text classification.

---

# 6. FINAL BALANCED-TEST MODEL COMPARISON

All four models were evaluated on the same 1,501-product test partition.

| Model                        | Category          | Test Accuracy | Test Macro F1 |
| ---------------------------- | ----------------- | ------------: | ------------: |
| TF-IDF + Logistic Regression | Traditional ML    |        82.01% |        0.8188 |
| 1D CNN                       | Deep Learning     |        88.94% |        0.8889 |
| Bidirectional LSTM           | Deep Learning     |        89.81% |        0.8977 |
| **DistilBERT**               | **Deep Learning** |    **91.34%** |    **0.9132** |

## 6.1 Model ranking

1. DistilBERT — 91.34%.
2. Bidirectional LSTM — 89.81%.
3. 1D CNN — 88.94%.
4. TF-IDF + Logistic Regression — 82.01%.

DistilBERT improved test accuracy by:

- 9.33 percentage points over Logistic Regression.
- 2.40 percentage points over CNN.
- 1.53 percentage points over BiLSTM.

## 6.2 Model selection decision

**DistilBERT is the leading model candidate based on the balanced test-set accuracy and macro F1-score.**

It was selected for the reusable prediction interface.

However, the balanced-test leaderboard should not be treated as a complete measure of real-world performance.

A separate natural-distribution evaluation is necessary because the original dataset is highly imbalanced.

---

# 7. NATURAL-DISTRIBUTION EVALUATION

## 7.1 Purpose

The initial experiments used a balanced dataset with 2,500 products from each NOVA group.

However, the full eligible English dataset has a substantially different class distribution.

Therefore, an additional evaluation was performed using a 10,000-product sample that approximates the natural distribution.

This helps investigate whether the selected model performs similarly when NOVA 4 is much more common and NOVA 2 is rare.

## 7.2 Natural sample preparation

Script:

```text
src/deep_learning/data_preparation/prepare_natural_sample.py
```

Output:

```text
data/deep_learning/datasets/natural_distribution_sample.csv
```

The preparation process:

1. Scanned the original Parquet dataset.
2. Filtered English records.
3. Retained valid NOVA labels and nonempty ingredient text.
4. Excluded normalized descriptions already found in the balanced partitions.
5. Used reservoir sampling with random seed 42.
6. Selected 10,000 records.

Reported results:

| Measure                           |    Result |
| --------------------------------- | --------: |
| Original source rows              | 4,895,629 |
| Eligible records after exclusions |   485,831 |
| Excluded products                 |    54,683 |
| Sample size                       |    10,000 |

## 7.3 Natural sample class distribution

| NOVA group |   Products | Percentage |
| ---------- | ---------: | ---------: |
| NOVA 1     |        562 |      5.62% |
| NOVA 2     |         35 |      0.35% |
| NOVA 3     |      1,615 |     16.15% |
| NOVA 4     |      7,788 |     77.88% |
| **Total**  | **10,000** |   **100%** |

The sample reflects the strong imbalance of the eligible English dataset.

## 7.4 Evaluation script

```text
src/deep_learning/evaluation/evaluate_natural_distribution.py
```

Predictions were generated using the saved DistilBERT model.

Output:

```text
data/deep_learning/reports/natural_distribution_predictions.csv
```

## 7.5 Overall results

| Metric                | Result |
| --------------------- | -----: |
| Total products        | 10,000 |
| Correct predictions   |  9,011 |
| Incorrect predictions |    989 |
| Accuracy              | 90.11% |
| Macro F1              | 0.6933 |

## 7.6 Class-wise results

| NOVA Group | Precision | Recall | F1-score | Support |
| ---------- | --------: | -----: | -------: | ------: |
| NOVA 1     |      0.77 |   0.86 |     0.82 |     562 |
| NOVA 2     |      0.13 |   0.83 |     0.22 |      35 |
| NOVA 3     |      0.70 |   0.90 |     0.79 |   1,615 |
| NOVA 4     |      1.00 |   0.90 |     0.95 |   7,788 |

## 7.7 Confusion matrix

| Actual / Predicted | NOVA 1 | NOVA 2 | NOVA 3 | NOVA 4 |
| ------------------ | -----: | -----: | -----: | -----: |
| NOVA 1             |    486 |     38 |     36 |      2 |
| NOVA 2             |      2 |     29 |      4 |      0 |
| NOVA 3             |     59 |     82 |  1,453 |     21 |
| NOVA 4             |     82 |     77 |    586 |  7,043 |

## 7.8 Interpretation

Natural-distribution accuracy was 90.11%, but macro F1 fell to 0.6933.

This difference is important because the dataset contains a large proportion of NOVA 4 products.

The model performs strongly on NOVA 4, which contributes significantly to overall accuracy.

However, NOVA 2 has very low precision.

Only 29 of the 226 products predicted as NOVA 2 were actually NOVA 2.

There were only 35 actual NOVA 2 records in the natural sample, so its class-wise metrics should be interpreted cautiously.

**Conclusion:** Overall accuracy alone is not sufficient to establish reliable real-world performance. Class-wise performance, especially for underrepresented NOVA groups, must also be considered.

---

# 8. NATURAL-DISTRIBUTION ERROR ANALYSIS

## 8.1 Purpose

After the natural-distribution evaluation, an additional investigation was conducted to understand the model's incorrect predictions.

Script:

```text
src/deep_learning/evaluation/analyze_natural_errors.py
```

Output:

```text
data/deep_learning/reports/natural_distribution_errors.csv
```

## 8.2 False NOVA 2 predictions

There were 197 incorrect predictions where the model predicted NOVA 2.

| Actual NOVA group | Predicted as NOVA 2 |
| ----------------- | ------------------: |
| NOVA 1            |                  38 |
| NOVA 3            |                  82 |
| NOVA 4            |                  77 |
| **Total**         |             **197** |

Examples included descriptions containing:

- Roasted peanuts and sea salt.
- Almonds and maltodextrin.
- Blue corn, water, lime and salt.
- Canola oil and additives.
- Dextrose and sweeteners.

The model appears to associate certain ingredient patterns with NOVA 2 even when the provided dataset label belongs to another group.

These errors may involve missing manufacturing context, class overlap or label quality.

They should not automatically be treated as incorrect source labels without verification.

## 8.3 NOVA 4 predicted as NOVA 3

There were 586 products whose actual label was NOVA 4 but were predicted as NOVA 3.

Examples included:

- Complex meal ingredient descriptions.
- Soy sauce and additives.
- Chips and snacks.
- Spice blends.
- Biscuits.
- Jam.
- Mints.
- Processed meat products.

These errors suggest that ingredient patterns may overlap between classes.

Manufacturing context, complex text and label quality may also contribute.

## 8.4 Duplicate analysis in the natural sample

Reported results:

| Measure                                       | Result |
| --------------------------------------------- | -----: |
| Products with repeated descriptions           |    232 |
| Repeated unique descriptions                  |    110 |
| Repeated descriptions with conflicting labels |      1 |

Duplicate descriptions were therefore not the main explanation for the observed natural-sample errors.

## 8.5 Error-analysis conclusion

The error investigation identified NOVA 2 false positives and NOVA 4-to-NOVA 3 confusion as important areas for improvement.

The findings should guide future data-quality review and model experiments.

No automatic label correction or class-prior adjustment has been implemented.

---

# 9. PREDICTION CONFIDENCE ANALYSIS

## 9.1 Purpose

The confidence analysis investigates how strongly DistilBERT assigns probability scores to its predictions.

The analysis uses the 10,000-product natural-distribution prediction report and compares confidence values for correct and incorrect predictions.

Script:

```text
src/deep_learning/evaluation/analyze_prediction_confidence.py
```

Input:

```text
data/deep_learning/reports/natural_distribution_predictions.csv
```

## 9.2 Overall confidence statistics

| Statistic             | Confidence |
| --------------------- | ---------: |
| Number of predictions |     10,000 |
| Mean                  |     95.43% |
| Standard deviation    |       9.66 |
| Minimum               |     32.64% |
| 25th percentile       |     97.33% |
| Median                |     99.06% |
| 75th percentile       |     99.53% |
| Maximum               |     99.71% |

## 9.3 Correct versus incorrect predictions

| Statistic          | Correct predictions | Incorrect predictions |
| ------------------ | ------------------: | --------------------: |
| Count              |               9,011 |                   989 |
| Mean confidence    |              96.82% |                82.77% |
| Standard deviation |                7.35 |                 16.54 |
| Minimum            |              36.76% |                32.64% |
| Median             |              99.23% |                89.51% |
| Maximum            |              99.71% |                99.54% |

Correct predictions generally have higher confidence than incorrect predictions.

However, some incorrect predictions also have very high confidence. This shows that a high Softmax score does not guarantee a correct classification.

## 9.4 False NOVA 2 predictions

The model made 197 incorrect predictions where the predicted class was NOVA 2.

| Actual NOVA group |   Count | Mean confidence | Median confidence |
| ----------------- | ------: | --------------: | ----------------: |
| NOVA 1            |      38 |          74.79% |            73.41% |
| NOVA 3            |      82 |          80.77% |            88.52% |
| NOVA 4            |      77 |          77.35% |            78.82% |
| **Total**         | **197** |                 |                   |

These results reinforce that NOVA 2 predictions are an important area for improvement.

## 9.5 NOVA 4 predicted as NOVA 3

There were 586 products whose actual class was NOVA 4 but were predicted as NOVA 3.

| Statistic          | Confidence |
| ------------------ | ---------: |
| Count              |        586 |
| Mean               |     85.33% |
| Standard deviation |      15.62 |
| Minimum            |     34.15% |
| 25th percentile    |     76.76% |
| Median             |     93.10% |
| 75th percentile    |     97.56% |
| Maximum            |     98.95% |

Some of these incorrect predictions were made with high confidence, indicating that the model can be uncertain about class boundaries even when its output score is high.

## 9.6 Confidence by predicted class

| Predicted NOVA group | Count | Mean confidence | Median confidence |
| -------------------- | ----: | --------------: | ----------------: |
| NOVA 1               |   629 |          91.34% |            98.06% |
| NOVA 2               |   226 |          80.12% |            86.42% |
| NOVA 3               | 2,079 |          92.01% |            97.61% |
| NOVA 4               | 7,066 |          97.29% |            99.41% |

## 9.7 Conclusion

The confidence analysis shows that correct predictions generally receive higher confidence scores.

However, some incorrect predictions also receive very high scores.

Therefore, the Softmax confidence value should be treated as a model-generated score rather than a guarantee of correctness.

The results also highlight the need for further investigation of NOVA 2 predictions and NOVA 4-to-NOVA 3 errors.

---

# 10. MODEL CALIBRATION ANALYSIS

## 10.1 Purpose

Calibration analysis investigates whether the model's predicted probabilities correspond to its observed classification accuracy.

A well-calibrated model should, for example, be correct approximately 90% of the time among a sufficiently large group of predictions assigned 90% confidence.

Script:

```text
src/deep_learning/evaluation/analyze_model_calibration.py
```

The analysis was conducted using the 1,481-product validation dataset.

## 10.2 Calibration results

| Metric                           | Result |
| -------------------------------- | -----: |
| Validation products              |  1,481 |
| Accuracy                         | 90.75% |
| Expected Calibration Error (ECE) | 0.0427 |
| Multiclass Brier Score           | 0.1539 |

Lower ECE and Brier scores generally indicate better calibration.

## 10.3 Reliability table

The script divided predictions into ten confidence intervals.

| Confidence interval | Count | Mean confidence | Accuracy |
| ------------------- | ----: | --------------: | -------: |
| 0–10%               |     0 |               — |        — |
| 10–20%              |     0 |               — |        — |
| 20–30%              |     0 |               — |        — |
| 30–40%              |     1 |          33.45% |    0.00% |
| 40–50%              |     5 |          48.67% |   60.00% |
| 50–60%              |    23 |          54.28% |   30.43% |
| 60–70%              |    49 |          66.54% |   71.43% |
| 70–80%              |    50 |          75.30% |   68.00% |
| 80–90%              |   104 |          86.67% |   83.65% |
| 90–100%             | 1,249 |          98.13% |   94.32% |

## 10.4 Interpretation

The Expected Calibration Error was 0.0427, indicating a relatively small average difference between confidence and observed accuracy across the chosen bins.

The high-confidence 90–100% bin contained 1,249 predictions.

Its average confidence was 98.13%, while its actual accuracy was 94.32%.

This indicates some overconfidence in the model's high-confidence predictions.

The lower-confidence bins also show variation between confidence and actual accuracy.

## 10.5 Generated reports

The calibration script generated:

```text
data/deep_learning/reports/

calibration_reliability.csv
calibration_reliability.png
calibration_summary.txt
```

The CSV contains the reliability-bin statistics, the PNG contains the reliability diagram, and the text file contains the calibration summary.

## 10.6 Limitations

The calibration analysis used the validation partition, which was also involved in model selection.

Therefore, these results are diagnostic and should not be treated as independent final calibration evidence.

Temperature scaling and other probability-calibration techniques have not been implemented.

The inference interface currently returns the model's raw Softmax probabilities.

## 10.7 Conclusion

The calibration analysis provides an additional measure of how the model's probability outputs relate to observed accuracy.

Although the ECE is 0.0427, the high-confidence bin shows that the model can be overconfident.

Future work should investigate calibration methods using a separate calibration dataset.

---

---

# 11. MODEL INFERENCE INTERFACE

## 11.1 Implementation

A reusable prediction interface was implemented using the fine-tuned DistilBERT model.

File:

```text
src/deep_learning/inference/predict.py
```

Saved model directory:

```text
data/deep_learning/models/distilbert_food_nova/
```

## 11.2 Main functions

The inference module provides two main functions:

**`load_model()`**

- Loads the saved tokenizer.
- Loads the fine-tuned DistilBERT classification model.
- Selects an available execution device.
- Sets the model to evaluation mode.
- Returns the tokenizer and model.

**`predict_nova(ingredient_text, tokenizer, model)`**

- Accepts ingredient text.
- Validates the input.
- Tokenizes the text with a maximum length of 256 tokens.
- Runs the model without gradient calculation.
- Applies Softmax to the output logits.
- Returns the predicted NOVA group and class probability scores.

## 11.3 Prediction output

The prediction function returns:

- `predicted_nova`: predicted NOVA group from 1 to 4.
- `label`: human-readable NOVA label.
- `confidence`: highest Softmax score as a percentage.
- `class_probabilities`: probability scores for all four classes.

These confidence scores are raw model outputs and are not guaranteed probabilities of correctness.

## 11.4 Verification

The interactive inference interface was successfully executed:

```bash
python -m src.deep_learning.inference.predict
```

The saved-model smoke test was also completed:

```bash
python -m src.deep_learning.evaluation.test_predictor
```

Reported result:

```text
Correct predictions: 10/10
Sample accuracy: 100.00%
```

This verifies that the saved model and prediction interface work together on the tested sample. It does not replace the full held-out test accuracy of 91.34%.

## 11.5 Current status and limitation

The reusable inference interface is implemented and smoke-tested.

It accepts ingredient text rather than raw camera images or barcodes.

Backend API development, barcode lookup integration and Android integration have not been implemented or verified.

---

# 12. MODEL PERFORMANCE BENCHMARK

## 12.1 Purpose

The benchmark evaluates the practical inference performance of the saved DistilBERT model on the local development machine.

Script:

```text
src/deep_learning/evaluation/benchmark_model_performance.py
```

The benchmark measures:

- Saved model directory size.
- Model loading time.
- Prediction latency.
- Average inference time.
- Median inference time.
- Minimum and maximum inference time.
- Approximate prediction throughput.

The benchmark performs warm-up predictions before measuring 50 individual predictions.

## 12.2 Hardware and environment

| Property         | Details               |
| ---------------- | --------------------- |
| Processor        | Apple M1              |
| Memory           | 8 GB unified memory   |
| Execution device | Apple MPS             |
| Framework        | PyTorch               |
| Model            | Fine-tuned DistilBERT |

## 12.3 Benchmark results

| Metric                         |                   Result |
| ------------------------------ | -----------------------: |
| Saved model directory size     |                256.11 MB |
| Model loading time             |             2.48 seconds |
| Device                         |            MPS (`mps:0`) |
| Number of measured predictions |                       50 |
| Average inference time         |                 57.26 ms |
| Median inference time          |                 51.20 ms |
| Minimum inference time         |                 32.75 ms |
| Maximum inference time         |                205.83 ms |
| Approximate throughput         | 17.46 predictions/second |

## 12.4 Interpretation

The model takes approximately 2.48 seconds to load in the measured run.

After loading, its average single-text inference time was 57.26 milliseconds on the tested Apple M1 machine using MPS.

The model directory occupies approximately 256.11 MB.

These results demonstrate that the saved model can perform local inference on the tested development machine.

The measurements are specific to the tested environment and benchmark procedure. They do not establish performance on Android devices or production servers.

## 12.5 Generated report

The benchmark result was saved to:

```text
data/deep_learning/reports/model_performance_benchmark.txt
```

## 12.6 Deployment considerations

The current model uses PyTorch and Hugging Face Transformers.

Because the saved model directory is approximately 256 MB and Android compatibility has not been tested, backend inference is the provisional deployment recommendation.

An Android application could send ingredient text to a backend, receive the predicted NOVA group and display the result.

Direct Android deployment would require model conversion, mobile-runtime compatibility testing and device-level performance benchmarking.

## 12.7 Conclusion

The benchmark confirms that the current DistilBERT checkpoint can be loaded and used for relatively fast inference on the development machine.

The model's size, loading time and inference latency have now been measured and documented.

Further testing is required before selecting a final production or mobile deployment approach.
