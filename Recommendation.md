Based on the cross-validation results, I recommend Logistic Regression with L1 regularization (C=0.1) because it achieved the highest F1 score among the real models (0.3420), with strong recall (0.6256) for identifying churners. Accuracy alone is not sufficient for this problem: the most-frequent dummy achieved the highest accuracy (0.8375) but completely failed to detect churners, with precision, recall, and F1 all equal to 0. This shows that in an imbalanced churn dataset, accuracy can be misleading, while F1 gives a better balance between precision and recall. Compared with the stratified dummy baseline (F1 = 0.1730), the L1 model performs meaningfully better, suggesting that it has learned useful signal rather than guessing according to class proportions. The final test-set evaluation supports this recommendation, as the selected model achieved an F1 score of 0.3787 and recall of 0.6531 on unseen data, which is consistent with and slightly better than the cross-validation results.


| Model | Mean Accuracy | Std Accuracy | Mean Precision | Mean Recall | Mean F1 |
|---|---:|---:|---:|---:|---:|
| LogReg (default) | 0.7158 | 0.0273 | 0.2572 | 0.3949 | 0.3101 |
| LogReg (L1, C=0.1) | 0.6092 | 0.0267 | 0.2359 | 0.6256 | 0.3420 |
| RidgeClassifier | 0.8317 | 0.0057 | 0.4390 | 0.1231 | 0.1893 |
| Most-frequent Dummy | 0.8375 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Stratified Dummy | 0.7450 | 0.0180 | 0.1829 | 0.1641 | 0.1730 |



**Final Test-Set Metrics for Selected Best Model: LogReg (L1, C=0.1)**

- Accuracy: 0.6500
- Precision: 0.2667
- Recall: 0.6531
- F1: 0.3787