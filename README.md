# Pulsar classification (imbalanced data)

A classical machine-learning pipeline that separates **pulsar candidates** from radio-frequency interference and noise,
using 7 statistics of the integrated pulse profile and the DM-SNR curve. Written in 2024 as an exercise in
handling an imbalanced dataset honestly.

## Method

1. Exploratory analysis with `ydata-profiling`: no missing values, strong class imbalance, and highly correlated features.
2. Stratified train/test split.
3. Pipeline: `StandardScaler` → `PCA` → `KNeighborsClassifier`.
4. `GridSearchCV` (5-fold) over the number of PCA components, k, weights and distance metric (280 candidates).
5. Evaluation with precision, recall, F1 and balanced accuracy, not accuracy alone.

## Result (test set)

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| non-pulsar | 0.99 | 1.00 | 0.99 | 2,936 |
| pulsar | 0.88 | 0.45 | 0.60 | 62 |

The model rarely raises a false alarm, but it misses more than half of the pulsars. High accuracy hides this,
which is the main lesson of the project. Next steps would be class weights or resampling, and threshold tuning on the PR curve.

## Run

```bash
pip install -r requirements.txt
jupyter notebook pulsar_classification.ipynb
```

Data: [`Pulsar_cleaned.csv`](Pulsar_cleaned.csv), a cleaned version of a public Kaggle pulsar dataset
(outlier removal and feature selection by the dataset author). The notebook comments are in Polish.

**Status:** 2024, not maintained.

## License

[MIT](LICENSE)
