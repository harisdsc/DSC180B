## train.py

### Models

> `python3 train.py` **`<model>`**

* `log-reg` : Train Logistic Regression (default).
* `xgboost` : Train XGBoost Classifier.
* `catboost` : Train CatBoost Classifier.
* `lightgbm` : Train LightGBM Classifier.
* `all` : Train and evaluate all models sequentially.

### Options

> `python3 train.py <model>` **`--tune`**

* `--tune` : Run Optuna hyperparameter optimization (100 trials) before training and save best config.

---

## src/pipeline.py

### Data

> `python3 src/pipeline.py`

* `pipeline` : Load raw data, generate features (income, balance, transactions), and save processed data to `data/features.pqt` (called within train.py)

---