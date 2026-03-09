# CashFlow Credit Scoring
**DSC 180B Capstone Project · UC San Diego · Prism Data**

Evaluate credit risk using machine learning on transaction-level cash flows as a data-driven alternative to traditional scoring.

---

## Introduction
Traditional credit scores rely on historical repayment data, often excluding those with limited formal credit history. This project develops a framework to assess creditworthiness via real-time behavioral signals such as income consistency, spending composition, and liquidity trends derived from bank transactions. Our results demonstrate that these cash-flow signals significantly enhance risk prediction for a more inclusive financial system.

## Installation & Setup
1. **Clone the Repository:**
   ```
   git clone https://github.com/harisdsc/DSC180B.git
   cd DSC180B
   ```
2. **Environment Setup:**
    ```
    python3 -m venv env
    source env/bin/activate
    pip install -r requirements.txt
    ```

## Data
The project processes hierarchical financial records:

- **Consumer Data**: Evaluation dates and delinquency outcomes (15,000 rows)

- **Account Data**: Types, balance dates, and amounts (24,466 rows)

- **Transaction Data**: Records with categories, amounts, and dates (6.4M rows)

- **Category Map**: Mapping for 50 transaction categories
  
## Usage
1. **Data Pipeline**:

```
python3 src/pipeline.py
```
Output is cached at data/features.pqt

2. **Model Training**:

```
python3 train.py <model> [--tune]
```
- **Models**: `log-reg`, `xgboost`, `catboost`, `lightgbm`
- `--tune`: Runs 100 trials of Optuna hyperparameter optimization

## Project Structure

- `configs/`: Optimized model hyperparameters
- `data/`: Local storage for datasets and features (git-ignored)
- `models`: Saved trained model binaries
- `src/`: Source code fro loading, engineerin, and evaluation
- `train.py`: Main script for model training and seletion
  
## Results

The **XGBoost** model achieved the highest perfomance with an **AUC-ROC of ...**
e XGBoost model achieved the highest performance with an AUC-ROC of 0.8391.

- XGBoost: ... AUC

- LightGBM: ... AUC

- CatBoost: ... AUC

- Logistic Regression: ... AUC
---
**Team**: Ada Mo, Brighton Chan, Haris Saif, Kyle Choi
**Mentors**: Kyle Nero & Daniel Matthew (Prism Data)