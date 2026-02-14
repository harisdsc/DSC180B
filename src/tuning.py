from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from optuna.samplers import TPESampler
from catboost import CatBoostClassifier
from catboost.utils import get_gpu_device_count
import xgboost as xgb
import lightgbm as lgb
import optuna

GPU_AVAILABLE = get_gpu_device_count() > 0
print(f"GPU Available: {GPU_AVAILABLE}")

def tune_hyperparameters(model_name, X, y, n_trials=30):
    print(f"Starting Optuna optimization for {model_name}...")
    def objective(trial):
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        
        if model_name == 'xgboost':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'gamma': trial.suggest_float('gamma', 0, 5),
                'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1, 10), # Handling imbalance
                'eval_metric': 'auc',
                'n_jobs': -1,
                'verbosity': 0
            }
            if GPU_AVAILABLE:
                params['tree_method'] = 'hist'
                params['device'] = 'gpu'

            model = xgb.XGBClassifier(**params)
            
        elif model_name == 'lightgbm':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'num_leaves': trial.suggest_int('num_leaves', 20, 150),
                'max_depth': trial.suggest_int('max_depth', -1, 15),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
                'class_weight': trial.suggest_categorical('class_weight', [None, 'balanced']),
                'verbose': -1,
                'n_jobs': -1
            }
            if GPU_AVAILABLE:
                params['device'] = 'gpu'

            model = lgb.LGBMClassifier(**params)
            
        elif model_name == 'catboost':
            params = {
                'iterations': trial.suggest_int('iterations', 100, 1000),
                'depth': trial.suggest_int('depth', 4, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
                'border_count': trial.suggest_int('border_count', 32, 255),
                'auto_class_weights': trial.suggest_categorical('auto_class_weights', ['None', 'Balanced']),
                'verbose': 0,
                'allow_writing_files': False
            }
            if GPU_AVAILABLE:
                params['task_type'] = 'GPU'

            model = CatBoostClassifier(**params)
            
        elif model_name == 'log-reg':
            C = trial.suggest_float('C', 1e-4, 100, log=True)
            model = Pipeline([
                ('scaler', StandardScaler()),
                ('clf', LogisticRegression(C=C, solver='lbfgs', max_iter=25_000, class_weight='balanced'))
            ])

        scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc', n_jobs=-1)
        return scores.mean()

    study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials)

    print(f"Best params: {study.best_params}")
    print(f"Best CV AUC: {study.best_value}")
    
    if model_name == 'log-reg':
        best_model = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(**study.best_params, solver='lbfgs', max_iter=25_000, class_weight='balanced'))
        ])
    elif model_name == 'xgboost':
        if GPU_AVAILABLE:
            best_model = xgb.XGBClassifier(**study.best_params, eval_metric='auc', n_jobs=-1, verbosity=0, tree_method='gpu_hist')
        else:
            best_model = xgb.XGBClassifier(**study.best_params, eval_metric='auc', n_jobs=-1, verbosity=0)
    elif model_name == 'lightgbm':
        if GPU_AVAILABLE:
            best_model = lgb.LGBMClassifier(**study.best_params, verbose=-1, n_jobs=-1, device='gpu')
        else:
            best_model = lgb.LGBMClassifier(**study.best_params, verbose=-1, n_jobs=-1)
    elif model_name == 'catboost':
        if GPU_AVAILABLE:
            best_model = CatBoostClassifier(**study.best_params, verbose=0, allow_writing_files=False, task_type='GPU')
        else:
            best_model = CatBoostClassifier(**study.best_params, verbose=0, allow_writing_files=False)
        
    return best_model, study.best_params, study.best_value