import lightgbm as lgb
import numpy as np
import pandas as pd
import gc
from config import LGBM_BASE_PARAMS, TARGET, WEIGHT, HORIZONS, SEED
from evaluation import weighted_rmse_score


def train_lgbm_single(X_train, y_train, w_train,
                       X_val=None, y_val=None, w_val=None,
                       params=None, num_boost_round=5000,
                       early_stopping_rounds=100, verbose_eval=200):
    params = params or LGBM_BASE_PARAMS.copy()
    
    cat_features = [c for c in X_train.columns 
                    if X_train[c].dtype.name == "category"]
    
    dtrain = lgb.Dataset(X_train, label=y_train, weight=w_train,
                         categorical_feature=cat_features)
    
    callbacks = [lgb.log_evaluation(period=verbose_eval)]
    valid_sets = [dtrain]
    valid_names = ["train"]
    
    if X_val is not None:
        dval = lgb.Dataset(X_val, label=y_val, weight=w_val,
                           reference=dtrain, categorical_feature=cat_features)
        valid_sets.append(dval)
        valid_names.append("valid")
        callbacks.append(lgb.early_stopping(stopping_rounds=early_stopping_rounds))
    
    model = lgb.train(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        valid_sets=valid_sets,
        valid_names=valid_names,
        callbacks=callbacks,
    )
    
    # Evaluate
    if X_val is not None:
        y_pred_val = model.predict(X_val, num_iteration=model.best_iteration)
        score = weighted_rmse_score(y_val, y_pred_val, w_val)
        print(f"\n>>> Validation Score: {score:.5f} "
              f"(best iter: {model.best_iteration})")
    
    gc.collect()
    return model


def train_per_horizon(train_df, features, 
                      split_ts_index=3000, params=None,
                      num_boost_round=5000, early_stopping_rounds=100):
    models = {}
    scores = {}
    
    for h in HORIZONS:
        print(f"\n{'='*60}")
        print(f"Training model for Horizon = {h}")
        print(f"{'='*60}")
        
        h_data = train_df[train_df["horizon"] == h]
        
        # Time-based split
        h_train = h_data[h_data["ts_index"] <= split_ts_index]
        h_val   = h_data[h_data["ts_index"] >  split_ts_index]
        
        X_tr, y_tr = h_train[features], h_train[TARGET]
        w_tr = h_train[WEIGHT]
        X_vl, y_vl = h_val[features], h_val[TARGET]
        w_vl = h_val[WEIGHT]
        
        print(f"  Train: {X_tr.shape}, Val: {X_vl.shape}")
        
        model = train_lgbm_single(
            X_tr, y_tr, w_tr,
            X_vl, y_vl, w_vl,
            params=params,
            num_boost_round=num_boost_round,
            early_stopping_rounds=early_stopping_rounds,
        )
        
        models[h] = model
        
        # Score
        y_pred = model.predict(X_vl, num_iteration=model.best_iteration)
        scores[h] = weighted_rmse_score(y_vl, y_pred, w_vl)
    
    print(f"\n{'='*60}")
    print("Per-horizon validation scores:")
    for h, s in scores.items():
        print(f"  Horizon {h:2d}: {s:.5f}")
    print(f"{'='*60}")
    
    return models, scores


def predict_per_horizon(models, test_df, features):
    predictions = []
    
    for h in HORIZONS:
        model = models[h]
        h_test = test_df[test_df["horizon"] == h].copy()
        
        h_test["prediction"] = model.predict(
            h_test[features], num_iteration=model.best_iteration
        )
        predictions.append(h_test[["id", "prediction"]])
    
    submission = pd.concat(predictions, axis=0)
    
    # Sanity checks
    assert submission.shape[0] == test_df.shape[0], \
        f"Row count mismatch: {submission.shape[0]} vs {test_df.shape[0]}"
    assert submission["prediction"].isnull().sum() == 0, \
        "Found NaN in predictions!"
    
    return submission


def create_submission(submission_df, filename="submission.csv"):
    from config import OUTPUT_DIR
    import os
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, filename)
    
    submission_df[["id", "prediction"]].to_csv(path, index=False)
    print(f"Saved: {path} ({submission_df.shape[0]} rows)")
    print(submission_df.head())
    return path
