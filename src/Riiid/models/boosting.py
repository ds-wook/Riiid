import gc
from typing import List, Tuple, Union

import lightgbm as lgb
import pandas as pd
from lightgbm import Booster
from sklearn.metrics import roc_auc_score


# Function for training and evaluation
def train_and_evaluate(
    params_path: str,
    train: pd.DataFrame,
    valid: pd.DataFrame,
    feature_engineering: bool = False,
    verbose: Union[int, bool] = False,
) -> Tuple[str, List[str], Booster]:

    TARGET = "answered_correctly"
    # Features to train and predict
    FEATURES = [
        "prior_question_elapsed_time",
        "prior_question_had_explanation",
        "part",
        "answered_correctly_u_avg",
        "elapsed_time_u_avg",
        "explanation_u_avg",
        "answered_correctly_q_avg",
        "elapsed_time_q_avg",
        "explanation_q_avg",
        "answered_correctly_uq_count",
        "timestamp_u_recency_1",
        "timestamp_u_recency_2",
        "timestamp_u_recency_3",
        "timestamp_u_incorrect_recency",
    ]

    # Delete some training data to experiment faster
    if feature_engineering:
        train = train.sample(15000000, random_state=2021)
    gc.collect()
    print(f"Traning with {train.shape[0]} rows and {len(FEATURES)} features")
    drop_cols = list(set(train.columns) - set(FEATURES))
    y_train = train[TARGET]
    y_val = valid[TARGET]
    # Drop unnecessary columns
    train.drop(drop_cols, axis=1, inplace=True)
    valid.drop(drop_cols, axis=1, inplace=True)
    gc.collect()

    lgb_train = lgb.Dataset(train[FEATURES], y_train)
    lgb_valid = lgb.Dataset(valid[FEATURES], y_val)
    del train, y_train
    gc.collect()

    params = pd.read_pickle(params_path)

    model = lgb.train(
        params=params,
        train_set=lgb_train,
        num_boost_round=10000,
        valid_sets=[lgb_train, lgb_valid],
        early_stopping_rounds=10,
        verbose_eval=verbose,
    )

    print(
        "Our ROC AUC score for the validation data is:",
        roc_auc_score(
            y_val, model.predict(valid[FEATURES], num_iteration=model.best_iteration)
        ),
    )

    return TARGET, FEATURES, model
