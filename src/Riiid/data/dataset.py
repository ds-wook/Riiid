import gc
from collections import defaultdict
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd


# Funcion for user stats with loops
def add_features(
    df: pd.DataFrame,
    answered_correctly_u_count: Dict[int, int],
    answered_correctly_u_sum: Dict[int, int],
    elapsed_time_u_sum: Dict[int, int],
    explanation_u_sum: Dict[int, int],
    timestamp_u: Dict[int, int],
    timestamp_u_incorrect: Dict[int, int],
    answered_correctly_q_count: Dict[int, int],
    answered_correctly_q_sum: Dict[int, int],
    elapsed_time_q_sum: Dict[int, int],
    explanation_q_sum: Dict[int, int],
    answered_correctly_uq: Dict[int, int],
    update: bool = True,
) -> pd.DataFrame:
    # -----------------------------------------------------------------------
    # Client features
    answered_correctly_u_avg = np.zeros(len(df), dtype=np.float32)
    elapsed_time_u_avg = np.zeros(len(df), dtype=np.float32)
    explanation_u_avg = np.zeros(len(df), dtype=np.float32)
    timestamp_u_recency_1 = np.zeros(len(df), dtype=np.float32)
    timestamp_u_recency_2 = np.zeros(len(df), dtype=np.float32)
    timestamp_u_recency_3 = np.zeros(len(df), dtype=np.float32)
    timestamp_u_incorrect_recency = np.zeros(len(df), dtype=np.float32)
    # -----------------------------------------------------------------------
    # Question features
    answered_correctly_q_avg = np.zeros(len(df), dtype=np.float32)
    elapsed_time_q_avg = np.zeros(len(df), dtype=np.float32)
    explanation_q_avg = np.zeros(len(df), dtype=np.float32)
    # -----------------------------------------------------------------------
    # User Question
    answered_correctly_uq_count = np.zeros(len(df), dtype=np.int32)
    # -----------------------------------------------------------------------

    for num, row in enumerate(
        df[
            [
                "user_id",
                "answered_correctly",
                "content_id",
                "prior_question_elapsed_time",
                "prior_question_had_explanation",
                "timestamp",
            ]
        ].values
    ):

        # Client features assignation
        # ------------------------------------------------------------------
        if answered_correctly_u_count[row[0]] != 0:
            answered_correctly_u_avg[num] = (
                answered_correctly_u_sum[row[0]] / answered_correctly_u_count[row[0]]
            )
            elapsed_time_u_avg[num] = (
                elapsed_time_u_sum[row[0]] / answered_correctly_u_count[row[0]]
            )
            explanation_u_avg[num] = (
                explanation_u_sum[row[0]] / answered_correctly_u_count[row[0]]
            )
        else:
            answered_correctly_u_avg[num] = np.nan
            elapsed_time_u_avg[num] = np.nan
            explanation_u_avg[num] = np.nan

        if len(timestamp_u[row[0]]) == 0:
            timestamp_u_recency_1[num] = np.nan
            timestamp_u_recency_2[num] = np.nan
            timestamp_u_recency_3[num] = np.nan
        elif len(timestamp_u[row[0]]) == 1:
            timestamp_u_recency_1[num] = row[5] - timestamp_u[row[0]][0]
            timestamp_u_recency_2[num] = np.nan
            timestamp_u_recency_3[num] = np.nan
        elif len(timestamp_u[row[0]]) == 2:
            timestamp_u_recency_1[num] = row[5] - timestamp_u[row[0]][1]
            timestamp_u_recency_2[num] = row[5] - timestamp_u[row[0]][0]
            timestamp_u_recency_3[num] = np.nan
        elif len(timestamp_u[row[0]]) == 3:
            timestamp_u_recency_1[num] = row[5] - timestamp_u[row[0]][2]
            timestamp_u_recency_2[num] = row[5] - timestamp_u[row[0]][1]
            timestamp_u_recency_3[num] = row[5] - timestamp_u[row[0]][0]

        if len(timestamp_u_incorrect[row[0]]) == 0:
            timestamp_u_incorrect_recency[num] = np.nan
        else:
            timestamp_u_incorrect_recency[num] = (
                row[5] - timestamp_u_incorrect[row[0]][0]
            )

        # ------------------------------------------------------------------
        # Question features assignation
        if answered_correctly_q_count[row[2]] != 0:
            answered_correctly_q_avg[num] = (
                answered_correctly_q_sum[row[2]] / answered_correctly_q_count[row[2]]
            )
            elapsed_time_q_avg[num] = (
                elapsed_time_q_sum[row[2]] / answered_correctly_q_count[row[2]]
            )
            explanation_q_avg[num] = (
                explanation_q_sum[row[2]] / answered_correctly_q_count[row[2]]
            )
        else:
            answered_correctly_q_avg[num] = np.nan
            elapsed_time_q_avg[num] = np.nan
            explanation_q_avg[num] = np.nan
        # ------------------------------------------------------------------
        # Client Question assignation
        answered_correctly_uq_count[num] = answered_correctly_uq[row[0]][row[2]]
        # ------------------------------------------------------------------
        # ------------------------------------------------------------------
        # Client features updates
        answered_correctly_u_count[row[0]] += 1
        elapsed_time_u_sum[row[0]] += row[3]
        explanation_u_sum[row[0]] += int(row[4])
        if len(timestamp_u[row[0]]) == 3:
            timestamp_u[row[0]].pop(0)
            timestamp_u[row[0]].append(row[5])
        else:
            timestamp_u[row[0]].append(row[5])
        # ------------------------------------------------------------------
        # Question features updates
        answered_correctly_q_count[row[2]] += 1
        elapsed_time_q_sum[row[2]] += row[3]
        explanation_q_sum[row[2]] += int(row[4])
        # ------------------------------------------------------------------
        # Client Question updates
        answered_correctly_uq[row[0]][row[2]] += 1
        # ------------------------------------------------------------------
        # Flag for training and inference
        if update:
            # ------------------------------------------------------------------
            # Client features updates
            answered_correctly_u_sum[row[0]] += row[1]
            if row[1] == 0:
                if len(timestamp_u_incorrect[row[0]]) == 1:
                    timestamp_u_incorrect[row[0]].pop(0)
                    timestamp_u_incorrect[row[0]].append(row[5])
                else:
                    timestamp_u_incorrect[row[0]].append(row[5])

            # ------------------------------------------------------------------
            # Question features updates
            answered_correctly_q_sum[row[2]] += row[1]
            # ------------------------------------------------------------------

    user_df = pd.DataFrame(
        {
            "answered_correctly_u_avg": answered_correctly_u_avg,
            "elapsed_time_u_avg": elapsed_time_u_avg,
            "explanation_u_avg": explanation_u_avg,
            "answered_correctly_q_avg": answered_correctly_q_avg,
            "elapsed_time_q_avg": elapsed_time_q_avg,
            "explanation_q_avg": explanation_q_avg,
            "answered_correctly_uq_count": answered_correctly_uq_count,
            "timestamp_u_recency_1": timestamp_u_recency_1,
            "timestamp_u_recency_2": timestamp_u_recency_2,
            "timestamp_u_recency_3": timestamp_u_recency_3,
            "timestamp_u_incorrect_recency": timestamp_u_incorrect_recency,
        }
    )

    df = pd.concat([df, user_df], axis=1)
    return df


def update_features(
    df,
    answered_correctly_u_sum: Dict[int, int],
    answered_correctly_q_sum: Dict[int, int],
    timestamp_u_incorrect: Dict[int, int],
):
    for row in df[
        ["user_id", "answered_correctly", "content_id", "content_type_id", "timestamp"]
    ].values:
        if row[3] == 0:
            # ------------------------------------------------------------------
            # Client features updates
            answered_correctly_u_sum[row[0]] += row[1]
            if row[1] == 0:
                if len(timestamp_u_incorrect[row[0]]) == 1:
                    timestamp_u_incorrect[row[0]].pop(0)
                    timestamp_u_incorrect[row[0]].append(row[4])
                else:
                    timestamp_u_incorrect[row[0]].append(row[4])
            # ------------------------------------------------------------------
            # Question features updates
            answered_correctly_q_sum[row[2]] += row[1]
            # ------------------------------------------------------------------


def read_and_preprocess(
    feature_engineering: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, Dict[int, Any]]:

    train_pickle = "../input/riiid-cross-validation-files/cv1_train.pickle"
    valid_pickle = "../input/riiid-cross-validation-files/cv1_valid.pickle"
    question_file = "../input/riiid-test-answer-prediction/questions.csv"

    # Read data
    feld_needed = [
        "timestamp",
        "user_id",
        "answered_correctly",
        "content_id",
        "content_type_id",
        "prior_question_elapsed_time",
        "prior_question_had_explanation",
    ]
    train = pd.read_pickle(train_pickle)[feld_needed]
    valid = pd.read_pickle(valid_pickle)[feld_needed]
    # Delete some trianing data to don't have ram problems
    if feature_engineering:
        train = train.iloc[-40000000:]

    # Filter by content_type_id to discard lectures
    train = train.loc[train.content_type_id == False].reset_index(drop=True)
    valid = valid.loc[valid.content_type_id == False].reset_index(drop=True)

    # Changing dtype to avoid lightgbm error
    train[
        "prior_question_had_explanation"
    ] = train.prior_question_had_explanation.fillna(False).astype("int8")
    valid[
        "prior_question_had_explanation"
    ] = valid.prior_question_had_explanation.fillna(False).astype("int8")

    # Fill prior question elapsed time with the mean
    prior_question_elapsed_time_mean = (
        train["prior_question_elapsed_time"].dropna().mean()
    )
    train["prior_question_elapsed_time"].fillna(
        prior_question_elapsed_time_mean, inplace=True
    )
    valid["prior_question_elapsed_time"].fillna(
        prior_question_elapsed_time_mean, inplace=True
    )

    # Merge with question dataframe
    questions_df = pd.read_csv(question_file)
    questions_df["part"] = questions_df["part"].astype(np.int32)
    questions_df["bundle_id"] = questions_df["bundle_id"].astype(np.int32)

    train = pd.merge(
        train,
        questions_df[["question_id", "part"]],
        left_on="content_id",
        right_on="question_id",
        how="left",
    )
    valid = pd.merge(
        valid,
        questions_df[["question_id", "part"]],
        left_on="content_id",
        right_on="question_id",
        how="left",
    )

    # Client dictionaries
    answered_correctly_u_count = defaultdict(int)
    answered_correctly_u_sum = defaultdict(int)
    elapsed_time_u_sum = defaultdict(int)
    explanation_u_sum = defaultdict(int)
    timestamp_u = defaultdict(list)
    timestamp_u_incorrect = defaultdict(list)

    # Question dictionaries
    answered_correctly_q_count = defaultdict(int)
    answered_correctly_q_sum = defaultdict(int)
    elapsed_time_q_sum = defaultdict(int)
    explanation_q_sum = defaultdict(int)

    # Client Question dictionary
    answered_correctly_uq = defaultdict(lambda: defaultdict(int))

    print("User feature calculation started...")
    print("\n")
    train = add_features(
        train,
        answered_correctly_u_count,
        answered_correctly_u_sum,
        elapsed_time_u_sum,
        explanation_u_sum,
        timestamp_u,
        timestamp_u_incorrect,
        answered_correctly_q_count,
        answered_correctly_q_sum,
        elapsed_time_q_sum,
        explanation_q_sum,
        answered_correctly_uq,
    )
    valid = add_features(
        valid,
        answered_correctly_u_count,
        answered_correctly_u_sum,
        elapsed_time_u_sum,
        explanation_u_sum,
        timestamp_u,
        timestamp_u_incorrect,
        answered_correctly_q_count,
        answered_correctly_q_sum,
        elapsed_time_q_sum,
        explanation_q_sum,
        answered_correctly_uq,
    )
    gc.collect()
    print("User feature calculation completed...")
    print("\n")

    features_dicts = {
        "answered_correctly_u_count": answered_correctly_u_count,
        "answered_correctly_u_sum": answered_correctly_u_sum,
        "elapsed_time_u_sum": elapsed_time_u_sum,
        "explanation_u_sum": explanation_u_sum,
        "answered_correctly_q_count": answered_correctly_q_count,
        "answered_correctly_q_sum": answered_correctly_q_sum,
        "elapsed_time_q_sum": elapsed_time_q_sum,
        "explanation_q_sum": explanation_q_sum,
        "answered_correctly_uq": answered_correctly_uq,
        "timestamp_u": timestamp_u,
        "timestamp_u_incorrect": timestamp_u_incorrect,
    }

    return train, valid, questions_df, prior_question_elapsed_time_mean, features_dicts
