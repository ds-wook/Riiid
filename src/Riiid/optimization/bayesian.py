import warnings
from typing import Callable, Sequence, Union

import neptune.new as neptune
import neptune.new.integrations.optuna as optuna_utils
import optuna
import pandas as pd
import yaml
from hydra.utils import to_absolute_path
from models.boosting import train_and_evaluate
from neptune.new.exceptions import NeptuneMissingApiTokenException
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from optuna.study import Study
from optuna.trial import FrozenTrial, Trial
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")


class BayesianOptimizer:
    def __init__(
        self, objective_function: Callable[[Trial], Union[float, Sequence[float]]]
    ):
        self.objective_function = objective_function

    def build_study(self, trials: FrozenTrial, verbose: bool = False):
        try:
            run = neptune.init(
                project="ds-wook/optiver-prediction", tags="optimization"
            )

            neptune_callback = optuna_utils.NeptuneCallback(
                run, plots_update_freq=1, log_plot_slice=False, log_plot_contour=False
            )
            sampler = TPESampler(seed=42)
            study = optuna.create_study(
                study_name="TPE Optimization",
                direction="minimize",
                sampler=sampler,
                pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=5),
            )
            study.optimize(
                self.objective_function, n_trials=trials, callbacks=[neptune_callback]
            )
            run.stop()

        except NeptuneMissingApiTokenException:
            sampler = TPESampler(seed=42)
            study = optuna.create_study(
                study_name="optimization", direction="minimize", sampler=sampler
            )
            study.optimize(self.objective_function, n_trials=trials)
        if verbose:
            self.display_study_statistics(study)

        return study

    @staticmethod
    def display_study_statistics(study: Study):
        print("Best trial:")
        trial = study.best_trial
        print("  Value: ", trial.value)
        print("  Params: ")
        for key, value in trial.params.items():
            print(f"    '{key}': {value},")

    @staticmethod
    def lgbm_save_params(study: Study, params_name: str):
        params = study.best_trial.params
        params["seed"] = 42
        params["feature_fraction_seed"] = 42
        params["bagging_seed"] = 42
        params["drop_seed"] = 42
        params["boosting"] = "gbdt"
        params["objective"] = "rmse"
        params["verbosity"] = -1
        params["n_jobs"] = -1

        path = to_absolute_path("../../parameters/" + params_name)
        with open(path, "w") as p:
            yaml.dump(params, p)


def lgbm_objective(
    trial: FrozenTrial,
    train: pd.DataFrame,
    valid: pd.Series,
) -> float:
    params = {
        "objective": "binary",
        "metric": "AUC",
        "boosting_type": "dart",  # To improve AUC
        "lambda_l1": trial.suggest_loguniform("lambda_l1", 0.1, 0.9),
        "lambda_l2": trial.suggest_loguniform("lambda_l2", 0.1, 0.9),
        "num_leaves": trial.suggest_int("num_leaves", 2, 256),
        "feature_fraction": trial.suggest_uniform("feature_fraction", 0.4, 1.0),
        "bagging_fraction": trial.suggest_uniform("bagging_fraction", 0.4, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "device": "gpu",
        "gpu_platform_id": 0,
        "gpu_device_id": 0,
        "verbose": 0,
    }
    target, features, model = train_and_evaluate(params, train, valid)
    y_va = valid[features]
    # Prediction
    y_pred = model.predict(valid[features], num_iteration=model.best_iteration)
    accuracy = roc_auc_score(y_va, y_pred, labels="ROC curve", average="weighted")

    return accuracy
