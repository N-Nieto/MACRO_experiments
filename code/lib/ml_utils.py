from collections.abc import Sequence
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


class TunedModel(TypedDict):
    """Output of get_inner_loop_optuna."""

    model: xgb.XGBClassifier
    best_params: dict[str, Any]


def get_inner_loop_optuna(
    X: pd.DataFrame | pd.Series,
    y: np.ndarray | pd.Series,
    kf_inner: StratifiedKFold,
    params_optuna: dict[str, Any],
) -> TunedModel:
    """
    Get the best model with OPTUNA

    Optimize hyperparameters using Optuna for an XGBoost model and train the final model.
  
    Parameters:
        X (pd.DataFrame): The input features.
        y (pd.Series): The target variable.
        kf_inner (KFold): Inner cross-validation strategy.

    Returns:
        dict: A dictionary containing the final model, best hyperparameters.
    """
    def objective(trial: optuna.Trial) -> float:
        """
        Objective function to optimize using Optuna.

        Parameters:
            trial (optuna.Trial): The Optuna trial.

        Returns:
            float: The value to minimize or maximize during optimization.
        """
        # Define the hyperparameters to tune
        params = {
            "objective": params_optuna["objective"],
            "eval_metric": params_optuna["eval_metric"],
            "max_depth": trial.suggest_int(
                "max_depth",
                params_optuna["max_depth_min"],
                params_optuna["max_depth_max"],
                log=True,
            ),
            "alpha": trial.suggest_float(
                "alpha",
                params_optuna["alpha_min"],
                params_optuna["alpha_max"],
                log=True,
            ),
            "lambda": trial.suggest_float(
                "lambda",
                params_optuna["lambda_min"],
                params_optuna["lambda_max"],
                log=True,
            ),
            "eta": trial.suggest_float(
                "eta", params_optuna["eta_min"], params_optuna["eta_max"], log=True
            ),
        }
        # put data in the right format
        dtrain = xgb.DMatrix(X, label=y)
        # allows to remove non-promising trials
        pruning_callback = optuna.integration.XGBoostPruningCallback(trial, "test-auc")

        cv_results = pd.DataFrame(xgb.cv(
            params,
            dtrain,
            num_boost_round=params_optuna["num_boost_round"],
            early_stopping_rounds=params_optuna["early_stopping_rounds"],
            maximize=True,
            callbacks=[pruning_callback],
            folds=kf_inner,
        ))
        # include n_estimators in trial to use it later
        trial.set_user_attr("n_estimators", len(cv_results))

        best_auc = float(cv_results["test-auc-mean"].iloc[-1])

        return best_auc

    # Pruner instance
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=5)
    # Study Instance
    sampler = optuna.samplers.TPESampler(seed=params_optuna["random_state"])
    study = optuna.create_study(pruner=pruner, sampler=sampler, direction="maximize")
    # Optimize the objective
    study.optimize(
        objective, n_trials=params_optuna["optuna_trials"], show_progress_bar=True
    )
    # Print the best hyperparameters and their value
    best_params = study.best_params

    # Train the final model with the best hyperparameters on the full dataset
    final_model = xgb.XGBClassifier(
        **best_params,
        random_state=params_optuna["random_state"],
        n_estimators=study.best_trial.user_attrs["n_estimators"],
    )
    final_model.fit(X, y)

    return {"model": final_model, "best_params": best_params}


def compute_results(
    i_fold: int,
    model: str,
    prob: np.ndarray,
    y: np.ndarray | pd.Series,
    result: list[list[Any]],
    rs: bool = False,
    rpn: int = 0,
    ths_range: float | Sequence[float] = 0.5,
    n_removed_features: int = 0,
) -> list[list[Any]]:
    """
    Calculate evaluation metrics by fold and append results to the given list. # noqa

    Parameters:
        i_fold (int): Index of the fold.
        model (str): Model name or identifier.
        rs (bool): Random State.
        rpn (int): Random Permutation number.
        prob (np.ndarray): Probability predictions.
        y (np.ndarray): True labels.
        ths_range (float | Sequence[float]): Thresholds for binary classification.
        n_removed_features (int): Number of removed features.
        result (list[list[Any]]): list to store the results.

    Returns:
        list[list[Any]]: Updated list with appended results.
    """
    # If a single threshold was provided, convert in list for iteration
    if isinstance(ths_range, (int, float)):
        ths_range = [ths_range]

    y = np.asarray(y).astype(int)
    prob = np.asarray(prob, dtype=float)
    positive = y == 1
    n_pos = int(positive.sum())
    n_neg = len(y) - n_pos
    # The AUC does not depend on the threshold
    auc = roc_auc_score(y, prob)

    for ths in ths_range:
        # Calculate the predictions using the passed ths
        prediction = prob > ths
        tp = int(np.sum(prediction & positive))
        fp = int(np.sum(prediction & ~positive))

        # Same values as the sklearn/imblearn metrics (0 when undefined)
        sensitivity = tp / n_pos if n_pos else 0.0
        specificity = (n_neg - fp) / n_neg if n_neg else 0.0
        bacc = (sensitivity + specificity) / 2
        f1 = 2 * tp / (2 * tp + fp + (n_pos - tp)) if tp else 0.0
        recall = sensitivity

        # Append results
        result.append(
            [
                i_fold,
                model,
                rs,
                rpn,
                ths,
                n_removed_features,
                bacc,
                auc,
                f1,
                specificity,
                sensitivity,
                recall,
            ]
        )

    return result


def results_to_df(result: list[list[Any]]) -> pd.DataFrame:
    """
    Convert the list of results to a DataFrame.

    Parameters:
        result (list[list[Any]]): list containing results.

    Returns:
        pd.DataFrame: DataFrame containing results with labeled columns.
    """
    result_df = pd.DataFrame(
        result,
        columns=[
            "Fold",
            "Model",
            "Random State",
            "Random Permutation Number",
            "Thresholds",
            "Number of Removed Features",
            "Balanced ACC",
            "AUC",
            "F1",
            "Specificity",
            "Sensitivity",
            "Recall",
        ],
    )
    return result_df


def save_best_model_params(
    i_fold: int,
    model_name: str,
    model: TunedModel,
    result: list[list[Any]],
    rs: bool = False,
    rpn: int = 0,
) -> list[list[Any]]:
    """
  
    Save the best model parameters to the result list.

    Parameters:
        i_fold (int): Index of the fold.
        model_name (str): Name or identifier of the model.
        model (TunedModel): Output of get_inner_loop_optuna (model and best parameters).
        rs (bool): Random State
        rpn (int): Random Permutation number
        result (list[list[Any]]): list to store the results.

    Returns:
        list[list[Any]]: Updated list with appended results.
    """
    result.append(
        [
            i_fold,
            model_name,
            rs,
            rpn,
            model["model"].n_estimators,
            model["best_params"]["alpha"],
            model["best_params"]["lambda"],
            model["best_params"]["eta"],
            model["best_params"]["max_depth"],
        ]
    )

    return result


def estimator_to_df(results: list[list[Any]]) -> pd.DataFrame:
    """
    Convert results to a DataFrame.

    Args:
        results (list[list[Any]]): list containing results.

    Returns:
        pd.DataFrame: DataFrame containing results.
    """
    columns: list[str] = [
        "Fold",
        "Model",
        "Random State",
        "Random Permutation",
        "Number of Estimators",
        "alpha",
        "lambda",
        "eta",
        "max_depth",
    ]
    return pd.DataFrame(results, columns=columns)


def save_fold_predictions(
    save_dir: Path,
    name: str,
    predictions: Sequence[np.ndarray],
    y_true: Sequence[np.ndarray],
    patient_ids: Sequence[np.ndarray],
) -> None:
    """
    Save out-of-fold predictions in the layout of the plotting notebooks:
    ``<name>_predictions.csv``, ``<name>_y_true.csv`` and
    ``<name>_patient_ID.csv``, one column per outer fold and one row per test
    patient (shorter folds padded with NaN).
    """
    for values, suffix in [
        (predictions, "predictions"),
        (y_true, "y_true"),
        (patient_ids, "patient_ID"),
    ]:
        pd.DataFrame([list(v) for v in values]).T.to_csv(
            Path(save_dir) / f"{name}_{suffix}.csv"
        )
