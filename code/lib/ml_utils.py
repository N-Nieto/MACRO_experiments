import optuna
import numpy as np
import pandas as pd
import xgboost as xgb
from typing import Dict, Union, List, Any
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    f1_score,
    recall_score,
    balanced_accuracy_score,
    roc_auc_score,
)
from imblearn.metrics import specificity_score, sensitivity_score


def get_inner_loop_optuna(
    X: pd.DataFrame,
    y: pd.Series,
    kf_inner: StratifiedKFold,
    params_optuna: Dict[str, float],
) -> Dict[str, Union[xgb.XGBClassifier, Dict[str, float]]]:
    """
    Get the best model with OPTUNA

    Optimize hyperparameters using Optuna for an XGBoost model and train the final model.
    # noqa
    Parameters:
        X (pd.DataFrame): The input features.
        y (pd.Series): The target variable.
        kf_inner (KFold): Inner cross-validation strategy.

    Returns:
        dict: A dictionary containing the final model, best hyperparameters.
    """
    out = {}

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
        pruning_callback = optuna.integration.XGBoostPruningCallback(trial, "test-auc")  # noqa

        cv_results = xgb.cv(
            params,
            dtrain,
            num_boost_round=params_optuna["num_boost_round"],
            early_stopping_rounds=params_optuna["early_stopping_rounds"],  # noqa
            maximize=True,
            callbacks=[pruning_callback],
            folds=kf_inner,
        )
        # include n_estimators in trial to use it later
        trial.set_user_attr("n_estimators", len(cv_results))

        best_auc = cv_results["test-auc-mean"].values[-1]

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
    out["model"] = final_model
    out["best_params"] = best_params

    return out


def compute_results(
    i_fold: int,
    model: str,
    prob: np.ndarray,
    y: np.ndarray,
    result: List[List[Union[int, str, float]]],
    rs: bool = False,
    rpn: int = 0,
    ths_range: Union[float, List[float]] = 0.5,
    n_removed_features: int = 0,
) -> List[List[Union[int, str, float]]]:
    """
    Calculate evaluation metrics by fold and append results to the given list. # noqa

    Parameters:
        i_fold (int): Index of the fold.
        model (str): Model name or identifier.
        rs (bool): Random State.
        rpn (int): Random Permutation number.
        prob (np.ndarray): Probability predictions.
        y (np.ndarray): True labels.
        ths_range (Union[float, List[float]]): Thresholds for binary classification.
        n_removed_features (int): Number of removed features.
        result (List[List[Union[int, str, float]]]): List to store the results.

    Returns:
        List[List[Union[int, str, float]]]: Updated list with appended results.
    """
    # If a float value was provided, convert in list for iteration
    if isinstance(ths_range, float):
        ths_range = [ths_range]

    for ths in ths_range:
        # Calculate the predictions using the passed ths
        prediction = (prob > ths).astype(int)

        # Compute all the metrics
        bacc = balanced_accuracy_score(y, prediction)
        auc = roc_auc_score(y, prob)
        f1 = f1_score(y, prediction)
        specificity = specificity_score(y, prediction)
        sensitivity = sensitivity_score(y, prediction)
        recall = recall_score(y, prediction)

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


def results_to_df(result: List[List[Union[int, str, float]]]) -> pd.DataFrame:
    """
    Convert the list of results to a DataFrame.

    Parameters:
        result (List[List[Union[int, str, float]]]): List containing results.

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
    model: Dict[str, Union[xgb.XGBClassifier, Dict[str, float]]],
    result: List[List[Union[int, str, float]]],
    rs: bool = False,
    rpn: int = 0,
) -> List[List[Union[int, str, float]]]:
    """
    # noqa
    Save the best model parameters to the result list.

    Parameters:
        i_fold (int): Index of the fold.
        model_name (str): Name or identifier of the model.
        model (Dict[str, Union[object, Dict[str, Union[int, float]]]]): Dictionary containing the model and its best parameters.
        rs (bool): Random State
        rpn (int): Random Permutation number
        result (List[List[Union[int, str, float]]]): List to store the results.

    Returns:
        List[List[Union[int, str, float]]]: Updated list with appended results.
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


def estimator_to_df(results: List[List[Any]]) -> pd.DataFrame:
    """
    Convert results to a DataFrame.

    Args:
        results (List[List[Any]]): List containing results.

    Returns:
        pd.DataFrame: DataFrame containing results.
    """
    columns: List[str] = [
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
