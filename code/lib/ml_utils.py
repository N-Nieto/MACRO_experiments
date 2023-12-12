import numpy as np
import pandas as pd
import xgboost as xgb
import optuna
from typing import Dict, Union, List
from sklearn.model_selection import KFold
from mapie.classification import MapieClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from imblearn.metrics import specificity_score, sensitivity_score


def get_inner_loop_optuna(X: pd.DataFrame,
                          y: pd.Series,
                          kf_inner: KFold
                          ) -> Dict[str,
                                    Union[xgb.XGBClassifier,
                                          Dict[str, float], MapieClassifier]]:
    """ # noqa
    Optimize hyperparameters using Optuna for an XGBoost model and train the final model.

    Parameters:
        X (pd.DataFrame): The input features.
        y (pd.Series): The target variable.
        kf_inner (KFold): Inner cross-validation strategy.

    Returns:
        dict: A dictionary containing the final model, best hyperparameters, and Mapie classifier.
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
            "objective": "binary:logistic",
            'eval_metric': 'auc',
            'max_depth': trial.suggest_int('max_depth', 2, 12, log=True),
            'alpha': trial.suggest_float("alpha", 1e-8, 1.0, log=True),
            'lambda': trial.suggest_float("lambda", 1e-8, 1.0, log=True),
            'eta': trial.suggest_float("eta", 0.01, 0.3, log=True),
        }
        # put data in the right format
        dtrain = xgb.DMatrix(X, label=y)
        # allows to remove non-promising trials
        pruning_callback = optuna.integration.XGBoostPruningCallback(trial, "test-auc")         # noqa

        cv_results = xgb.cv(params, dtrain, num_boost_round=10000,
                            early_stopping_rounds=100, maximize=True,
                            callbacks=[pruning_callback], folds=kf_inner)
        # include n_estimators in trial to use it later
        trial.set_user_attr("n_estimators", len(cv_results))

        best_auc = cv_results["test-auc-mean"].values[-1]

        return best_auc

    # Pruner instance
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=5)
    # Study Instance
    sampler = optuna.samplers.TPESampler(seed=23)
    study = optuna.create_study(pruner=pruner,
                                sampler=sampler,
                                direction='maximize')
    # Optimize the objective
    study.optimize(objective, n_trials=100, show_progress_bar=True)
    # Print the best hyperparameters and their value
    best_params = study.best_params

    # Train the final model with the best hyperparameters on the full dataset
    final_model = xgb.XGBClassifier(**best_params,
                                    random_state=23,
                                    n_estimators=study.best_trial.user_attrs["n_estimators"])   # noqa
    final_model.fit(X, y)
    out["model"] = final_model
    out["best_params"] = best_params

    # Create a Mapie classifier using the final model
    # State "prefit" as the model fitted before. Use all data to calibrate
    final_model_mapie = MapieClassifier(estimator=final_model, method="lac",
                                        cv="prefit").fit(X, y)
    out["model_mapie"] = final_model_mapie
    return out


def compute_results_by_fold(i_fold: int, model: str,
                            rs: float, rpn: float, prob: np.ndarray,
                            y: np.ndarray, ths: Union[float, List[float]],
                            result: List[List[Union[int, str, float]]]
                            ) -> List[List[Union[int, str, float]]]:
    """
    Calculate evaluation metrics by fold and append results to the given list.

    Parameters:
        i_fold (int): Index of the fold.
        model (str): Model name or identifier.
        rs (float): Random State.
        rpn (float): Random Permutation number
        prob (np.ndarray): Probability predictions.
        y (np.ndarray): True labels.
        ths (Union[float, List[float]]): Threshold for binary classification.
        result (List[List[Union[int, str, float]]]): List to store the results.

    Returns:
        List[List[Union[int, str, float]]]: Updated list with appended results.
    """
    # Calculate the predictions using the passed ths
    prediction = (prob > ths).astype(int)

    # Compute all the metrics
    bacc = balanced_accuracy_score(y, prediction)
    auc = roc_auc_score(y, prob)
    f1 = f1_score(y, prediction)

    # Append results
    result.append([i_fold, model, rs, rpn, bacc, auc, f1])

    return result


def compute_results_several_ths(i_fold: int, model: str, prob: np.ndarray,
                                y: np.ndarray,
                                ths_range: List[Union[float, List[float]]],
                                result: List[List[Union[int, str, float]]]
                                ) -> List[List[Union[int, str, float]]]:
    """ # noqa
    Calculate evaluation metrics for multiple thresholds and append results to the given list.

    Parameters:
        i_fold (int): Index of the fold.
        model (str): Model name or identifier.
        prob (np.ndarray): Probability predictions.
        y (np.ndarray): True labels.
        ths_range (List[Union[float, List[float]]]): List of thresholds or range of thresholds.
        result (List[List[Union[int, str, float]]]): List to store the results.

    Returns:
        List[List[Union[int, str, float]]]: Updated list with appended results.
    """
    for ths in ths_range:
        # Calculate the predictions using the passed ths
        prediction = (prob > ths).astype(int)

        # Compute all the metrics
        bacc = balanced_accuracy_score(y, prediction)
        auc = roc_auc_score(y, prob)
        f1 = f1_score(y, prediction)

        # Append results
        result.append([i_fold, model, ths, bacc, auc, f1])

    return result


def compute_results_by_fold_and_percentage(i_fold: int, model: str,
                                           perc: float, prob: np.ndarray,
                                           y: np.ndarray,
                                           ths: float,
                                           result: List[List[Union[int,
                                                                   str,
                                                                   float]]]
                                           ) -> List[List[Union[int,
                                                                str,
                                                                float]]]:
    """     # noqa
    Calculate evaluation metrics by fold and percentage and append results to the given list.

    Parameters:
        i_fold (int): Index of the fold.
        model (str): Model name or identifier.
        perc (float): Some percentage value.
        prob (np.ndarray): Probability predictions.
        y (np.ndarray): True labels.
        ths (float): Threshold for binary classification.
        result (List[List[Union[int, str, float]]]): List to store the results.

    Returns:
        List[List[Union[int, str, float]]]: Updated list with appended results.
    """
    # Calculate the predictions using the passed ths

    prediction = (prob > ths).astype(int)
    # Compute all the metrics

    bacc = balanced_accuracy_score(y, prediction)
    auc = roc_auc_score(y, prob)
    f1 = f1_score(y, prediction)
    specificity = specificity_score(y, prediction)
    sensitivity = sensitivity_score(y, prediction)
    precision = precision_score(y, prediction)
    recall = recall_score(y, prediction)
    # Append results

    result.append([i_fold, model, perc, bacc, auc,
                   f1, specificity, sensitivity, precision, recall])

    return result


def save_best_model_params(i_fold: int, model_name: str,
                           model: Dict[str, Union[xgb.XGBClassifier,
                                                  Dict[str, float],
                                                  MapieClassifier]],
                           rs: float, rpn: float,
                           result: List[List[Union[int, str, float]]]
                           ) -> List[List[Union[int, str, float]]]:
    """ # noqa
    Save the best model parameters to the result list.

    Parameters:
        i_fold (int): Index of the fold.
        model_name (str): Name or identifier of the model.
        model (Dict[str, Union[object, Dict[str, Union[int, float]]]]): Dictionary containing the model and its best parameters.
        rs (float): Random State
        rpn (float): Random Permutation number
        result (List[List[Union[int, str, float]]]): List to store the results.

    Returns:
        List[List[Union[int, str, float]]]: Updated list with appended results.
    """
    result.append([i_fold, model_name, rs, rpn, model["model"].n_estimators,
                   model["best_params"]["alpha"],
                   model["best_params"]["lambda"],
                   model["best_params"]["eta"],
                   model["best_params"]["max_depth"]])

    return result
