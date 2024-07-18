import pandas as pd
import xgboost as xgb
import optuna
from typing import Dict
from sklearn.model_selection import KFold


class XGBClassifier_optuna_es():

    def __init__(self,
                 kf_inner: KFold,
                 params_optuna: Dict[str, float]) -> None:

        self.kf_inner = kf_inner
        self.params_optuna = params_optuna

        return

    def fit(self,
            X: pd.DataFrame,
            y: pd.Series,
            ) -> xgb.XGBClassifier:
        """
        Get the best model with OPTUNA
        
        Optimize hyperparameters using Optuna for an XGBoost model and train the final model.
        # noqa
        Parameters:
            X (pd.DataFrame): The input features.
            y (pd.Series): The target variable.

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
                "objective": self.params_optuna["objective"],
                'eval_metric': self.params_optuna["eval_metric"],
                'max_depth': trial.suggest_int('max_depth',
                                               self.params_optuna["max_depth_min"],         # noqa
                                               self.params_optuna["max_depth_max"],         # noqa
                                               log=True),
                'alpha': trial.suggest_float("alpha",
                                             self.params_optuna["alpha_min"],
                                             self.params_optuna["alpha_max"],
                                             log=True),
                'lambda': trial.suggest_float("lambda",
                                              self.params_optuna["lambda_min"],
                                              self.params_optuna["lambda_max"],
                                              log=True),
                'eta': trial.suggest_float("eta",
                                           self.params_optuna["eta_min"],
                                           self.params_optuna["eta_max"],
                                           log=True),
            }
            # put data in the right format
            dtrain = xgb.DMatrix(X, label=y)
            # allows to remove non-promising trials
            pruning_callback = optuna.integration.XGBoostPruningCallback(trial, "test-auc")                 # noqa

            cv_results = xgb.cv(params, dtrain,
                                num_boost_round=self.params_optuna["num_boost_round"],                      # noqa
                                early_stopping_rounds=self.params_optuna["early_stopping_rounds"],          # noqa
                                maximize=True,
                                callbacks=[pruning_callback],
                                folds=self.kf_inner)
            # include n_estimators in trial to use it later
            trial.set_user_attr("n_estimators", len(cv_results))

            best_auc = cv_results["test-auc-mean"].values[-1]

            return best_auc

        # Pruner instance
        pruner = optuna.pruners.MedianPruner(n_warmup_steps=5)
        # Study Instance
        sampler = optuna.samplers.TPESampler(seed=self.params_optuna["random_state"])                       # noqa
        study = optuna.create_study(pruner=pruner,
                                    sampler=sampler,
                                    direction='maximize')
        # Optimize the objective
        study.optimize(objective, n_trials=self.params_optuna["optuna_trials"],
                       show_progress_bar=True)
        # Print the best hyperparameters and their value
        best_params = study.best_params

        # Train the final model with the best hyperparameters
        # on the full dataset
        final_model = xgb.XGBClassifier(**best_params,
                                        random_state=self.params_optuna["random_state"],            # noqa
                                        n_estimators=study.best_trial.user_attrs["n_estimators"])   # noqa
        final_model.fit(X, y)

        return final_model

    def predict(self, X):
        return super().predict(X)

    def score(self, X, y):
        return super().score(X, y)

    def predict_proba(self, X):
        return super().predict_proba(X)

# %%
