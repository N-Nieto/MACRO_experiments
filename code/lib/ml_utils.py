import numpy as np
import pandas as pd
import julearn as jl
import optuna
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from imblearn.metrics import specificity_score, sensitivity_score
from lib.data_load_utils import load_table
from mapie.classification import MapieClassifier


def feature_analysis(features, results, data_dir, scoring, final_data):
    importance_scores = np.zeros(len(features))
    # Get the feature importance scores
    for cv in range(results.shape[0]):
        # Get the estimator for each cross-validation
        clf = results["estimator"][cv]
        # Sum the importance for each feature
        importance_scores += clf.get_params()["xgbclassifier"].feature_importances_  # noqa

    # Normalize the imporance to 1
    importance_scores = importance_scores / importance_scores.max()
    # Generate a dataframe with the importance
    features_importance = pd.DataFrame({"Importance": importance_scores})
    features_importance["Feature Code"] = features
    # Sort the more important features in the top
    features_importance.sort_values(by="Importance", inplace=True,
                                    ascending=False)

    # Compute the feature frecuency
    features_frec = pd.DataFrame(final_data.notna().sum())
    features_frec.drop("patient_ID", inplace=True)
    features_frec["Feature Code"] = features_frec.index
    features_frec.rename(columns={0: "Frecuency"}, inplace=True)
    # Normalize the Frecuency. Percentage of patients with information
    features_frec["Frecuency"] = 100 * (features_frec["Frecuency"] / features_frec["Frecuency"].max()) # noqa
    # Meage both importance and Frecuency
    features_information = pd.merge(left=features_importance,
                                    right=features_frec,
                                    on="Feature Code")
    # Load table with features names
    table = load_table(data_dir)
    features_information = features_information.merge(table,
                                                      left_on='Feature Code',
                                                      right_on='NAME',
                                                      how='left')
    features_information.drop(['NAME', "MEMNAME"], axis=1, inplace=True)
    features_information.rename(columns={'LABEL': 'Feature Name'},
                                inplace=True)
    features_information = features_information.reindex(columns=["Feature Code", # noqa
                                                                 "Feature Name", # noqa
                                                                 "Frecuency",
                                                                 "Importance"])

    # Saving results
    results_to_save = results.loc[:, ["test_roc_auc", "test_accuracy",
                                  "test_f1", "repeat", "fold"]]
    # Compute also the mean scores to save
    # Initialize list
    mean_results = []
    col_name = []
    for score in scoring:
        # Append the results for each score
        mean_results.append(results_to_save["test_"+score].mean())
        col_name.append("test_"+score)

    # Put in a dataframe to save
    mean_results = pd.DataFrame(mean_results)
    # Change format for a consistent saving
    mean_results = mean_results.T
    mean_results.columns = col_name

    return features_information, results_to_save, mean_results


def performance_drop_by_feature(final_data, results, features,
                                endpoint_to_use, model,
                                scoring,
                                features_information,
                                random_state):
    results_drop = pd.DataFrame(columns=features)

    for feature in features:
        data_loop = final_data.drop(columns=feature)
        features_loop = [elem for elem in features if elem != feature]
        results_loop = jl.run_cross_validation(X=features_loop,
                                               y=endpoint_to_use,
                                               model=model, data=data_loop,
                                               scoring=scoring, n_jobs=-1,
                                               return_estimator="cv",
                                               seed=random_state
                                               )
        # Get the mean scores
        print("Performance without " + feature)
        # save the difference in performance between the model with and
        # without the feature
        valid_columns = results_loop.select_dtypes(include=[np.number]).columns
        results_loop = results_loop[valid_columns].mean()
        results_mean = results[valid_columns].mean()
        results_drop[feature] = results_loop - results_mean

    # Formating for saving
    results_drop = results_drop.T
    # Get the features codes
    results_drop["Feature Code"] = results_drop.index

    # Merge this results with the previus feature information
    features_information = features_information.merge(results_drop,
                                                      left_on='Feature Code',
                                                      right_on='Feature Code',
                                                      how='left')
    # Remove those variables not needed
    features_information.drop(['fit_time', "score_time",
                              "repeat", "fold"], axis=1, inplace=True)
    # Rename the scorings
    for score in scoring:
        features_information.rename(columns={"test_"+score: "Drop in "+score},
                                    inplace=True)

    return features_information


def get_predictions_xgb_early_stopping(X_train, Y_train, X_test,
                                       params):
    out = {}
    # Split the train data in train and validation
    X_inner_train, X_val, \
        y_train_inner, y_val = train_test_split(
                                        X_train,
                                        Y_train,
                                        test_size=params['val_percentage'],
                                        random_state=params['random_state'],
                                        stratify=Y_train)

    # Build a first model
    model = XGBClassifier(n_estimators=params['initial_n_estimators'],
                          n_jobs=-1,
                          reg_alpha=params['reg_alpha'],
                          reg_lambda=params['reg_lambda'],
                          missing=-999,
                          eval_metric=params['eval_metric'],
                          early_stopping_rounds=params['early_stopping_rounds'], # noqa
                          random_state=params['random_state'],
                          verbosity=0)

    # Model fit to find the best model parameters
    model.fit(X=X_inner_train, y=y_train_inner,
              eval_set=[(X_val, y_val)], verbose=0)

    # Create a model with the max iterations set as the best epoches
    model = XGBClassifier(n_estimators=model.best_ntree_limit,
                          n_jobs=-1,
                          reg_alpha=params['reg_alpha'],
                          reg_lambda=params['reg_lambda'],
                          missing=-999,
                          eval_metric=params['eval_metric'],
                          random_state=params['random_state'],
                          verbosity=0)

    # Model fit in the whole data and the best number of trees
    model.fit(X=X_train, y=Y_train)
    # get the predictions with the final model over the test data
    y_pred = model.predict_proba(X_test)[:, 1]

    # Return the predictions and the model
    out['model'] = model
    out['pred'] = y_pred
    return out


def get_inner_loop_predictions(X, Y, kf, params):

    out = {}
    cv_preds = None

    for i_fold, (inner_train_index, val_index) in enumerate(kf.split(X, Y)):       # noqa

        # Patients used for train and internal XGB validation
        X_inner_train = X[inner_train_index]
        Y_inner_train = Y[inner_train_index]

        # Patients used to generete a prediction
        X_val = X[val_index]

        # Get the predictions of the admission model trained with early stopping    # noqa
        # and retrained over all the train data
        model = get_predictions_xgb_early_stopping(
                                    X_inner_train,
                                    Y_inner_train,
                                    X_val,
                                    params=params)

        # Store 24hs predictions
        if cv_preds is None:
            cv_preds = np.ones((X.shape[0])) * -1
        cv_preds[val_index] = model["pred"]

    # Get the best model obtained in the early stopping
    # and fit it on the whole train data
    model["model"].fit(X, Y)

    out["model"] = model["model"]
    out["pred"] = cv_preds
    return out


def get_inner_loop_predictions_df(X, Y, kf, params):

    out = {}
    cv_preds = np.ones((X.shape[0])) * -1
    for inner_train_index, val_index in kf.split(X, Y):

        # Patients used for train and internal XGB validation
        X_inner_train = X.iloc[inner_train_index, :]
        Y_inner_train = Y[inner_train_index]

        # Patients used to generete a prediction
        X_val = X.iloc[val_index, :]

        # Get the predictions of the admission model
        # trained with early stopping
        # and retrained over all the train data
        model = get_predictions_xgb_early_stopping(
                                    X_inner_train,
                                    Y_inner_train,
                                    X_val,
                                    params=params)

        # Store 24hs predictions
        cv_preds[val_index] = model["pred"]

    # Get the best model obtained in the early stopping
    # and fit it on the whole train data
    model["model"].fit(X, Y)

    out["model"] = model["model"]
    out["pred"] = cv_preds

    return out


def get_combine_model(admission_preds, cascade_norisk_preds=None, cascade_risk_preds=None, thr=None): # noqa
    # If only one cascade model is applied
    if cascade_norisk_preds is None:
        cascade_norisk_preds = admission_preds
    if cascade_risk_preds is None:
        cascade_risk_preds = admission_preds
    # Get those patients where the admission model predict 0
    admission_prediction = (admission_preds > thr).astype(int)
    # Initialize
    combined_proba = -1*np.ones(admission_preds.shape)
    # Get those patients predicted as No Risk
    mask = admission_prediction == 0

    # Replace the No Risk with the predictions using the cascade
    combined_proba[mask] = cascade_norisk_preds[mask]

    # Repeat for Risk patients
    mask = admission_prediction == 1

    combined_proba[mask] = cascade_risk_preds[mask]

    # Now the admission model is the combined model
    return combined_proba


def compute_results_by_fold(i_fold, model, rs, rpn, prob, y, ths, result):
    # Calculate the predictions using the passed ths
    prediction = (prob > ths).astype(int)
    # compute all the metrics
    bacc = balanced_accuracy_score(y, prediction)
    auc = roc_auc_score(y, prob)
    f1 = f1_score(y, prediction)
    specificity = specificity_score(y, prediction)
    sensitivity = sensitivity_score(y, prediction)
    # append results
    result.append([i_fold, model, rs, rpn, bacc, auc, f1, specificity,
                   sensitivity])
    return result


def compute_results_several_ths(i_fold, model, prob, y,
                                ths_range, result):
    for ths in ths_range:
        # Calculate the predictions using the passed ths
        prediction = (prob > ths).astype(int)
        # compute all the metrics
        bacc = balanced_accuracy_score(y, prediction)
        auc = roc_auc_score(y, prob)
        f1 = f1_score(y, prediction)
        specificity = specificity_score(y, prediction)
        sensitivity = sensitivity_score(y, prediction)
        # append results
        result.append([i_fold, model, ths, bacc, auc, f1, specificity,
                       sensitivity])
    return result


def compute_results_by_fold_and_percentage(i_fold, model, perc, prob, y, ths,
                                           result):
    prediction = (prob > ths).astype(int)

    bacc = balanced_accuracy_score(y, prediction)
    auc = roc_auc_score(y, prob)
    f1 = f1_score(y, prediction)
    specificity = specificity_score(y, prediction)
    sensitivity = sensitivity_score(y, prediction)
    precision = precision_score(y, prediction)
    recall = recall_score(y, prediction)

    result.append([i_fold, model, perc, bacc, auc, f1, specificity,
                   sensitivity, precision, recall])

    return result


def save_best_model_params(i_fold, model_name, model, rs, rpn,  result):

    result.append([i_fold, model_name, rs, rpn, model["model"].n_estimators,
                   model["best_params"]["alpha"],
                   model["best_params"]["lambda"],
                  model["best_params"]["eta"],
                  model["best_params"]["max_depth"]])

    return result


def get_inner_loop_optuna(X, y, kf_inner):
    out = {}

    def objective(trial):
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
        # allows to remove non promessing trials
        pruning_callback = optuna.integration.XGBoostPruningCallback(trial, "test-auc")             # noqa

        cv_resuts = xgb.cv(params, dtrain, num_boost_round=10000,
                           early_stopping_rounds=100, maximize=True,
                           callbacks=[pruning_callback], folds=kf_inner)
        # include n_estimators in trial to use it later
        trial.set_user_attr("n_estimators", len(cv_resuts))

        best_auc = cv_resuts["test-auc-mean"].values[-1]

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
                                    n_estimators=study.best_trial.user_attrs["n_estimators"])           # noqa
    final_model.fit(X, y)
    out["model"] = final_model
    out["best_params"] = best_params

    # Create a Mapie classifier using the final model
    # State "prefit" as the model fitted before. Use all data to calibrate
    final_model_mapie = MapieClassifier(estimator=final_model, method="lac",
                                        cv="prefit").fit(X, y)
    out["model_mapie"] = final_model_mapie
    return out
