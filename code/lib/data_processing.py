import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold


# Data processing
def randomly_replace_with_nan(df, prob=0.1, basic_features=[]):
    """
    Randomly replaces values in the DataFrame with NaN based
    on the specified probability,
    but leaves the values of basic_features unchanged.

    Parameters:
        df (pandas.DataFrame): The input DataFrame.
        prob (float): Probability of replacing a value with NaN.
                      Default is 0.1 (10%).
        basic_features (list): List of feature names that
                               should never be replaced with NaN.

    Returns:
        pandas.DataFrame: The DataFrame with some values
                          randomly replaced with NaN.
    """
    mask = np.random.rand(*df.shape) < prob
    # Set the elements in basic_features to False in the mask to
    # prevent NaN replacement
    for feature in basic_features:
        mask[:, df.columns.get_loc(feature)] = False
    df = df.where(~mask, np.nan)
    return df


def remove_low_variance_features(data, variance_ths):
    feature_pre = data.shape[1]
    # Create a VarianceThreshold instance with a threshold
    selector = VarianceThreshold(threshold=variance_ths)

    # Fit the selector to your data
    selector.fit(data)

    # Get the indices of the selected features
    selected_features_indices = selector.get_support(indices=True)

    # Filter the DataFrame to keep only the selected features
    data = data.iloc[:, selected_features_indices]
    feature_post = data.shape[1]
    print(str(feature_pre-feature_post) + " Features deleted")
    return data


def naming_for_shap(data_dir, data):

    table = pd.read_excel(data_dir + "Shorten_table.xlsx", sheet_name=None,
                          index_col=0)
    table = table['Sheet1']
    name_mapping = dict(zip(table['NAME'], table['LABEL']))

    data.rename(columns=name_mapping, inplace=True)
    data.columns = data.columns.str.replace(' ', '_')
    data.columns = data.columns.str.replace(",", "_")
    data.columns = data.columns.str.replace('[', '(')
    data.columns = data.columns.str.replace(']', ')')
    data.columns = data.columns.str.replace("<", "less")
    data.columns = data.columns.str.replace("_-_", "_", regex=True)
    data.columns = data.columns.str.replace("__", "_", regex=True)

    return data
