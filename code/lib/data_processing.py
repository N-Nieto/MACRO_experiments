import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from typing import List


def randomly_replace_with_nan(df: pd.DataFrame, prob: float = 0.1,
                              basic_features: list = []) -> pd.DataFrame:
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
    # Generate a random mask based on the specified probability
    mask = np.random.rand(*df.shape) < prob

    # Set the elements in basic_features to False in the mask to
    # prevent NaN replacement
    for feature in basic_features:
        mask[:, df.columns.get_loc(feature)] = False

    # Apply the mask to replace values with NaN
    df = df.where(~mask, np.nan)

    return df


def remove_low_variance_features(data: pd.DataFrame,
                                 variance_ths: float) -> pd.DataFrame:
    """
    Remove low-variance features from a DataFrame.

    Parameters:
    - data (pd.DataFrame): The input DataFrame.
    - variance_ths (float): The variance threshold for feature selection.

    Returns:
    - pd.DataFrame: The DataFrame with low-variance features removed.
    """

    # Record the number of features before the removal
    feature_pre = data.shape[1]

    # Create a VarianceThreshold instance with the specified threshold
    selector = VarianceThreshold(threshold=variance_ths)

    # Fit the selector to your data
    selector.fit(data)

    # Get the indices of the selected features
    selected_features_indices = selector.get_support(indices=True)

    # Filter the DataFrame to keep only the selected features
    data = data.iloc[:, selected_features_indices]

    # Record the number of features after the removal
    feature_post = data.shape[1]

    # Print the number of features deleted
    print(str(feature_pre - feature_post) + " Features deleted")

    return data


def naming_for_shap(data_dir: str, data: pd.DataFrame) -> pd.DataFrame:
    """
    # noqa
    Rename columns of a DataFrame based on a mapping provided in an Excel file.

    Parameters:
    - data_dir (str): The directory containing the Excel file with the column mapping.
    - data (pd.DataFrame): The DataFrame whose columns need to be renamed.

    Returns:
    - pd.DataFrame: The DataFrame with renamed columns.
    """

    # Read the Excel file into a DataFrame
    table = pd.read_excel(data_dir + "Shorten_table.xlsx",
                          sheet_name=None, index_col=0)

    # Specific sheet to use
    table = table['Sheet1']

    # Create a dictionary for column name mapping
    name_mapping = dict(zip(table['NAME'], table['LABEL']))

    # Rename columns based on the mapping
    data.rename(columns=name_mapping, inplace=True)

    # Replace spaces with underscores
    data.columns = data.columns.str.replace(' ', '_')

    # Replace commas with underscores
    data.columns = data.columns.str.replace(",", "_")

    # Replace square brackets with parentheses
    data.columns = data.columns.str.replace('[', '(')
    data.columns = data.columns.str.replace(']', ')')

    # Replace '<' with 'less'
    data.columns = data.columns.str.replace("<", "less")

    # Replace consecutive underscores with a single underscore
    data.columns = data.columns.str.replace("_-_", "_", regex=True)
    data.columns = data.columns.str.replace("__", "_", regex=True)

    return data


def remove_random_features_fix_number(df: pd.DataFrame,
                                      features_num: int,
                                      basic_features: list = []
                                      ) -> pd.DataFrame:
    """
    # noqa
    Remove the same number of features from the DataFrame for each row randomly,
    leaving the values of basic_features unchanged.

    Parameters:
        df (pandas.DataFrame): The input DataFrame.
        features_num (int): The number of features to be removed for each row.
        basic_features (list): List of feature names that
                               should never be removed.

    Returns:
        pandas.DataFrame: The DataFrame with the same number of features randomly removed for each row.
    """
    # Determine the available features excluding basic_features
    available_features = list(set(df.columns) - set(basic_features))

    # Ensure the specified number of features to remove
    # is not greater than the available features
    features_num = min(features_num, len(available_features))

    # Iterate through each row and drop
    # the specified number of features randomly
    for i in range(df.shape[0]):
        # Shuffle the order of available features
        shuffled_features = np.random.choice(available_features,
                                             size=features_num, replace=False)

        # Drop the identified features for the current row
        df.iloc[i, df.columns.isin(shuffled_features)] = np.nan

    return df


# Function to round each column of X with different number of digits
def round_columns_with_different_digits(X: np.ndarray,
                                        rounded_digits: List[int]
                                        ) -> np.ndarray:
    rounded_columns = []
    for i in range(X.shape[1]):  # Loop through each column
        rounded_column = np.round(X[:, i], decimals=rounded_digits[i])
        rounded_columns.append(rounded_column)
    return np.column_stack(rounded_columns)


# Creating Shock Groups
def define_shock_group(row):

    if row['lactate_max'] > 2 and row['doubled_creat'] == 1 or row['any_pressor'] >= 1: # noqa
        SCAI = "C"
    elif row['total_pressors'] > row['total_pressors_first_hour']:
        SCAI = "D"
    elif row['any_pressor_first_hour'] >= 2 or row['iabp'] == 1:
        SCAI = "E"
    else:
        SCAI = "NO"

    return SCAI
