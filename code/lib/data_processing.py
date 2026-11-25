from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold


def randomly_replace_with_nan(
    df: pd.DataFrame,
    prob: float = 0.1,
    basic_features: list | None = None,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
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
        rng (np.random.Generator): Random generator, for reproducibility.

    Returns:
        pandas.DataFrame: The DataFrame with some values
                          randomly replaced with NaN.
    """
    # Generate a random mask based on the specified probability
    if basic_features is None:
        basic_features = []
    if rng is None:
        rng = np.random.default_rng()
    mask = rng.random(df.shape) < prob

    # Set the elements in basic_features to False in the mask to
    # prevent NaN replacement
    for feature in basic_features:
        mask[:, df.columns.get_loc(feature)] = False

    # Apply the mask to replace values with NaN
    df = df.where(~mask, np.nan)

    return df


def remove_low_variance_features(
    data: pd.DataFrame, variance_ths: float
) -> pd.DataFrame:
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


def naming_for_shap(data_dir: Path, data: pd.DataFrame) -> pd.DataFrame:
    """
    Rename columns of a DataFrame based on a mapping provided in an Excel file.

    Parameters:
    - data_dir (str): The directory containing the Excel file with the column mapping.
    - data (pd.DataFrame): The DataFrame whose columns need to be renamed.

    Returns:
    - pd.DataFrame: The DataFrame with renamed columns.
    """

    # Read the Excel file into a DataFrame
    table = pd.read_excel(data_dir / "Shorten_table.xlsx", sheet_name=None, index_col=0)

    # Specific sheet to use
    table = table["Sheet1"]

    # Create a dictionary for column name mapping
    name_mapping = dict(zip(table["NAME"], table["LABEL"]))

    # Rename columns based on the mapping
    data = data.rename(columns=name_mapping)

    # Replace spaces with underscores
    data.columns = data.columns.str.replace(" ", "_")

    # Replace commas with underscores
    data.columns = data.columns.str.replace(",", "_")

    # Replace square brackets with parentheses
    data.columns = data.columns.str.replace("[", "(")
    data.columns = data.columns.str.replace("]", ")")

    # Replace '<' with 'less'
    data.columns = data.columns.str.replace("<", "less")

    # Replace consecutive underscores with a single underscore
    data.columns = data.columns.str.replace("_-_", "_", regex=True)
    data.columns = data.columns.str.replace("__", "_", regex=True)

    return data


def remove_random_features_fix_number(
    df: pd.DataFrame,
    features_num: int,
    basic_features: Iterable[str] | None = None,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """
    For every row, set ``features_num`` randomly chosen features to NaN,
    never touching ``basic_features``. Each row gets its own random choice.

    The input DataFrame is NOT modified: a masked copy is returned. (The
    previous implementation wrote NaNs into its input, so calling it in a loop
    on the same DataFrame accumulated removals across iterations.)

    A chosen feature that is already missing stays missing, so (as in the
    worst/best-case scenarios) "removed" means "set to missing".

    Parameters:
        df (pandas.DataFrame): The input DataFrame.
        features_num (int): Number of features to remove per row (capped at
                            the number of removable features).
        basic_features (Iterable[str]): Features that are never removed.
        rng (np.random.Generator): Random generator, for reproducibility.
                                   Defaults to a fresh, unseeded generator.

    Returns:
        pandas.DataFrame: Copy of ``df`` with ``features_num`` features
                          set to NaN in every row.
    """
    if rng is None:
        rng = np.random.default_rng()
    basic = set(basic_features) if basic_features is not None else set()

    # Column order (not set order) keeps the choice reproducible for a given rng
    available = [col for col in df.columns if col not in basic]
    features_num = min(features_num, len(available))
    if features_num <= 0 or df.empty:
        return df.copy()

    available_pos = np.array([df.columns.get_loc(col) for col in available])
    # Independent random permutation of the removable features for each row;
    # the first ``features_num`` of each permutation are removed.
    order = np.argsort(rng.random((len(df), len(available))), axis=1)
    removed_pos = available_pos[order[:, :features_num]]

    mask = np.zeros(df.shape, dtype=bool)
    mask[np.arange(len(df))[:, None], removed_pos] = True

    return df.mask(mask)


# Function to round each column of X with different number of digits
def round_columns_with_different_digits(
    X: np.ndarray, rounded_digits: list[int]
) -> np.ndarray:
    """
    Round each column of a 2D array with a different number of decimal places.

    Parameters
    ----------
    X : np.ndarray
        The input 2D array where each column will be rounded.
    rounded_digits : List[int]
        A list of integers specifying the number of decimal places
        to round for each column.

    Returns
    -------
    np.ndarray
        A 2D array with each column rounded to the specified number of decimal places.
    """
    rounded_columns = []
    for i in range(X.shape[1]):  # Loop through each column
        rounded_column = np.round(X[:, i], decimals=rounded_digits[i])
        rounded_columns.append(rounded_column)
    return np.column_stack(rounded_columns)


# Creating Shock Groups
def define_shock_group(row: pd.Series) -> str:
    """
    Define the shock group based on specific conditions.

    Parameters
    ----------
    row : pd.Series
        A row of a DataFrame containing the necessary columns
        to determine the shock group.

    Returns
    -------
    str
        The shock group classification ('C', 'D', 'E', or 'NO').
    """
    if row["lactate_max"] > 2 and row["doubled_creat"] == 1 or row["any_pressor"] >= 1:
        SCAI = "C"
    elif row["total_pressors"] > row["total_pressors_first_hour"]:
        SCAI = "D"
    elif row["any_pressor_first_hour"] >= 2 or row["iabp"] == 1:
        SCAI = "E"
    else:
        SCAI = "NO"

    return SCAI
