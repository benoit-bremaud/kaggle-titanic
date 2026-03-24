"""Feature engineering functions for the Titanic competition.

All functions follow the Train/Transform Split pattern (ADR-019):
- They take a DataFrame and transform it in place or return a new column.
- Statistics (e.g., medians, mappings) are passed as parameters, not computed
  inside the function, so the caller controls what was fit on train only.
"""

import pandas as pd


def extract_title(df: pd.DataFrame) -> pd.Series:
    """Extract title from the Name column and group rare titles.

    Args:
        df: DataFrame with a 'Name' column (format: "Last, Title. First").

    Returns:
        Series with grouped titles: Mr, Miss, Mrs, Master, Rare.
    """
    title = df["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)

    title_mapping = {
        "Mr": "Mr",
        "Miss": "Miss",
        "Ms": "Miss",
        "Mlle": "Miss",
        "Mrs": "Mrs",
        "Mme": "Mrs",
        "Master": "Master",
    }
    title = title.map(lambda t: title_mapping.get(t, "Rare"))
    return title


def create_family_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create FamilySize and IsAlone features.

    Args:
        df: DataFrame with 'SibSp' and 'Parch' columns.

    Returns:
        DataFrame with added 'FamilySize' and 'IsAlone' columns.
    """
    df = df.copy()
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
    return df


def encode_sex(df: pd.DataFrame) -> pd.Series:
    """Encode Sex as binary: female=1, male=0.

    Args:
        df: DataFrame with a 'Sex' column.

    Returns:
        Series with encoded values.
    """
    return df["Sex"].map({"female": 1, "male": 0})


def encode_embarked(df: pd.DataFrame) -> pd.Series:
    """Encode Embarked as integers: S=0, C=1, Q=2.

    Args:
        df: DataFrame with an 'Embarked' column.

    Returns:
        Series with encoded values.
    """
    return df["Embarked"].map({"S": 0, "C": 1, "Q": 2})


def encode_title(title_series: pd.Series) -> pd.Series:
    """Encode Title as integers: Mr=0, Miss=1, Mrs=2, Master=3, Rare=4.

    Args:
        title_series: Series with grouped title strings.

    Returns:
        Series with encoded values.
    """
    mapping = {"Mr": 0, "Miss": 1, "Mrs": 2, "Master": 3, "Rare": 4}
    return title_series.map(mapping)


COLUMNS_TO_DROP = ["Name", "Ticket", "PassengerId"]
"""Columns to drop before modeling — not predictive, just noise."""


def apply_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all feature engineering steps to a DataFrame.

    This is the main entry point called from the notebook.
    It applies all transformations in order and drops unused columns.

    Args:
        df: Raw DataFrame (after data cleaning).

    Returns:
        Transformed DataFrame ready for modeling.
    """
    df = create_family_features(df)
    df["Title"] = extract_title(df)
    df["Title"] = encode_title(df["Title"])
    df["Sex"] = encode_sex(df)
    df["Embarked"] = encode_embarked(df)

    cols_to_drop = [c for c in COLUMNS_TO_DROP if c in df.columns]
    df = df.drop(columns=cols_to_drop)

    return df
