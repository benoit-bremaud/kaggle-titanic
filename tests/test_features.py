"""Unit tests for src/features.py."""

import pandas as pd
import pytest

from src.features import (
    apply_feature_engineering,
    create_family_features,
    encode_embarked,
    encode_sex,
    encode_title,
    extract_title,
)


@pytest.fixture()
def sample_df() -> pd.DataFrame:
    """Create a minimal Titanic-like DataFrame for testing."""
    return pd.DataFrame(
        {
            "PassengerId": [1, 2, 3],
            "Name": [
                "Braund, Mr. Owen Harris",
                "Cumings, Mrs. John Bradley",
                "Heikkinen, Miss. Laina",
            ],
            "Sex": ["male", "female", "female"],
            "SibSp": [1, 1, 0],
            "Parch": [0, 0, 0],
            "Ticket": ["A/5 21171", "PC 17599", "STON/O2. 3101282"],
            "Embarked": ["S", "C", "S"],
        }
    )


class TestExtractTitle:
    def test_basic_titles(self, sample_df: pd.DataFrame) -> None:
        titles = extract_title(sample_df)
        assert list(titles) == ["Mr", "Mrs", "Miss"]

    def test_rare_title(self) -> None:
        df = pd.DataFrame({"Name": ["Smith, Dr. John"]})
        assert extract_title(df).iloc[0] == "Rare"

    def test_mlle_mapped_to_miss(self) -> None:
        df = pd.DataFrame({"Name": ["Dupont, Mlle. Marie"]})
        assert extract_title(df).iloc[0] == "Miss"

    def test_mme_mapped_to_mrs(self) -> None:
        df = pd.DataFrame({"Name": ["Dupont, Mme. Marie"]})
        assert extract_title(df).iloc[0] == "Mrs"


class TestCreateFamilyFeatures:
    def test_family_size(self, sample_df: pd.DataFrame) -> None:
        result = create_family_features(sample_df)
        assert list(result["FamilySize"]) == [2, 2, 1]

    def test_is_alone(self, sample_df: pd.DataFrame) -> None:
        result = create_family_features(sample_df)
        assert list(result["IsAlone"]) == [0, 0, 1]

    def test_does_not_modify_original(self, sample_df: pd.DataFrame) -> None:
        create_family_features(sample_df)
        assert "FamilySize" not in sample_df.columns


class TestEncodings:
    def test_encode_sex(self, sample_df: pd.DataFrame) -> None:
        result = encode_sex(sample_df)
        assert list(result) == [0, 1, 1]

    def test_encode_embarked(self, sample_df: pd.DataFrame) -> None:
        result = encode_embarked(sample_df)
        assert list(result) == [0, 1, 0]

    def test_encode_title(self) -> None:
        titles = pd.Series(["Mr", "Miss", "Mrs", "Master", "Rare"])
        result = encode_title(titles)
        assert list(result) == [0, 1, 2, 3, 4]


class TestApplyFeatureEngineering:
    def test_output_columns(self, sample_df: pd.DataFrame) -> None:
        result = apply_feature_engineering(sample_df)
        assert "FamilySize" in result.columns
        assert "IsAlone" in result.columns
        assert "Title" in result.columns

    def test_drops_unused_columns(self, sample_df: pd.DataFrame) -> None:
        result = apply_feature_engineering(sample_df)
        assert "Name" not in result.columns
        assert "Ticket" not in result.columns
        assert "PassengerId" not in result.columns

    def test_does_not_modify_original(self, sample_df: pd.DataFrame) -> None:
        apply_feature_engineering(sample_df)
        assert "Name" in sample_df.columns

    def test_sex_is_numeric(self, sample_df: pd.DataFrame) -> None:
        result = apply_feature_engineering(sample_df)
        assert result["Sex"].dtype in [int, "int64"]

    def test_embarked_is_numeric(self, sample_df: pd.DataFrame) -> None:
        result = apply_feature_engineering(sample_df)
        assert result["Embarked"].dtype in [int, "int64"]
