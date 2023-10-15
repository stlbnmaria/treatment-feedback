from fuzzywuzzy import fuzz
import os
import pandas as pd
from typing import List, Tuple


def load_and_extract_treatments(path: str) -> Tuple[pd.DataFrame, List[str]]:
    """
    Loads the dataframe and extracts unique treatments.

    Parameters:
    - path (str): Path to the preprocessed CSV file.

    Returns:
    - Tuple: A tuple containing the loaded dataframe and the list of unique treatments.
    """
    df = pd.read_csv(path)
    treatments_to_check = df["treatment"].dropna().unique().tolist()
    return df, treatments_to_check


def apply_fuzzy_logic(df: pd.DataFrame, treatments_to_check: List[str]) -> pd.DataFrame:
    """
    Applies fuzzy logic to identify treatments in comments and calculate treatment change scores.

    Parameters:
    - df (pd.DataFrame): The dataframe to process.
    - treatments_to_check (List[str]): List of treatments to check for.

    Returns:
    - pd.DataFrame: Processed dataframe with added columns for fuzzy treatments in comments, delta treatment, and treatment change score.
    """
    THRESHOLD = 80
    df["fuzzy_treatments_in_comment"] = df["comment"].apply(
        lambda comment: ", ".join(
            [
                treatment
                for treatment in treatments_to_check
                if fuzz.ratio(treatment.lower(), str(comment).lower()) >= THRESHOLD
            ]
        )
    )

    df["fuzzy_delta_treatment"] = df.apply(
        lambda row: ", ".join(
            [
                treatment
                for treatment in row["fuzzy_treatments_in_comment"].split(", ")
                if treatment != row["treatment"]
            ]
        )
        if row["fuzzy_treatments_in_comment"]
        else None,
        axis=1,
    )

    df["fuzzy_treatment_change_score"] = df.apply(
        lambda row: -2
        if 1 <= row["rate"] <= 2
        else (
            -1
            if 3 <= row["rate"] <= 4
            else (0 if row["rate"] == 5 else (1 if 6 <= row["rate"] <= 7 else 2))
        ),
        axis=1,
    )

    return df


def main():
    # Define the path to the preprocessed data
    path = os.path.join(
        os.getcwd(), "..", "data_preprocessing", "data", "preprocessed.csv"
    )
    output_path = os.path.join(
        os.getcwd(), "..", "data_preprocessing", "data", "treatment_evolution.csv"
    )

    # Load the dataframe and extract treatments
    df, treatments_to_check = load_and_extract_treatments(path)

    # Apply fuzzy logic
    df = apply_fuzzy_logic(df, treatments_to_check)

    df.to_csv(output_path)

    print("------- Treatment Evolution Quantification Completed -------")


if __name__ == "__main__":
    main()
