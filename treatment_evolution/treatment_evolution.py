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
    Applies fuzzy logic to identify treatments in comments and calculate delta treatments.

    Parameters:
    - df (pd.DataFrame): The dataframe containing treatment and comment data.
    - treatments_to_check (List[str]): List of treatments to check in comments.

    Returns:
    - pd.DataFrame: The dataframe with new columns for identified treatments and delta treatments.
    """
    # Threshold for fuzzy matching
    THRESHOLD = 80

    # Helper function to find fuzzy treatments in a comment
    def find_fuzzy_treatment(comment):
        treatments_found = []
        for treatment in treatments_to_check:
            for word in comment.split():
                if fuzz.ratio(treatment.lower(), word.lower()) >= THRESHOLD:
                    treatments_found.append(treatment)
                    break
        return ", ".join(treatments_found) if treatments_found else None

    # Helper function to get the fuzzy delta treatment for a row
    def get_fuzzy_delta_treatment(row):
        treatments_in_comment_list = (
            row["fuzzy_treatments_in_comment"].split(", ")
            if row["fuzzy_treatments_in_comment"]
            else []
        )
        if row["treatment"] in treatments_in_comment_list:
            treatments_in_comment_list.remove(row["treatment"])
        return (
            ", ".join(treatments_in_comment_list)
            if treatments_in_comment_list
            else None
        )

    # Apply the functions
    df["fuzzy_treatments_in_comment"] = df["comment"].apply(find_fuzzy_treatment)
    df["fuzzy_delta_treatment"] = df.apply(get_fuzzy_delta_treatment, axis=1)

    return df


def quantify_treatment_change(row):
    # Check if fuzzy_delta_treatment is None/null
    if pd.isnull(row["fuzzy_delta_treatment"]):
        return None

    # Define the rating quantification based on the given scale
    if 1 <= row["rate"] <= 2:
        return -2
    elif 3 <= row["rate"] <= 4:
        return -1
    elif row["rate"] == 5:
        return 0
    elif 6 <= row["rate"] <= 7:
        return 1
    else:
        return 2


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
    df = df.dropna(subset=["treatment"])

    # Apply fuzzy logic
    df = apply_fuzzy_logic(df, treatments_to_check)

    df["fuzzy_treatment_change_score"] = df.apply(quantify_treatment_change, axis=1)

    df.to_csv(output_path)

    print("------- Treatment Evolution Quantification Completed -------")


if __name__ == "__main__":
    main()
