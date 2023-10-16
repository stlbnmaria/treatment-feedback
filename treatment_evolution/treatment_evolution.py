from fuzzywuzzy import fuzz
import os
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Union
import yaml


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


def find_fuzzy_treatment(
    comment: str, treatments_to_check: List[str], fuzzy_threshold: int
) -> str:
    """
    Helper function to find fuzzy treatments in a comment

    Parameters:
    - comment (str): Comment in the row of the df
    - treatments_to_check (List[str]): List of unique treatments
    - fuzzy_threshold (int): Threshold for fuzzy words

    Returns:
    - str: list of treatments in comments as string
    """
    treatments_found = []
    for treatment in treatments_to_check:
        for word in comment.split():
            if fuzz.ratio(treatment.lower(), word.lower()) >= fuzzy_threshold:
                treatments_found.append(treatment)
                break
    return ", ".join(treatments_found) if treatments_found else None


def get_fuzzy_delta_treatment(row: pd.Series) -> List[str]:
    """
    Helper function to get the fuzzy delta treatment for a row

    Parameters:
    - row (pd.Series): A row of the dataframe

    Returns:
    - List: list of treatments in comments that patient is not taking anymore
    """
    treatments_in_comment_list = (
        row["fuzzy_treatments_in_comment"].split(", ")
        if row["fuzzy_treatments_in_comment"]
        else []
    )
    if row["treatment"] in treatments_in_comment_list:
        treatments_in_comment_list.remove(row["treatment"])
    return treatments_in_comment_list


def apply_fuzzy_logic(
    df: pd.DataFrame, treatments_to_check: List[str], fuzzy_threshold: int
) -> pd.DataFrame:
    """
    Applies fuzzy logic to identify treatments in comments and calculate delta treatments.

    Parameters:
    - df (pd.DataFrame): The dataframe containing treatment and comment data.
    - treatments_to_check (List[str]): List of treatments to check in comments.

    Returns:
    - pd.DataFrame: The dataframe with new columns for identified treatments and delta treatments.
    """

    # Apply the functions
    df["fuzzy_treatments_in_comment"] = df["comment"].apply(
        lambda x: find_fuzzy_treatment(x, treatments_to_check, fuzzy_threshold)
    )
    df["fuzzy_delta_treatment"] = df.apply(get_fuzzy_delta_treatment, axis=1)

    return df


def quantify_treatment_change(row: pd.Series) -> Union[int, None]:
    """
    Gets a score based on the rating.

    Parameters:
    - row (pd.Series): A row of the dataframe

    Returns:
    - int / None: Returns a score in [-2,2] or returns None if there was no treatment evolution.
    """
    # Check if fuzzy_delta_treatment is None/null
    if not row["fuzzy_delta_treatment"]:
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


def main(config_path: Path = Path("config.yaml")):
    # read the path from the config.yaml file
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # define the path to the preprocessed data
    path = Path(config_path.parent / config["file_path"])
    output_path = Path(config_path.parent / config["output_path"])

    # load the dataframe and extract treatments
    df, treatments_to_check = load_and_extract_treatments(path)
    # drop entries with no treatment and anitbody in medication
    df = df.dropna(subset=["treatment"])

    # apply fuzzy logic
    df = apply_fuzzy_logic(df, treatments_to_check, config["fuzzy_threshold"])

    # rate the treatment evolution
    df["fuzzy_treatment_change_score"] = df.apply(quantify_treatment_change, axis=1)

    # subselect columns, drop no treatment evolutions and explode list of previous treatments
    df = df[["text_index", "fuzzy_delta_treatment", "fuzzy_treatment_change_score"]]
    df = df.dropna(subset=["fuzzy_treatment_change_score"])
    df = df.explode("fuzzy_delta_treatment")

    # save to output path
    df.to_csv(output_path)

    print("------- Treatment Evolution Quantification Completed -------")


if __name__ == "__main__":
    main()
