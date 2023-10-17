import pandas as pd
from pathlib import Path
from transformers import pipeline
from typing import List


def phrase_classification(
    df: pd.DataFrame,
    file_path: Path,
    category_labels: List[str],
    column_name_phrase: str = "phrases",
) -> pd.DataFrame:
    """classifies the extracted phrases into topics

    Args:
        - df (pd.DataFrame): the input dataframe
        - file_path (Path): file for output csv
        - category_labels (List[str]): list of topics in config.yaml
        - column_name_phrase (str): the row name of the dataframe
            that contains the phrases that should be classified

    Returns:
        pd.DataFrame: the output dataframe in which each phrase
        is represented by a row
    """

    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

    # Create a list to store the new rows
    new_rows = []

    # Iterate through the original DataFrame
    for _, row in df.iterrows():
        phrases = row[column_name_phrase]

        # If there are phrases, classify and add new rows
        if phrases:
            for phrase in phrases:
                result = classifier(phrase, category_labels)
                categories = result["labels"]
                scores = result["scores"]
                new_row = row.copy()  # Create a copy of the original row
                new_row["phrase"] = phrase
                new_row["category"] = categories
                new_row["score"] = scores
                new_row["score_price"] = scores[categories.index('price')]
                new_rows.append(new_row)
        else:
            # If no phrases, add an empty row
            new_row = row.copy()
            new_row["phrase"] = None
            new_row["category"] = []
            new_row["score"] = []
            new_rows.append(new_row)

    # Creates a new DataFrame from the new rows
    row_df = pd.DataFrame(new_rows)
    row_df.drop(columns=column_name_phrase)
    if file_path:
        row_df.to_csv(file_path, index=False)

    return row_df
