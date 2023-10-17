import pandas as pd
from transformers import pipeline


def phrase_classification(
    df: pd.DataFrame, column_name_phrase: str = "phrases"
) -> pd.DataFrame:
    """classifies the extracted phrases into topics

    Args:
        df (pd.DataFrame): the input dataframe
        column_name_phrase (str): the row name of the dataframe
        that contains the phrases that should be classified

    Returns:
        pd.DataFrame: the output dataframe in which each phrase
        is represented by a row
    """

    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

    # the categories for the classification
    category_labels = [
        "cost",
        "side effects",
        "improved",
    ]

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
    row_df.drop(columns="phrases")
    row_df.to_csv("data_preprocessing/data/row_csv.csv", index=False)

    return row_df
