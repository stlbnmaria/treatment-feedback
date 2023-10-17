import pandas as pd
from transformers import pipeline

def phrase_classification(df: pd.DataFrame):

    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

    category_labels = [
    "cost",
    "side effects",
    "improved",
    ]

    # Create a list to store the new rows
    new_rows = []

    # Iterate through the original DataFrame
    for index, row in df.iterrows():
        phrases = row["phrases"]
        
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
    return row_df

    


