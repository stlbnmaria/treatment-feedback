import pandas as pd
from pathlib import Path
from keywords_extraction import extract_keywords_from_comments

def wordcloud() -> None:

    """
    Reads data from a CSV file and extracts keywords from comments based on the configuration
    specified in a YAML file. Generates a CSV file needed to display a wordcloud in Tableau.

    Args:
        None

    Returns:
        None
    """

    df = pd.read_csv("../data/preprocessed.csv")
    dataframe = extract_keywords_from_comments(df, Path("../config.yaml"))

    dataframe = dataframe.loc[:, ['text_index', 'keywords_comment']]
    expanded_df = dataframe.explode('keywords_comment', ignore_index=True)
    expanded_df = expanded_df.rename(columns={'keywords_comment': 'word'})
    print(expanded_df)

    # Set the output path of the csv
    output_path = Path("../data/wordcloud.csv")

    # Save the data to csv
    expanded_df.to_csv(output_path, index=False)

wordcloud()