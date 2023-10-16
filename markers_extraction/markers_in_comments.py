from data_preprocessing.data_preprocess import lemmatize_case
import pandas as pd
from nltk.stem import PorterStemmer


def stem_tokens(tokens):
    stemmer = PorterStemmer()
    return ' '.join(stemmer.stem(token) for token in tokens)


def find_markers_in_comments(df: pd.DataFrame, markers: list, disease: str, topic: str) -> pd.DataFrame:
    """
    Find markers in comments within a dataset and save the resulting DataFrame as a CSV file.

    Parameters:
    - df (pandas.DataFrame): A DataFrame containing comments and other data.
    - markers (list): A list of markers or keywords to search for within the comments.
    - disease (str): The name of the disease under consideration.
    - topic (str): The specific topic or context for marker identification.

    Returns:
    - pandas.DataFrame: The modified DataFrame with marker identification columns.
    """
    lemmatized_markers = [lemmatize_case(phrase) for phrase in markers]
    processed_markers = [stem_tokens(tokens) for tokens in lemmatized_markers]

    # Create a dictionary with original and steam markers
    marker_dictionary = {}

    for i in range(len(markers)):
        marker_dictionary[markers[i]] = {
            'processed': processed_markers[i]
        }

    df["comment_stem"] = df["processed_comment"].apply(lambda x: x.split())
    df["stem_comment"] = df["processed_comment"].apply(stem_tokens)

    # Create the columns for markers with value 1 if marker is in the comment and 0 otherwise
    for column in markers:
        df[column] = df['comment_stem'].apply(lambda x: 1 if marker_dictionary.get(column, {}).get('processed', None) in x else 0)

    df = df.loc[:, processed_markers + ["text_index"]]

    # Change layout of the table
    df_long = pd.melt(df, id_vars="text_index", value_vars=df.columns[1:])
    df_long = df_long[df_long["value"] == 1]
    df_long.drop(columns="value", inplace=True)
    df_long.columns = ["text_index", "side effects"]

    # Save file as the CSV file
    csv_file_path = f"markers_{disease}_{topic}.csv"
    df_long.to_csv(csv_file_path, index=False)
