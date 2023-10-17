import pandas as pd
from rake_nltk import Rake
from typing import List


def extract_keyphrase(text: str, r: Rake) -> List[str]:
    """extracts key phrases from a text input (in our case the comments)

    Args:
        text (str): the text that from which phrases should be extracted
        r (Rank): Rake object to extract phrases

    Returns:
        List of the key phrases from the text
    """
    r.extract_keywords_from_text(text)
    return r.get_ranked_phrases()


def phrase_extraction(
    df: pd.DataFrame, column_name_comment: str = "comment"
) -> pd.DataFrame:
    """takes a dataframe as input and extracts the key phrases
       of a column and saves these as a new column

    Args:
        df (pd.Dataframe): the Dataframe from which to extract the keywords
        column_name_comment (str): the column which has the untreated comments

    Returns:
         pd.DataFrame: Returns the input dataframe with two added columns
                       with the key words from the processed and unproccessed columns
    """
    # setting min length to 4 to extract small phrases
    r = Rake(min_length=4, max_length=10)

    # applies phrase extraction to chosen comment and creates new column "phrases"
    df["phrases"] = df[column_name_comment].apply(lambda x: extract_keyphrase(x, r))

    return df
