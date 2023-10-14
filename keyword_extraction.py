# import ast
# import json
import pandas as pd
from rake_nltk import Rake

def keyword_extraction(df: pd.Dataframe):
    """takes a dataframe as input and extracts the keywords
       of a column and saves these as a new column
    

    Args:
        df (pd.Dataframe): the Dataframe from which to extract the keywords

    Returns:
         pd.DataFrame: Returns the input dataframe with the added column
                       with the key words
    """
    r = Rake()

    def extract_keywords(text):
        r.extract_keywords_from_text(text)
        return r.get_ranked_phrases()
    
    df["keywords_comment"] = df["comment"].apply(extract_keywords)

    def extract_keywords_from_list(word_list):
        text = ' '.join(word_list)
        r.extract_keywords_from_text(text)
        return r.get_ranked_phrases()

    df["keywords_processed_column"] = df["processed_comment"].apply(extract_keywords_from_list)

    df.to_csv("data_preprocessing/data/preprocessed.csv")
    return df