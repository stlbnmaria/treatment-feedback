from ast import literal_eval
import pandas as pd
from rake_nltk import Rake

def keyword_extraction(df: pd.DataFrame,
                       column_name_comment: str = "comment",
                       column_name_processed: str = "processed_comment"):
    """takes a dataframe as input and extracts the keywords
       of a column and saves these as a new column
    

    Args:
        df (pd.Dataframe): the Dataframe from which to extract the keywords
        column_name_comment (str): the column which has the untreated comments
        column_name_processed (str): the column which has the proccessed comments

    Returns:
         pd.DataFrame: Returns the input dataframe with two added columns
                       with the key words from the processed and unproccessed columns
    """
    r = Rake(max_length=2)

    def extract_keywords(text):
        r.extract_keywords_from_text(text)
        return r.get_ranked_phrases()
    
    df["keywords_comment"] = df[column_name_comment].apply(extract_keywords)

    def extract_keywords_from_list(word_list):
        text = ' '.join(word_list)
        r.extract_keywords_from_text(text)
        return r.get_ranked_phrases()

    df["keywords_processed_column"] = df[column_name_processed].apply(extract_keywords_from_list)

    df.to_csv("data_preprocessing/data/preprocessed.csv")
    return df