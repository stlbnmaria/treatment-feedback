import pandas as pd
from rake_nltk import Rake

def phrase_extraction(df: pd.DataFrame,
                       column_name_comment: str = "comment"):
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
    # setting min length to 4 to extract small phrases
    r = Rake(min_length=4, max_length=10)

    # function for extracting the phrases 
    def extract_keyphrase(text):
        """extracts key phrases from a text input (in our case the comments)

        Args:
            text (str): the text that from which phrases should be extracted

        Returns:
            r: the key phrases from the text
        """
        r.extract_keywords_from_text(text)
        return r.get_ranked_phrases()
    
    # applies phrase extraction to chosen comment and creates new column "phrases"
    df["phrases"] = df[column_name_comment].apply(extract_keyphrase)

    df.to_csv("data_preprocessing/data/preprocessed.csv")
    return df