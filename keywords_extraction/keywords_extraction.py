from ast import literal_eval
import pandas as pd
from pathlib import Path
from rake_nltk import Rake
import yaml


def extract_keywords(text):
    r = Rake(max_length=2)
    r.extract_keywords_from_text(text)
    return r.get_ranked_phrases()


def extract_keywords_from_comments(df: pd.DataFrame, config_data: Path("config.yaml")) -> pd.DataFrame:
    
    """
    Takes a dataframe as input, extracts the keywords from the commemts and save the output if specified.

    Args:
    - df (pd.Dataframe): The Dataframe from which to extract the keywords
    - file_path (Path): Path to config file.

    Returns:
    - pandas.DataFrame: The input DataFrame with a new column containing keywords for each comment.
    """

    # Read the path from the config.yaml file
    with open(config_data) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    df["keywords_comment"] = df["comment"].apply(extract_keywords)

    # Change column to list of strings instead of whole string
    df["keywords_comment"] = df.processed_comment.apply(
        lambda x: literal_eval(str(x))
    )

    print("-------- Key words extraction done -------")

    # Set the output path of the csv
    output_path = config.get("keywords_output_file_path", None)

    if output_path:
        # Save the data to csv if requested
        df.to_csv(Path(config_data.parent / output_path), index=False)
    
    return df
