from ast import literal_eval
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import pandas as pd
from pathlib import Path
from rake_nltk import Rake
import string
from typing import List
import yaml


def kewords_lemmatization(value: str) -> str:
    """
    Lemmatizes keywords.

    Parameters:
    - value (str): The input text string to be lemmatized.

    Returns:
    - str: A string of the lemmatized comment.
    """
    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()

    lemmatized_tokens = [
        lemmatizer.lemmatize(token)
        for token in word_tokenize(
            value.lower().strip().translate(str.maketrans("", "", string.punctuation))
        )
        if token not in stop_words and token.isalpha()
    ]
    return " ".join(
        lemmatized_tokens
    )  # Join the lemmatized tokens into a single string


def lemmatize_case(value: str) -> List[str]:
    """
    Lemmatizes a given text string by tokenizing it, converting to lowercase,
    removing punctuation, and lemmatizing non-stopword, alphabetical tokens.

    Parameters:
    - value (str): The input text string to be lemmatized.

    Returns:
    - list: A list of lemmatized tokens from the input text, excluding stop words and non-alphabetical tokens.
          If the input value is None or NaN, an empty list is returned.
    """

    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()

    if pd.notna(value):
        return [
            lemmatizer.lemmatize(token)
            for token in word_tokenize(
                value.lower()
                .strip()
                .translate(str.maketrans("", "", string.punctuation))
            )
            if token not in stop_words and token.isalpha()
        ]
    else:
        return []


def remove_disease_terms(row: pd.Series) -> List[str]:
    """
    Filter out words from a processed comment that appear in the names of the treatment,
    disease and anti-body for this disease.

    Parameters:
    - row (pandas.Series): A pandas Series containing columns.

    Returns:
    - list: A list of words from 'processed_comment' that are not found in any of the lemmatized sets
    ('lemmatized_disease', 'lemmatized_treatment', 'lemmatized_antibody').
    """

    return [
        keyword
        for keyword in row["keywords_comment"]
        if all(
            word
            not in [
                row["lemmatized_disease"],
                row["lemmatized_treatment"],
                row["lemmatized_antibody"],
                "uc",
            ]
            for word in keyword.split()
        )
    ]


def extract_keywords(text: str) -> List[str]:
    """
    Lemmatizes keywords.

    Parameters:
    - text (str): The input text string to extract keywords from.

    Returns:
    - List[str]: A list of extracted keywords for that string.
    """
    r = Rake(max_length=2)
    r.extract_keywords_from_text(text)
    return r.get_ranked_phrases()


def extract_keywords_from_comments(
    df: pd.DataFrame, config_data: Path("config.yaml")
) -> pd.DataFrame:
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

    # change column to list of strings instead of whole string
    df["keywords_comment"] = df.keywords_comment.apply(lambda x: literal_eval(str(x)))

    df["keywords_comment"] = df["keywords_comment"].apply(
        lambda keywords: [kewords_lemmatization(keyword) for keyword in keywords]
    )

    # Convert to lowercase, tokenize, remove stopwords, and apply lemmatization
    df["lemmatized_disease"] = df["disease"].apply(lemmatize_case)
    df["lemmatized_antibody"] = df["antibody"].apply(lemmatize_case)
    df["lemmatized_treatment"] = df["treatment"].apply(lemmatize_case)

    df["keywords_comment"] = df.apply(remove_disease_terms, axis=1)

    # Set the output path of the csv
    output_path = config.get("keywords_output_file_path", None)

    if output_path:
        # Save the data to csv if requested
        df.to_csv(Path(config_data.parent / output_path), index=False)

    return df
