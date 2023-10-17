import ast
import numpy as np
import os
import pandas as pd
from transformers import pipeline


def topic_condition(row: pd.Series) -> str:
    """

    Selects which topic to be the topic of the phrase.

    Parameters:
    - row(pd.Series): Row of the dataframe for which we are defining the topics.
    Returns:
    - str: string of the topic selected.
    """
    if len(row["category"]) > 0 and len(row["score"]) > 0:
        if row["score_price"] > 0.2:
            return "price"
        if row["score"][0] > 0.4:
            return row["category"][0]
        else:
            return "no topic"


def process_sent_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Load and processes the data from Topic extraction.

    Parameters:
    - path(str): Path to the data which is loaded.
    Returns:
    - pd.DataFrame: Dataframe contained transformed data.
    """

    # df = pd.read_csv(path)
    df["category"] = df["category"].apply(
        lambda x: ast.literal_eval(x) if pd.notnull(x) else []
    )
    df["score"] = df["score"].apply(
        lambda x: ast.literal_eval(x) if pd.notnull(x) else []
    )
    df["topic"] = df.apply(topic_condition, axis=1)
    df = df.drop(["category", "score"], axis=1)
    df = df[df["phrase"].notna()]
    return df


def sentiment_analysis_transformers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform sentiment analysis using a pretrained CSV model.

    Parameters:
    - df(pd.Dataframe): Dataframe upon wihch sentiment analysis will be conducted.
    Returns:
    - pd.DataFrame: DataFrame containing original data with added transformer sentiment labels.
    """

    # Ensure the 'phrase' column exists
    if "phrase" not in df.columns:
        raise ValueError("The CSV file must contain a 'phrase' column.")

    phrases = df["phrase"].tolist()

    # Load the classification pipeline
    classifier = pipeline("sentiment-analysis")

    # Classify the comments
    results = classifier(phrases)

    # Extract the labels and scores from the results
    df["transformer_sentiment_labels"] = [entry["label"] for entry in results]
    df["transformer_sentiment_labels"] = np.where(
        df["transformer_sentiment_labels"] == "NEGATIVE", 0, 1
    )
    return df


def sent_analysis(df, out_path):
    df = process_sent_data(df)
    df = sentiment_analysis_transformers(df)
    df.to_csv(out_path, index=False)
    print("------- Sentiment Analysis Completed -------")
