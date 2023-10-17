import ast
import numpy as np
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from transformers import pipeline
from typing import List, Tuple, Union


def load_process_sent_data(path: str) -> pd.DataFrame:
    """
    Load and processes the data from Topic extraction.

    Parameters:
    - path(str): Path to the data which is loaded.
    Returns:
    - pd.DataFrame: Dataframe contained transformed data.
    """

    df = pd.read_csv(path)
    df["category"] = df["category"].apply(
        lambda x: ast.literal_eval(x) if pd.notnull(x) else []
    )
    df["topic"] = df["category"].apply(lambda x: x[0] if len(x) > 0 else "")
    df = df.drop(["category", "score"], axis=1)
    df = df[df["phrase"].notna()]
    # df["phrase"] = df["phrase"].apply(lambda x : str(x))
    return df


def sentiment_analysis_transformers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform sentiment analysis using a pretrained CSV model.

    Parameters:
    - df(pd.Dataframe): Dataframe upon wihch sentiment analysis will be conducted.
    Returns:
    - pd.DataFrame: DataFrame containing original data with added transformer sentiment labels and scores.
    """

    # Ensure the 'comment' column exists
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


def main():
    # Define the path to the preprocessed data
    path = os.path.join(
        os.getcwd(), "..", "data_preprocessing", "data", "csv_for_sentiment.csv"
    )
    output_path = os.path.join(
        os.getcwd(),
        "..",
        "data_preprocessing",
        "data",
        "sent_analysis.csv",
    )

    # Preprocess data
    df = load_process_sent_data(path)

    #   Use pre-trained model to predict sentiment
    df = sentiment_analysis_transformers(df)

    # df_with_predictions.to_csv(output_path)
    df.to_csv(output_path)

    print("------- Sentiment Analysis Completed -------")


if __name__ == "__main__":
    main()
