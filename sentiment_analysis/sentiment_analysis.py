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


def preprocess_data(path: str, min_df: float, max_df: float) -> Tuple:
    """
    Loads and preprocesses the data by vectorizing the comments and assigning sentiments.

    Parameters:
    - path (str): The path to the CSV file.
    - min_df (float): Minimum frequency for words to be considered.
    - max_df (float): Maximum frequency for words to be considered.

    Returns:
    - Tuple: A tuple containing the vectorized comments, the true sentiments, and the original dataframe.
    """
    # Load data from CSV
    df = pd.read_csv(path)

    # Convert the string representation of lists back to actual lists
    df["processed_comment"] = df["processed_comment"].apply(
        lambda x: ast.literal_eval(x) if pd.notnull(x) else []
    )

    # Convert list of tokens into a single string for vectorization
    comments = df.processed_comment.map(lambda tokens: " ".join(tokens))

    # Assign sentiment based on rating
    df["true_sentiment"] = np.where(df["rate"] > 5, 1, 0)

    # Create a TF-IDF vectorizer and transform comments
    vectorizer = TfidfVectorizer(
        min_df=min_df, max_df=max_df, analyzer="word", ngram_range=(1, 2)
    )
    X = vectorizer.fit_transform(comments)
    y = df["true_sentiment"]

    return X, y, df


def train_and_evaluate(
    X: np.array, y: np.array, df: pd.DataFrame
) -> Tuple[str, pd.DataFrame]:
    """
    Trains a Logistic Regression model, predicts on the entire dataset, and evaluates its performance on the test set.

    Parameters:
    - X (np.array): The vectorized comments.
    - y (np.array): The sentiments.
    - df (pd.DataFrame): The original dataframe.

    Returns:
    - Tuple: The classification report of the model on the test set and the dataframe with logistic regression predictions for the entire dataset.
    """
    # Initialize the Logistic Regression classifier
    classifier = LogisticRegression(class_weight="balanced", random_state=2023)

    # Split the data into training and test sets
    X_train, _, y_train, _ = train_test_split(
        X, y, stratify=y, test_size=0.3, random_state=2023
    )

    # Train the classifier
    model = classifier.fit(X_train, y_train)

    # Predict sentiments for the entire dataset
    yhat = model.predict(X)
    df["predicted_logreg_sentiment"] = yhat

    # Evaluate the model's performance on the test set
    _, X_test, _, y_test = train_test_split(
        X, y, stratify=y, test_size=0.3, random_state=2023
    )
    y_test_pred = model.predict(X_test)
    report = classification_report(y_test, y_test_pred)

    return report, df


def sentiment_analysis_transformers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform sentiment analysis using a pretrained CSV model.

    Parameters:
    - file_path (str): The path to the CSV file.

    Returns:
    - pd.DataFrame: DataFrame containing original data with added transformer sentiment labels and scores.
    """

    # Ensure the 'comment' column exists
    if "comment" not in df.columns:
        raise ValueError("The CSV file must contain a 'comment' column.")

    comments = df["comment"].tolist()

    # Load the classification pipeline
    classifier = pipeline("sentiment-analysis")

    # Classify the comments
    results = classifier(comments)

    # Extract the labels and scores from the results
    df["transformer_sentiment_labels"] = [entry["label"] for entry in results]
    df["transformer_sentiment_scores"] = [entry["score"] for entry in results]

    return df


def main():
    # Define the path to the preprocessed data
    path = os.path.join(
        os.getcwd(), "..", "data_preprocessing", "data", "preprocessed.csv"
    )
    output_path = os.path.join(
        os.getcwd(),
        "..",
        "data_preprocessing",
        "data",
        "sent_analysis.csv",
    )

    # Preprocess data
    X, y, df = preprocess_data(path, min_df=0.01, max_df=0.99)

    # Train the model, predict, and evaluate
    report, df_with_predictions = train_and_evaluate(X, y, df)

    df_with_predictions = sentiment_analysis_transformers(df_with_predictions)

    df_with_predictions.to_csv(output_path)

    print("------- Sentiment Analysis Completed -------")


if __name__ == "__main__":
    main()
