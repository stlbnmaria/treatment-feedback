import os
import pandas as pd
from pathlib import Path
import spacy
import yaml


def rank_keywords_inside_topic(
    df: pd.DataFrame, topic: str, config_data
) -> pd.DataFrame:
    """
    Retrieve and calculate the similarity of important keywords related to a specific topic within a dataset of diseases.

    Parameters:
    - df (pandas.DataFrame): A DataFrame containing information about diseases, including a "disease" column and a "keywords_comment" column.
    - topic (str): The specific topic for which you want to find related keywords.

    The function performs the following steps:
    1. Filters the data to select the disease.
    2. Retrieves all unique keywords related to the selected disease.
    3. Calculates the similarity score between each keyword and the given topic using spaCy word vectors.
    4. Sorts the similarity scores in descending order.
    5. Saves the sorted similarity scores to a CSV file named "{disease}_{topic}.csv".

    Returns:
    - pandas.DataFrame: A DataFrame containing keywords with sorted similarity scores.
    """
    with open(config_data) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    nlp = spacy.load("en_core_web_md")

    similarity_scores_path = config.get("similarity_scores_path", "")

    for disease in df["disease"].unique():
        disease_data = df[df["disease"] == disease]

        # Get all unique keywords for the disease
        disease_keywords = disease_data["keywords_comment"]

        all_keywords = []
        for keyword in disease_keywords:
            all_keywords.extend(keyword)

        all_keywords = list(set(all_keywords))

        # Calculate similarity score between key words and topic
        similarity_scores = {}

        for keyword in all_keywords:
            keyword_vec = nlp(keyword)
            topic_vec = nlp(topic)
            similarity_score = keyword_vec.similarity(topic_vec)
            similarity_scores[keyword] = similarity_score

        # Sort the similarity scores in descending order
        sorted_similarity_scores = sorted(
            similarity_scores.items(), key=lambda x: x[1], reverse=True
        )
        similarity_scores = pd.DataFrame(
            sorted_similarity_scores, columns=["Keyword", "Similarity Score"]
        )

        print(f"-- Key words for {topic} for {disease} --")

        # Save similarity scores as the CSV file
        csv_file_name = f"scores_{disease}_{topic}.csv"
        os.makedirs(Path(config_data.parent / similarity_scores_path), exist_ok=True)
        similarity_scores.to_csv(
            Path(config_data.parent / similarity_scores_path / csv_file_name),
            index=False,
        )


def create_keywords_ranking_for_topics(
    df: pd.DataFrame, config_data: Path = Path("config.yaml")
):
    """
    Get lists of key words for selected disease and selcted topics ordered by similarity score with topic.

    Parameters:
    - df (pandas.DataFrame): A DataFrame containing information about diseases, including a "disease" column and a "keywords_comment" column.
    - topic (str): The specific topic for which you want to find related keywords.

    The function performs the following steps:
    1. Filters the data to select the disease.
    2. Retrieves all unique keywords related to the selected disease.
    3. Calculates the similarity score between each keyword and the given topic using spaCy word vectors.
    4. Sorts the similarity scores in descending order.
    5. Saves the sorted similarity scores to a CSV file named "{disease}_{topic}.csv".

    Returns:
    - pandas.DataFrame: A DataFrame containing keywords with sorted similarity scores.
    """
    # Read the path from the config.yaml file
    with open(config_data) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    topics = config.get("topics", [])

    for topic in topics:
        rank_keywords_inside_topic(df, topic, config_data)

    print("----------- Key words from all topics done ---------")
