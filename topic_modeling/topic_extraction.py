from data_preprocessing.data_preprocess import lemmatize_case
import pandas as pd
import spacy
from nltk.stem import PorterStemmer


def get_important_key_words_for_topic(data, topic):
    """
    Retrieve and calculate the similarity of important keywords related to a specific topic within a dataset of diseases.

    Parameters:
    - data (pandas.DataFrame): A DataFrame containing information about diseases, including a "disease" column and a "keywords_comment" column.
    - topic (str): The specific topic for which you want to find related keywords.

    The function performs the following steps:
    1. Filters the data to select the disease.
    2. Retrieves all unique keywords related to the selected disease.
    3. Calculates the similarity score between each keyword and the given topic using spaCy word vectors.
    4. Sorts the similarity scores in descending order.
    5. Saves the sorted similarity scores to a CSV file named "{disease}_{topic}.csv".

    Returns:
    - None
    """
    for disease in data["disease"].unique():
        disease_data = data[data["disease"] == "Crohn's Disease"]

        # Get all unique keywords for the disease
        disease_keywords = disease_data["keywords_comment"]
        for keyword in disease_keywords:
            disease_keywords.add(keyword)

        # Calculate similarity score between key words and topic
        similarity_scores = {}

        nlp = spacy.load("en_core_web_md")

        for keyword in disease_keywords:
            keyword_vec = nlp(keyword)
            target_vec = nlp(topic)
            similarity_score = keyword_vec.similarity(target_vec)
            similarity_scores[keyword] = similarity_score

        # Sort the similarity scores in descending order
        sorted_similarity_scores = sorted(
            similarity_scores.items(), key=lambda x: x[1], reverse=True
        )
        df = pd.DataFrame(
            sorted_similarity_scores, columns=["Keyword", "Similarity Score"]
        )

        # Save similarity scores as the CSV file
        csv_file_path = f"scores_{disease}_{topic}.csv"
        df.to_csv(csv_file_path, index=False)


def stem_tokens(tokens):
    stemmer = PorterStemmer()
    return " ".join(stemmer.stem(token) for token in tokens)


def find_markers_in_comments(df, markers, disease, topic):
    """
    Find markers in comments within a dataset and save the resulting DataFrame as a CSV file.

    Parameters:
    - data (pandas.DataFrame): A DataFrame containing comments and other data.
    - markers (list): A list of markers or keywords to search for within the comments.
    - disease (str): The name of the disease under consideration.
    - topic (str): The specific topic or context for marker identification.

    Returns:
    - pandas.DataFrame: The modified DataFrame with marker identification columns.
    """
    lemmatized_markers = [lemmatize_case(phrase) for phrase in markers]
    processed_markers = [stem_tokens(tokens) for tokens in lemmatized_markers]

    # Create a dictionary with original and steam markers
    marker_dictionary = {}

    for i in range(len(markers)):
        marker_dictionary[markers[i]] = {"processed": processed_markers[i]}

    df["comment_stem"] = df["processed_comment"].apply(lambda x: x.split())
    df["stem_comment"] = df["processed_comment"].apply(stem_tokens)

    # Create the columns for markers with value 1 if marker is in the comment and 0 otherwise
    for column in markers:
        df[column] = df["comment_stem"].apply(
            lambda x: 1
            if marker_dictionary.get(column, {}).get("processed", None) in x
            else 0
        )

    df = df.loc[:, processed_markers + ["text_index"]]

    # Change layout of the table
    df_long = pd.melt(df, id_vars="text_index", value_vars=df.columns[1:])
    df_long = df_long[df_long["value"] == 1]
    df_long.drop(columns="value", inplace=True)
    df_long.columns = ["text_index", "side effects"]

    # Save file as the CSV file
    csv_file_path = f"markers_{disease}_{topic}.csv"
    df_long.to_csv(csv_file_path, index=False)
    return df_long
