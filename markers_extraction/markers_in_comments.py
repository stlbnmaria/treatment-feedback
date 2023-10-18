import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize
from pathlib import Path
import string
import yaml


def stem_tokens(tokens):
    stemmer = PorterStemmer()
    return " ".join(stemmer.stem(token) for token in tokens)


def stem_tokens_markers(tokens):
    stemmer = PorterStemmer()
    return [stemmer.stem(token) for token in tokens]


def find_marker_in_comments(df: pd.DataFrame, keywords: list) -> pd.DataFrame:
    """
    Finds markers in the comments of a DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame containing 'processed_comment' column.
        keywords (list): List of keywords to search for.

    Returns:
        pd.DataFrame: A DataFrame with 1 in rows where any keyword is found, else 0.
    """
    stemmed_keywords = stem_tokens_markers(keywords)
    return df["processed_comment"].apply(
        lambda x: 1 if any(keyword in x for keyword in stemmed_keywords) else 0
    )


def search_markers_in_comments(df: pd.DataFrame, topics: list, disease: str):
    """
    Search for markers in the comments of a DataFrame for a specific disease, and saves result to csv.

    Args:
        df (pd.DataFrame): The DataFrame to search within.
        topics (list): A list of topics and corresponding markers.
        disease (str): The specific disease to filter by.

    Returns:
        pd.DataFrame: A DataFrame containing the found markers for the given disease.
    """
    df = df[df["disease"] == disease]
    result_df = pd.DataFrame()

    for topic, markers in topics:
        for marker, keywords in markers.items():
            df[marker] = find_marker_in_comments(df, keywords)

        df_topic = df[list(markers.keys()) + ["text_index"]]

        # Change layout of the table
        df_long = pd.melt(
            df_topic, id_vars="text_index", value_vars=df_topic.columns[1:]
        )
        df_long = df_long[df_long["value"] == 1]
        df_long.drop(columns="value", inplace=True)
        df_long.columns = ["text_index", "marker"]
        df_long["topic"] = topic

        # Concatenate datasets vertically
        if result_df.empty:
            result_df = df_long
        else:
            result_df = pd.concat([result_df, df_long], ignore_index=True)

    print(f"---{disease} markers created---")

    # Save file as the CSV file
    csv_file_path = f"markers_{disease}.csv"
    result_df.to_csv(csv_file_path, index=False)


def markers_in_comments(config_path: Path = Path("../config.yaml")):
    """
    The function reads a CSV file specified in the configuration, preprocesses comments, and searches for markers
    for Crohn's Disease and Ulcerative Colitis.

    Args:
        config_path (Path, optional): Path to the YAML configuration file. Default is "../config.yaml".
    """
    # Load the YAML configuration file
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Read the CSV file into a dataframe
    file_path = Path(config_path.parent / config["preprocessing_path"])
    df = pd.read_csv(file_path)

    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()

    # Convert to lowercase, tokenize, remove stopwords, and apply lemmatization
    df["processed_comment"] = df["comment"].apply(
        lambda text: [
            lemmatizer.lemmatize(token)
            for token in word_tokenize(
                text.lower()
                .strip()
                .translate(str.maketrans("", "", string.punctuation))
            )
            if token not in stop_words and token.isalpha()
        ]
    )

    df["processed_comment"] = df["processed_comment"].apply(stem_tokens)

    # Create file with markers for Crohn's Disease and Ulcerative Coliti
    search_markers_in_comments(df, config["chron_markers"].items(), "Crohn's Disease")
    search_markers_in_comments(df, config["uc_markers"].items(), "Ulcerative Colitis")
