import os
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import pandas as pd
import string
import yaml


# Download necessary resources for nltk
nltk.download("stopwords")
nltk.download("punkt")
nltk.download("wordnet")

# Read the path from the config.yaml file
with open(os.path.join(os.getcwd(), "data_preprocessing", "config.yaml")) as f:
    config = yaml.load(f, Loader=yaml.FullLoader)


def preprocess_data(
    file_path: str,
    diseases: list = [],
    antibodies: list = [],
    treatments: list = [],
    output_path: str = None,
) -> pd.DataFrame:
    """
    Preprocesses the CSV file specified by extracting treatment, disease, and antibody information
    and tokenizing the comments.

    Parameters:
    - file_path (str): Path to the CSV file.
    - diseases (list): List of diseases to filter. If empty, no filtering is applied.
    - antibodies (list): List of antibodies to filter. If empty, no filtering is applied.
    - treatments (list): List of treatments to filter. If empty, no filtering is applied.
     - output_path (str, optional): Path to save the processed dataframe as a CSV file.
                                   If not provided, the dataframe will not be saved.

    Returns:
    - pd.DataFrame: The processed dataframe with added 'treatment', 'disease', 'antibody',
                    and 'processed_comment' columns.
    """

    # Read the CSV file into a dataframe
    dataframe = pd.read_csv(file_path)

    # Extract treatment, disease, and antibody information
    dataframe["treatment"] = (
        dataframe["medication"].str.extract("^(.*?)(?:\s*\(.*\)|\s*for)")[0].str.strip()
    )
    dataframe["disease"] = (
        dataframe["medication"].str.extract("[fF]or (.*?)(?:,|$)")[0].str.strip()
    )
    dataframe["antibody"] = dataframe["medication"].str.extract("\(([^)]+)\)")

    # Filter the dataframe based on the specified diseases
    if diseases:
        dataframe = dataframe[dataframe["disease"].isin(diseases)]

    # Filter the dataframe based on the specified antibodies
    if antibodies:
        dataframe = dataframe[dataframe["antibody"].isin(antibodies)]

    # Filter the dataframe based on the specified treatments
    if treatments:
        dataframe = dataframe[dataframe["treatment"].isin(treatments)]

    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()

    # Convert to lowercase, tokenize, remove stopwords, and apply lemmatization
    dataframe["processed_comment"] = dataframe["comment"].apply(
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

    if output_path:
        dataframe.to_csv(output_path, index=False)

    return dataframe


def main():
    df = preprocess_data(
        config["file_path"],
        diseases=config.get("diseases", []),
        antibodies=config.get("antibodies", []),
        treatments=config.get("treatments", []),
        output_path=config.get("output_path", None),
    )
    print("-------- Data processing done -------")


if __name__ == "__main__":
    main()
