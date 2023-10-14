import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from pathlib import Path
import pandas as pd
import string
import yaml


# Download necessary resources for nltk
nltk.download("stopwords")
nltk.download("punkt")
nltk.download("wordnet")


def extract_filter_process(
    file_path: Path,
    diseases: list = [],
    antibodies: list = [],
    treatments: list = [],
) -> pd.DataFrame:
    """
    Preprocesses the CSV file specified by extracting treatment, disease, and antibody information
    and tokenizing the comments.

    Parameters:
    - file_path (Path): Path to the CSV file.
    - diseases (list): List of diseases to filter. If empty, no filtering is applied.
    - antibodies (list): List of antibodies to filter. If empty, no filtering is applied.
    - treatments (list): List of treatments to filter. If empty, no filtering is applied.

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

    return dataframe


def preprocess_data(config_data: Path = Path("config.yaml")):
    """
    Initiate preprocessing of data and save the output if specified.

    Parameters:
    - file_path (Path): Path to config file.

    Returns:
    - pd.DataFrame: The processed dataframe with added 'treatment', 'disease', 'antibody',
                    and 'processed_comment' columns.
    """

    # read the path from the config.yaml file
    with open(config_data) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    df = extract_filter_process(
        file_path=Path(config_data.parent / config["file_path"]),
        diseases=config.get("diseases", []),
        antibodies=config.get("antibodies", []),
        treatments=config.get("treatments", []),
    )
    print("-------- Data processing done -------")

    # set the output path of the csv
    output_path = config.get("output_path", None)
    if output_path:
        # save the data to csv if requested
        df.to_csv(Path(config_data.parent / output_path), index=False)

    return df


if __name__ == "__main__":
    preprocess_data()
