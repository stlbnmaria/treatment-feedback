import pandas as pd
from pathlib import Path
import yaml

from keywords_extraction import extract_keywords_from_comments


def wordcloud(config_path: Path = Path("config.yaml")) -> None:
    """
    Reads data from a CSV file and extracts keywords from comments based on the configuration
    specified in a YAML file. Generates a CSV file needed to display a wordcloud in Tableau.

    Args:
        config_path (Path): Path to config file.

    Returns:
        None
    """
    # Read the path from the config.yaml file
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Read the preprocessed data and extract keywords from it
    df = pd.read_csv(Path(config_path.parent / config["preprocessing_path"]))
    dataframe = extract_keywords_from_comments(df)

    # Transform each element of a list-like to a row, replicating index values
    dataframe = dataframe.loc[:, ["text_index", "keywords_comment"]]
    expanded_df = dataframe.explode("keywords_comment", ignore_index=True)
    expanded_df = expanded_df.rename(columns={"keywords_comment": "word"})

    # Set the output path of the csv
    output_path = Path(config_path.parent / config["wordcloud_path"])

    # Save the data to csv
    expanded_df.to_csv(output_path, index=False)


if __name__ == "__main__":
    wordcloud()
