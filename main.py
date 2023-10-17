import click
from pathlib import Path
import yaml

from data_preprocessing.data_preprocess import preprocess_data
from keywords_extraction.keywords_extraction import extract_keywords_from_comments
from markers_extraction.rank_keywords_inside_topic import (
    create_keywords_ranking_for_topics,
)
from phrase_modeling.phrase_classification import phrase_classification
from phrase_modeling.phrase_extraction import phrase_extraction
from sentiment_analysis.sentiment_analysis import sent_analysis
from markers_extraction.markers_in_comments import markers_in_comments


@click.command()
@click.option(
    "--config_path",
    default=Path("config.yaml"),
    type=click.Path(exists=True),
    help="Path to the config",
)
def main(config_path: Path):
    # perform data preprocessing
    df = preprocess_data(config_path)

    # extract keywords
    df_keywords = extract_keywords_from_comments(df, config_path)
    create_keywords_ranking_for_topics(df_keywords, config_path)

    # read the path from the config.yaml file
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    # do phase extraction
    df_phrase = phrase_extraction(
        df, min_length=config["min_length"], max_length=config["max_length"]
    )
    # do phrase classification
    df_phrase = phrase_classification(
        df_phrase,
        file_path=Path(config_path.parent / config["phrase_path"]),
        category_labels=config["topics"],
    )
    # do sentiment analysis on phrases
    df_phrase = sent_analysis(
        df_phrase, out_path=Path(config_path.parent / config["sent_phrase_path"])
    )
    markers_in_comments(config_path)


if __name__ == "__main__":
    main()
