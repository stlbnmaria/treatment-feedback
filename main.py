import click
from pathlib import Path

from data_preprocessing.data_preprocess import preprocess_data
from keywords_extraction.keywords_extraction import extract_keywords_from_comments
from markers_extraction.rank_keywords_inside_topic import (
    create_keywords_ranking_for_topics,
)


@click.command()
@click.option(
    "--config_data",
    default=Path("config.yaml"),
    type=click.Path(exists=True),
    help="Path to the data config",
)
def main(config_data: Path):
    df = preprocess_data(config_data)
    df_keywords = extract_keywords_from_comments(df, config_data)

    create_keywords_ranking_for_topics(df_keywords, config_data)


if __name__ == "__main__":
    main()
