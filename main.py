import click
from pathlib import Path

from data_preprocessing.data_preprocess import preprocess_data
from topic_modeling.keywords_extraction import extract_keywords_from_comments


@click.command()
@click.option(
    "--config_data",
    default=Path("data_preprocessing/config.yaml"),
    type=click.Path(exists=True),
    help="Path to the data preprocessing config",
)
def main(config_data: Path):
    df = preprocess_data(config_data)
    df = extract_keywords_from_comments(df, config_data)


if __name__ == "__main__":
    main()
