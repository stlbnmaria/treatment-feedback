import click
from pathlib import Path

from data_preprocessing.data_preprocess import preprocess_data
from keyword_extraction import keyword_extraction
from phrase_modeling.phrase_classification import phrase_classification
from phrase_modeling.phrase_extraction import phrase_extraction


@click.command()
@click.option(
    "--config_data",
    default=Path("data_preprocessing/config.yaml"),
    type=click.Path(exists=True),
    help="Path to the data preprocessing config",
)
def main(config_data: Path):
    df = preprocess_data(config_data)
    df = keyword_extraction(df)
    df = phrase_extraction(df)

    df_phrase = phrase_classification(df)


if __name__ == "__main__":
    main()
