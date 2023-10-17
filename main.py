import click
from pathlib import Path
import yaml

from data_preprocessing.data_preprocess import preprocess_data
from phrase_modeling.phrase_classification import phrase_classification
from phrase_modeling.phrase_extraction import phrase_extraction


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

    # read the path from the config.yaml file
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    # do phase extraction
    df = phrase_extraction(
        df, min_length=config["min_length"], max_length=config["max_length"]
    )
    # do phrase classification
    df_phrase = phrase_classification(
        df,
        file_path=Path(config_path.parent / config["phrase_path"]),
        category_labels=config["topics"],
    )


if __name__ == "__main__":
    main()
