import click
from pathlib import Path

from data_preprocessing.data_preprocess import preprocess_data


@click.command()
@click.option(
    "--config_data",
    default=Path("data_preprocessing/config.yaml"),
    type=click.Path(exists=True),
    help="Path to the data preprocessing config",
)
def main(config_data: Path):
    preprocess_data(config_data)


if __name__ == "__main__":
    main()
