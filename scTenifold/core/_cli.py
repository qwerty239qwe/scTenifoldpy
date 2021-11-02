import click
from pathlib import Path
import yaml
from scTenifold.data import read_mtx
from scTenifold.core._base import scTenifoldNet


@click.command()
@click.argument("-c", "--config", "config_file_path")
def main(config_file_path):
    with open(Path(config_file_path), "r") as f:
        data = yaml.safe_load(f)
    X_name, Y_name = data["X"]["name"], data["Y"]["name"]
    dataset = {X_name: read_mtx(data["X"]["mtx_path"],
                                data["X"]["gene_path"],
                                data["X"]["barcodes_path"]),
               Y_name: read_mtx(data["Y"]["mtx_path"],
                                data["Y"]["gene_path"],
                                data["Y"]["barcodes_path"])}

    sc = scTenifoldNet(dataset[X_name], dataset[Y_name], X_name, Y_name)


if __name__ == '__main__':
    main()
