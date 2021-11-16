import typer
from pathlib import Path
import yaml
from scTenifold.data import read_mtx
from scTenifold import scTenifoldNet, scTenifoldKnk

app = typer.Typer()

@app.command(name="config")
def get_config_file(
        config_type: int = typer.Option(1, "--type", "-t",
                                        help="Type, 1: scTenifoldNet, 2: scTenifoldKnk",
                                        min=1, max=2),
        file_path: str = typer.Option(
                        ".",
                        "--path",
                        "-p",
                        help="Path to generate empty config file")):
    config = scTenifoldNet.get_empty_config()


@app.command(name="net")
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