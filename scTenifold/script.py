from scTenifold.core.utils import get_test_df, read_mtx
from scTenifold.core.base import scTenifoldNet
from pathlib import Path


def main():
    mtx_path = Path("./data/morphineNeurons/morphine/morphine.mtx")
    gene_path = Path("./data/morphineNeurons/morphine/morphineGenes.tsv")
    barcodes_path = Path("./data/morphineNeurons/morphine/morphineBarcodes.tsv")
    treated_data = read_mtx(mtx_path, gene_path, barcodes_path)

    mtx_path = Path("./data/morphineNeurons/mock/control.mtx")
    gene_path = Path("./data/morphineNeurons/mock/controlGenes.tsv")
    barcodes_path = Path("./data/morphineNeurons/mock/controlBarcodes.tsv")
    control_data = read_mtx(mtx_path, gene_path, barcodes_path)