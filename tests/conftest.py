import pytest
from pathlib import Path
from scTenifold.data import read_mtx


@pytest.fixture(scope="session")
def control_data():
    mtx_path = Path("../data/morphineNeurons/mock/control.mtx")
    gene_path = Path("../data/morphineNeurons/mock/controlGenes.tsv")
    barcodes_path = Path("../data/morphineNeurons/mock/controlBarcodes.tsv")
    control_data = read_mtx(mtx_path, gene_path, barcodes_path)
    return control_data


@pytest.fixture(scope="session")
def treated_data():
    mtx_path = Path("../data/morphineNeurons/morphine/morphine.mtx")
    gene_path = Path("../data/morphineNeurons/morphine/morphineGenes.tsv")
    barcodes_path = Path("../data/morphineNeurons/morphine/morphineBarcodes.tsv")
    treated_data = read_mtx(mtx_path, gene_path, barcodes_path)
    return treated_data

