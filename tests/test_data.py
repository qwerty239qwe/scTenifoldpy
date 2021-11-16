from scTenifold.data import fetch_data


def test_featch_morphine_datasets():
    morphine = fetch_data("morphine")


def test_featch_AD_datasets():
    AD = fetch_data("AD")


def test_featch_Nkx2_KO_datasets():
    Nkx2_KO = fetch_data("Nkx2_KO")


def test_featch_aging_datasets():
    aging = fetch_data("aging")


def test_featch_dsRNA_datasets():
    dsRNA = fetch_data("dsRNA")


def test_featch_cetuximab_datasets():
    cetuximab = fetch_data("cetuximab")