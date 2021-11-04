from typing import Dict
import zipfile
import gzip
from io import BytesIO
import re
from pathlib import Path

import requests
import pandas as pd


_valid_ds_names = ["AD", "Nkx2_KO", "aging", "cetuximab", "dsRNA", "morphine"]
_repo_url = "https://raw.githubusercontent.com/{owner}/scTenifold-data/master/{ds_name}"
_repo_tree_url = "https://api.github.com/repos/{owner}/scTenifold-data/git/trees/main?recursive=1"


__all__ = ["list_data", "fetch_data"]


def fetch_and_extract(url, saved_path):
    resp = requests.get(url, stream=True)
    content = resp.content
    zf = zipfile.ZipFile(BytesIO(content))
    with zf as f:
        f.extractall(saved_path)


def list_data(owner="qwerty239qwe") -> dict:
    """

    Parameters
    ----------
    owner: owner name of dataset repo, default = 'qwerty239qwe'

    Returns
    -------
    data_info_tree: dict
        The obtainable data store in a dict, structure {'data_name': {'group': ['file_names']}}
    """
    tree = requests.get(_repo_tree_url.format(owner=owner)).json()['tree']
    ds_list = [p["path"] for p in tree if "/" not in p["path"]]
    s_pattern = re.compile(r"/")
    lv1, lv2 = [], []
    for t in tree:
        if len(re.findall(s_pattern, t)) == 1:
            lv1.append(t)
        elif len(re.findall(s_pattern, t)) == 2:
            lv2.append(t)
    result = {ds: tree[ds[ds.index(""):]] for ds in ds_list}
    return result


def fetch_data(ds_name: str,
               dataset_path: Path = Path(__file__).parent.parent.parent / Path("datasets"),
               download=True,
               owner="qwerty239qwe") -> Dict[str, pd.DataFrame]:
    requests.get(_repo_tree_url.format(owner=owner)).json()