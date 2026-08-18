import os
from functools import lru_cache
from typing import Dict, List, Union
import zipfile
from io import BytesIO
import re
from pathlib import Path

import requests
import pandas as pd


from ._io import read_mtx


_valid_ds_names = ["AD", "Nkx2_KO", "aging", "cetuximab", "dsRNA", "morphine"]
_repo_url = "https://raw.githubusercontent.com/{owner}/scTenifold-data/main/{ds_name}"
_repo_tree_url = "https://api.github.com/repos/{owner}/scTenifold-data/git/trees/main?recursive=1"


__all__ = ["list_data", "fetch_data"]


def _github_api_headers() -> Dict[str, str]:
    """Auth header for api.github.com, if a token is available.

    Unauthenticated requests to the GitHub REST API are capped at 60/hour
    per source IP -- easy to exhaust in CI, where many runs can share an
    IP. An authenticated request (e.g. the ambient ``GITHUB_TOKEN`` every
    GitHub Actions run gets for free) raises that to 5000/hour. Falls back
    to unauthenticated if no token is set, e.g. for a plain local install.
    """
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    return {"Authorization": f"Bearer {token}"} if token else {}


def fetch_and_extract(url: str, saved_path: Union[str, Path]) -> None:
    """Download a zip archive and extract it to ``saved_path``."""
    resp = requests.get(url, stream=True)
    resp.raise_for_status()
    content = resp.content
    zf = zipfile.ZipFile(BytesIO(content))
    with zf as f:
        f.extractall(saved_path)


def download_url(url: str, save_path: Union[str, Path], chunk_size: int = 128) -> None:
    """Stream ``url`` to disk at ``save_path``."""
    r = requests.get(url, stream=True)
    r.raise_for_status()
    with open(save_path, 'wb') as fd:
        for chunk in r.iter_content(chunk_size=chunk_size):
            fd.write(chunk)


@lru_cache(maxsize=None)
def _fetch_repo_tree(owner: str) -> tuple:
    """The actual network call behind list_data(), cached per owner.

    The repo tree this reads almost never changes within a process's
    lifetime, and fetch_data() calls list_data() once per dataset --
    without this, five fetch_data() calls in a row (e.g. one per test in
    tests/test_data.py) make five identical requests against a 60/hour
    unauthenticated rate limit for no benefit. Returns a tuple (not a
    list) so lru_cache can't hand callers a shared mutable object.
    """
    response = requests.get(_repo_tree_url.format(owner=owner), headers=_github_api_headers())
    response.raise_for_status()
    return tuple(response.json()["tree"])


def list_data(owner: str = "qwerty239qwe",
              return_list: bool = True) -> Union[Dict[str, Dict[str, List[str]]], List[str]]:
    """

    Parameters
    ----------
    owner: str, default = 'qwerty239qwe'
        owner name of dataset repo
    return_list: bool, default = True
        To return list of data name or return a dict indicating repo structure
    Returns
    -------
    data_info_tree: list or dict
        The obtainable data store in a dict, structure {'data_name': {'group': ['file_names']}}
        or in a list of data_names
    """
    tree = _fetch_repo_tree(owner)
    ds_list = [p["path"] for p in tree if "/" not in p["path"] and p["type"] == "tree"]
    if return_list:
        return ds_list

    s_pattern = re.compile(r"/")
    lv1, lv2 = {}, []
    for t in tree:
        if len(re.findall(s_pattern, t['path'])) == 1:
            lv1[t["path"]] = []
        elif len(re.findall(s_pattern, t['path'])) == 2:
            lv2.append(t["path"])
    for b in lv2:
        lv1[re.findall(r"(.*)/", b)[0]].append(b)

    ds_dic = {ds: {} for ds in ds_list}
    for k, v in lv1.items():
        ds_dic[re.findall(r"(.*)/", k)[0]][k] = v
    return ds_dic


def fetch_data(ds_name: str,
               dataset_path: Path = Path(__file__).parent.parent.parent / Path("datasets"),
               owner: str = "qwerty239qwe") -> Dict[str, pd.DataFrame]:
    """Fetch and load a remote scTenifold dataset by name.

    Parameters
    ----------
    ds_name
        Dataset name (one of :data:`_valid_ds_names`).
    dataset_path
        Local directory to cache downloads.
    owner
        GitHub owner of the ``scTenifold-data`` mirror.

    Returns
    -------
    Mapping from sample-group name to a genes-by-cells DataFrame.
    """
    if not dataset_path.is_dir():
        dataset_path.mkdir(parents=True)
    if ds_name not in _valid_ds_names:
        raise ValueError(f"Unknown dataset {ds_name!r}; expected one of {_valid_ds_names}")
    ds_dic = list_data(owner=owner, return_list=False)
    if ds_name not in ds_dic:
        raise ValueError(f"Dataset {ds_name!r} was not found in the remote data repository")

    result_df = {}

    for lv_1, files in ds_dic[ds_name].items():
        fn_names = {k: None for k in ["matrix", "genes", "barcodes"]}
        for f in files:
            if not (dataset_path / Path(lv_1)).is_dir():
                (dataset_path / Path(lv_1)).mkdir(parents=True, exist_ok=True)
            for fn_name in fn_names:
                if fn_name in f:
                    fn_names[fn_name] = f
            if not (dataset_path / Path(f)).exists():
                download_url(url=_repo_url.format(owner=owner, ds_name=f), save_path=(dataset_path / Path(f)))
        result_df[re.findall(r".*/(.*)", lv_1)[0]] = read_mtx(mtx_file_name=str((dataset_path / Path(fn_names["matrix"]))),
                                                              gene_file_name=str((dataset_path / Path(fn_names["genes"]))),
                                                              barcode_file_name=str((dataset_path / Path(fn_names["barcodes"])))
                                                              if fn_names["barcodes"] is not None else None) # optional
    return result_df
