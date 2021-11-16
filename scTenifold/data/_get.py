from typing import Dict, Union, List
import zipfile
import gzip
from io import BytesIO
import re
from pathlib import Path

import requests
import pandas as pd


from ._io import read_mtx


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


def download_url(url, save_path, chunk_size=128):
    r = requests.get(url, stream=True)
    with open(save_path, 'wb') as fd:
        for chunk in r.iter_content(chunk_size=chunk_size):
            fd.write(chunk)


def list_data(owner="qwerty239qwe", return_list=True) -> Union[dict, List[str]]:
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
    tree = requests.get(_repo_tree_url.format(owner=owner)).json()['tree']
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
               owner="qwerty239qwe") -> Dict[str, pd.DataFrame]:
    if not dataset_path.is_dir():
        dataset_path.mkdir(parents=True)
    ds_dic = list_data(owner=owner, return_list=False)

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