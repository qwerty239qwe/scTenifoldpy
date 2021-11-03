import requests
import zipfile
import gzip
from io import BytesIO

import pandas as pd


_valid_ds_names = ["AD", "Nkx2_KO", "aging", "cetuximab", "dsRNA", "morphine"]
_repo_url = "https://raw.githubusercontent.com/{owner}/scTenifold-data/master/{ds_name}"
_repo_tree_url = "https://api.github.com/repos/{owner}/scTenifold-data/git/trees/main?recursive=1"


def fetch_and_extract(url, saved_path):
    resp = requests.get(url, stream=True)
    content = resp.content
    zf = zipfile.ZipFile(BytesIO(content))
    with zf as f:
        f.extractall(saved_path)


def fetch_data(ds_name, download=True, owner="qwerty239qwe") -> pd:
    requests.get(_repo_tree_url.format(owner=owner)).json()