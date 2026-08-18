import pytest
import requests

from scTenifold.data import fetch_data


def _fetch_or_skip(ds_name):
    """fetch_data(), but skip (not fail) on GitHub's API rate limit.

    These tests depend on a live, third-party service outside this repo's
    control (60 requests/hour unauthenticated, shared across whoever else
    is on the same CI runner IP); a transient rate limit is a statement
    about GitHub's quota, not about the code under test, so it shouldn't
    read as a red build. Anything else -- a real fetch/parsing bug --
    still fails normally.
    """
    try:
        return fetch_data(ds_name)
    except requests.exceptions.HTTPError as exc:
        if exc.response is not None and exc.response.status_code == 403 and "rate limit" in str(exc).lower():
            pytest.skip(f"GitHub API rate limit exceeded fetching {ds_name!r} test data")
        raise


def test_featch_morphine_datasets():
    morphine = _fetch_or_skip("morphine")


def test_featch_AD_datasets():
    AD = _fetch_or_skip("AD")


def test_featch_Nkx2_KO_datasets():
    Nkx2_KO = _fetch_or_skip("Nkx2_KO")


def test_featch_aging_datasets():
    aging = _fetch_or_skip("aging")


def test_featch_dsRNA_datasets():
    dsRNA = _fetch_or_skip("dsRNA")


# def test_featch_cetuximab_datasets():
#     cetuximab = fetch_data("cetuximab")
