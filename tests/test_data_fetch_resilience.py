"""scTenifold.data._get's GitHub API resilience: authenticate when a token
is available, and cache the repo-tree listing so fetch_data() for N
datasets doesn't make N identical requests against GitHub's 60/hour
unauthenticated rate limit.
"""
import pytest

from scTenifold.data import _get


@pytest.fixture(autouse=True)
def _clear_repo_tree_cache():
    # lru_cache persists across tests otherwise, hiding call-count bugs
    # (or leaking a monkeypatched response into an unrelated test).
    _get._fetch_repo_tree.cache_clear()
    yield
    _get._fetch_repo_tree.cache_clear()


def test_no_token_sends_no_auth_header(monkeypatch):
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GH_TOKEN", raising=False)
    assert _get._github_api_headers() == {}


def test_github_token_env_var_sets_bearer_header(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "ghs_test123")
    monkeypatch.delenv("GH_TOKEN", raising=False)
    assert _get._github_api_headers() == {"Authorization": "Bearer ghs_test123"}


def test_gh_token_env_var_also_works(monkeypatch):
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.setenv("GH_TOKEN", "gh_test456")
    assert _get._github_api_headers() == {"Authorization": "Bearer gh_test456"}


class _FakeResponse:
    def __init__(self, tree):
        self._tree = tree

    def raise_for_status(self):
        pass

    def json(self):
        return {"tree": self._tree}


def test_repo_tree_is_fetched_once_across_repeated_calls(monkeypatch):
    tree = [
        {"path": "AD", "type": "tree"},
        {"path": "AD/sample1", "type": "tree"},
        {"path": "AD/sample1/matrix.mtx", "type": "blob"},
    ]
    calls = []

    def fake_get(url, headers=None):
        calls.append((url, headers))
        return _FakeResponse(tree)

    monkeypatch.setattr(_get.requests, "get", fake_get)

    # list_data() is what fetch_data() calls once per dataset; five calls
    # for five datasets (test_data.py's shape) must not mean five requests.
    for _ in range(5):
        result = _get.list_data(owner="qwerty239qwe", return_list=False)
    assert len(calls) == 1, f"expected 1 network call, got {len(calls)}"
    assert result == {"AD": {"AD/sample1": ["AD/sample1/matrix.mtx"]}}


def test_different_owners_are_cached_independently(monkeypatch):
    calls = []

    def fake_get(url, headers=None):
        calls.append(url)
        return _FakeResponse([{"path": "X", "type": "tree"}])

    monkeypatch.setattr(_get.requests, "get", fake_get)

    _get.list_data(owner="owner-a")
    _get.list_data(owner="owner-a")
    _get.list_data(owner="owner-b")
    assert len(calls) == 2  # one per distinct owner, not per call
