"""``scTenifold`` lazy re-export behaviour when the ``[xct]`` extra is absent.

Covers the ``scTenifold.__getattr__`` fallback in ``scTenifold/__init__.py``:
accessing a re-exported name must raise a clear ``ImportError`` with an
install hint, while unknown names still raise ``AttributeError``. This is
unreachable in the xct integration job (where scTenifoldXct is installed),
so it runs in the default torch-free matrix.
"""
import importlib.util

import pytest

import scTenifold

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("scTenifoldXct") is not None,
    reason="covers the xct-absent re-export branch; scTenifoldXct is installed",
)

_XCT_NAMES = ["scTenifoldXct", "merge_scTenifoldXct",
              "set_seed", "get_Xct_pairs", "plot_XNet"]


@pytest.mark.parametrize("name", _XCT_NAMES)
def test_getattr_raises_install_hint(name):
    """Accessing a re-exported name without the extra is a clear ImportError."""
    with pytest.raises(ImportError, match=r"pip install scTenifoldpy\[xct\]"):
        getattr(scTenifold, name)


def test_getattr_unknown_attribute():
    """Unknown attributes still raise AttributeError, not ImportError."""
    with pytest.raises(AttributeError, match="no attribute"):
        scTenifold.definitely_not_an_attribute
