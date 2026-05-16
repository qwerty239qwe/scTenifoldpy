from .core import *


__all__ = ['scTenifoldNet', 'scTenifoldKnk',
           "sc_QC", "make_networks", "manifold_alignment", "d_regulation",
           "compare_networks", "virtual_knockout",
           "scTenifoldXct", "merge_scTenifoldXct",
           "set_seed", "get_Xct_pairs", "plot_XNet"]


__version__ = "0.3.0"


# scTenifoldXct is maintained and released separately
# (https://github.com/cailab-tamu/scTenifoldXct, PyPI: scTenifoldXct).
# We do not vendor it; these names are re-exported lazily from the
# installed package so `import scTenifold` stays torch-free.
_XCT_EXPORTS = {"scTenifoldXct", "merge_scTenifoldXct",
                "set_seed", "get_Xct_pairs", "plot_XNet"}


def __getattr__(name):
    if name in _XCT_EXPORTS:
        try:
            import scTenifoldXct as _ext
        except ImportError as exc:
            raise ImportError(
                "scTenifoldXct is not installed. "
                "Install it with: pip install scTenifoldpy[xct]"
            ) from exc
        return getattr(_ext, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
