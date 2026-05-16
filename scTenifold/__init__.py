from .core import *


__all__ = ['scTenifoldNet', 'scTenifoldKnk',
           "sc_QC", "make_networks", "manifold_alignment", "d_regulation",
           "compare_networks", "virtual_knockout",
           "scTenifoldXct", "merge_scTenifoldXct",
           "set_seed", "get_Xct_pairs", "plot_XNet"]


__version__ = "0.3.0"


_XCT_EXPORTS = {"scTenifoldXct", "merge_scTenifoldXct",
                "set_seed", "get_Xct_pairs", "plot_XNet"}


def __getattr__(name):
    if name in _XCT_EXPORTS:
        try:
            from . import xct
        except ImportError as exc:
            raise ImportError(
                "scTenifoldXct requires optional dependencies. "
                "Install them with: pip install scTenifoldpy[xct]"
            ) from exc
        return getattr(xct, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
