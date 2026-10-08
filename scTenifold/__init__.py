from .core import *


__all__ = ['scTenifoldNet', 'scTenifoldKnk',
           "sc_QC", "make_networks", "manifold_alignment", "d_regulation",
           "compare_networks", "virtual_knockout",
           "heat_kernel", "hk_manifold_alignment", "knockout_direction",
           "scTenifoldXct", "merge_scTenifoldXct",
           "set_seed", "get_Xct_pairs", "plot_XNet"]


# The version is set in pyproject.toml
try:
    from importlib.metadata import PackageNotFoundError, version as _version
    __version__ = _version("scTenifoldpy")
except PackageNotFoundError:  # a source tree that is not installed
    __version__ = "0+unknown"


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
