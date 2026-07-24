"""SAM-RSP reproduction module for iSAID-5i.

Uses lazy imports to avoid hard dependency on thirdparty/SAM-RSP at package load time.
Only triggers the import when the symbol is first accessed.
"""


def __getattr__(name: str):
    """Lazy import: defer thirdparty/SAM-RSP dependency until first access."""
    if name == "BAMModel":
        from .bam import BAMModel as _BAMModel
        return _BAMModel
    if name == "SAMRSPModel":
        from .sam_rsp_model import SAMRSPModel as _SAMRSPModel
        return _SAMRSPModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["BAMModel", "SAMRSPModel"]
