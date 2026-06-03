"""Engine package exports.

Keep heavy model dependencies lazy so importing a light submodule such as
``engines.rgba_postprocess`` does not load transformers/torch model code during
app startup.
"""

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .mobile_sam import MobileSAMEngine
    from .rmbg2 import RMBG2Engine
    from .sam_hq import SAMHQEngine

__all__ = ["RMBG2Engine", "MobileSAMEngine", "SAMHQEngine"]


def __getattr__(name):
    if name == "RMBG2Engine":
        return import_module(".rmbg2", __name__).RMBG2Engine
    if name == "MobileSAMEngine":
        return import_module(".mobile_sam", __name__).MobileSAMEngine
    if name == "SAMHQEngine":
        return import_module(".sam_hq", __name__).SAMHQEngine
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
