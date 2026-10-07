"""backprop — tiny scalar autograd engine + neural net library."""

from .engine import Value
from . import nn

__version__ = "0.1.0"
__all__ = ["Value", "nn", "__version__"]
