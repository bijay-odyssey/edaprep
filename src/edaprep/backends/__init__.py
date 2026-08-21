"""Execution backends.

A narrow protocol between the library and a dataframe implementation.  See
:mod:`edaprep.backends.base` for what it covers and, more importantly, what it
deliberately does not.
"""

from .base import Backend, get_backend, register_backend
from .pandas_backend import PandasBackend

__all__ = ["Backend", "PandasBackend", "get_backend", "register_backend"]
