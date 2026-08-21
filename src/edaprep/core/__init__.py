"""Core abstractions: the transformer contract, fit context, journal, and pipeline."""

from .base import Transformer, check_is_fitted
from .context import FitContext
from .journal import Journal, JournalEntry

__all__ = ["Transformer", "check_is_fitted", "FitContext", "Journal", "JournalEntry"]
