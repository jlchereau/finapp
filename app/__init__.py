"""
Page set module for FinApp.

This is where pages are added to the reflex web app
See https://reflex.dev/docs/advanced-onboarding/code-structure/
"""

from .pages.backtest import page as backtest
from .pages.cache import page as cache
from .pages.compare import page as compare
from .pages.index import page as index
from .pages.markets import page as markets
from .pages.optimize import page as optimize
from .pages.portfolio import page as portfolio
from .pages.screen import page as screen
from .pages.test import page as test


__all__ = [
    "backtest",
    "cache",
    "compare",
    "index",
    "markets",
    "optimize",
    "portfolio",
    "screen",
    "test",
]
