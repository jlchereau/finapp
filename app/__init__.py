"""
Page set module for FinApp.

This is where pages are added to the reflex web app
See https://reflex.dev/docs/advanced-onboarding/code-structure/
"""

# from . import state, models
from .pages import (
    backtest,
    cache,
    index,
    optimize,
    portfolio,
    screen,
    test,
)
from .pages.markets import page as markets
from .pages.compare import page as compare

__all__ = [
    # "state",
    # "models",
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
