"""
Page set module for FinApp.

This is where pages are added to the reflex web app
See https://reflex.dev/docs/advanced-onboarding/code-structure/
"""


# from . import state, models
from .pages import (
    backtest,
    cache,
    compare,
    index,
    markets,
    optimize,
    portfolio,
    screen,
    test,
)

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
