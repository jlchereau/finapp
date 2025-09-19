"""
Markets flow package.

This package contains LlamaIndex workflows for financial market data collection.
All workflows follow FlowRunner architecture for consistent error handling and caching.
"""

from .bloomberg_commodity import fetch_bloomberg_commodity_data
from .buffet import fetch_buffet_indicator_data
from .crude_oil import fetch_crude_oil_data
from .crypto import fetch_crypto_data
from .currency import fetch_currency_data
from .msci_world import fetch_msci_world_data
from .precious_metals import fetch_precious_metals_data
from .vix import fetch_vix_data
from .yield_curve import fetch_yield_curve_data


__all__ = [
    "fetch_bloomberg_commodity_data",
    "fetch_buffet_indicator_data",
    "fetch_crude_oil_data",
    "fetch_crypto_data",
    "fetch_currency_data",
    "fetch_msci_world_data",
    "fetch_precious_metals_data",
    "fetch_vix_data",
    "fetch_yield_curve_data",
]
