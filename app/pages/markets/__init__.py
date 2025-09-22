"""Markets page components package."""

from .page import page
from .buffet_indicator import buffet_indicator, update_buffet_indicator
from .vix_chart import vix_chart, update_vix_chart
from .yield_curve import yield_curve, update_yield_curve
from .currency_chart import currency_chart, update_currency_chart
from .precious_metals import precious_metals, update_precious_metals
from .crypto_chart import crypto_chart, update_crypto_chart
from .crude_oil import crude_oil, update_crude_oil
from .bloomberg_commodity import bloomberg_commodity, update_bloomberg_commodity
from .msci_world import msci_world, update_msci_world
from .placeholder_charts import (
    fear_and_greed_chart,
    shiller_cape_chart,
    update_fear_and_greed,
    update_shiller_cape,
)

__all__ = [
    "page",
    "buffet_indicator",
    "update_buffet_indicator",
    "vix_chart",
    "update_vix_chart",
    "yield_curve",
    "update_yield_curve",
    "currency_chart",
    "update_currency_chart",
    "precious_metals",
    "update_precious_metals",
    "crypto_chart",
    "update_crypto_chart",
    "crude_oil",
    "update_crude_oil",
    "bloomberg_commodity",
    "update_bloomberg_commodity",
    "msci_world",
    "update_msci_world",
    "fear_and_greed_chart",
    "shiller_cape_chart",
    "update_fear_and_greed",
    "update_shiller_cape",
]
