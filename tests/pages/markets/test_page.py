"""Test markets page imports and basic functionality."""

from datetime import datetime


# Test imports to ensure the modularization works correctly
def test_markets_page_imports():
    """Test that all markets page imports work correctly."""
    # Test main page import
    from app.pages.markets.page import page

    # Test component imports
    from app.pages.markets.buffet_indicator import (
        buffet_indicator,
        update_buffet_indicator,
    )
    from app.pages.markets.vix_chart import (
        vix_chart,
        update_vix_chart,
    )
    from app.pages.markets.yield_curve import (
        yield_curve,
        update_yield_curve,
    )
    from app.pages.markets.currency_chart import (
        currency_chart,
        update_currency_chart,
    )
    from app.pages.markets.precious_metals import (
        precious_metals,
        update_precious_metals,
    )
    from app.pages.markets.crypto_chart import (
        crypto_chart,
        update_crypto_chart,
    )
    from app.pages.markets.crude_oil import crude_oil, update_crude_oil
    from app.pages.markets.bloomberg_commodity import (
        bloomberg_commodity,
        update_bloomberg_commodity,
    )
    from app.pages.markets.msci_world import (
        msci_world,
        update_msci_world,
    )
    from app.pages.markets.placeholder_charts import (
        fear_and_greed_chart,
        shiller_cape_chart,
        update_fear_and_greed,
        update_shiller_cape,
    )

    # Basic smoke tests - page() returns a component, not a callable
    assert page is not None
    assert callable(buffet_indicator)
    assert callable(vix_chart)
    assert callable(yield_curve)
    assert callable(currency_chart)
    assert callable(precious_metals)
    assert callable(crypto_chart)
    assert callable(crude_oil)
    assert callable(bloomberg_commodity)
    assert callable(msci_world)
    assert callable(fear_and_greed_chart)
    assert callable(shiller_cape_chart)

    # Test event handlers are callable
    assert callable(update_buffet_indicator)
    assert callable(update_vix_chart)
    assert callable(update_yield_curve)
    assert callable(update_currency_chart)
    assert callable(update_precious_metals)
    assert callable(update_crypto_chart)
    assert callable(update_crude_oil)
    assert callable(update_bloomberg_commodity)
    assert callable(update_msci_world)
    assert callable(update_fear_and_greed)
    assert callable(update_shiller_cape)


def test_market_state_initialization():
    """Test that MarketState initializes correctly."""
    from app.pages.markets.page import MarketState

    # Create state instance
    state = MarketState()

    # Test default values
    assert state.active_tab == "overview"
    assert isinstance(state.period_option, str)
    assert isinstance(state.period_options, list)
    assert len(state.period_options) > 0


def test_market_state_methods():
    """Test MarketState methods work correctly."""
    from app.pages.markets.page import MarketState

    state = MarketState(
        active_tab="overview",
        period_option="1Y",
        period_options=["1Y"],
    )

    # Test set_active_tab
    state.set_active_tab("us")
    assert state.active_tab == "us"

    # Test base_date computed var returns datetime
    base_date = state.base_date
    assert isinstance(base_date, datetime)

    # Test run_workflows returns a list
    workflow_events = state.run_workflows()
    assert isinstance(workflow_events, list)
    assert len(workflow_events) == 9  # All 9 charts are now implemented


def test_main_import_compatibility():
    """Test that the new structure maintains compatibility with main app imports."""
    # This tests the import used in app/__init__.py
    from app.pages.markets import page as markets

    assert markets is not None
