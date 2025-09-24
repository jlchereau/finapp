"""Test compare page imports and basic functionality."""

from datetime import datetime


# Test imports to ensure the modularization works correctly
def test_compare_page_imports():
    """Test that all compare page imports work correctly."""
    # Test main page import
    from app.pages.compare.page import page

    # Test component imports
    from app.pages.compare.returns_chart import (
        returns_chart,
        update_returns_chart,
    )
    from app.pages.compare.volatility_chart import (
        volatility_chart,
        update_volatility_chart,
    )
    from app.pages.compare.volume_chart import (
        volume_chart,
        update_volume_chart,
    )
    from app.pages.compare.rsi_chart import (
        rsi_chart,
        update_rsi_chart,
    )
    from app.pages.compare.metrics import (
        metrics,
        update_metrics,
    )

    # Basic smoke tests - page() returns a component, not a callable
    assert page is not None
    assert callable(returns_chart)
    assert callable(volatility_chart)
    assert callable(volume_chart)
    assert callable(rsi_chart)
    assert callable(metrics)

    # Test event handlers are callable
    assert callable(update_returns_chart)
    assert callable(update_volatility_chart)
    assert callable(update_volume_chart)
    assert callable(update_rsi_chart)
    assert callable(update_metrics)


def test_compare_state_initialization():
    """Test that CompareState initializes correctly."""
    from app.pages.compare.page import CompareState

    # Create state instance
    state = CompareState()

    # Test default values
    assert state.active_tab == "plots"
    assert state.currency == "USD"
    assert isinstance(state.period_option, str)
    assert isinstance(state.period_options, list)
    assert len(state.period_options) > 0
    assert isinstance(state.selected_tickers, list)
    assert isinstance(state.favorites, list)
    assert len(state.favorites) > 0


def test_compare_state_methods():
    """Test CompareState methods work correctly."""
    from app.pages.compare.page import CompareState

    state = CompareState()

    # Test set_active_tab
    state.set_active_tab("metrics")
    assert state.active_tab == "metrics"

    # Test set_currency
    state.set_currency("EUR")
    assert state.currency == "EUR"

    # Test set_ticker_input
    state.set_ticker_input("AAPL")
    assert state.ticker_input == "AAPL"

    # Test base_date computed var returns datetime
    base_date = state.base_date
    assert isinstance(base_date, datetime)

    # Test run_workflows returns a list
    workflow_events = state.run_workflows()
    assert isinstance(workflow_events, list)
    assert len(workflow_events) == 5  # 4 charts + 1 metrics


def test_main_import_compatibility():
    """Test that the new structure maintains compatibility with main app imports."""
    # This tests the import used in app/__init__.py
    from app.pages.compare import page as compare

    assert compare is not None


def test_chart_state_initialization():
    """Test that individual chart states initialize correctly."""
    from app.pages.compare.returns_chart import ReturnsChartState
    from app.pages.compare.volatility_chart import VolatilityChartState
    from app.pages.compare.volume_chart import VolumeChartState
    from app.pages.compare.rsi_chart import RSIChartState
    from app.pages.compare.metrics import MetricsState

    # Test all chart states
    returns_state = ReturnsChartState(
        
    )
    assert returns_state.loading is False
    assert returns_state.chart_figure is not None

    volatility_state = VolatilityChartState()
    assert volatility_state.loading is False
    assert volatility_state.chart_figure is not None

    volume_state = VolumeChartState()
    assert volume_state.loading is False
    assert volume_state.chart_figure is not None

    rsi_state = RSIChartState()
    assert rsi_state.loading is False
    assert rsi_state.chart_figure is not None

    metrics_state = MetricsState()
    assert metrics_state.loading is False
