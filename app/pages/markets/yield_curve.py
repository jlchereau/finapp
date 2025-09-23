"""Yield curve chart component."""

from datetime import datetime

import pandas as pd
import reflex as rx
import plotly.graph_objects as go

from app.lib.charts import (
    ChartConfig,
    MARKET_COLORS,
    add_historical_curves,
    get_default_theme_colors,
)
from app.lib.periods import fix_datetime
from app.flows.markets.yield_curve import fetch_yield_curve_data
from app.lib.logger import logger
from app.lib.exceptions import PageOutputException


# pylint: disable=inherit-non-class
class YieldCurveState(rx.State):
    """State for the yield curve chart component."""

    loading: rx.Field[bool] = rx.field(False)
    chart_figure: rx.Field[go.Figure] = rx.field(default_factory=go.Figure)

    def get_theme_colors(self):
        """Get neutral colors that work well in both themes."""
        return get_default_theme_colors()

    @rx.event
    async def update_yield_chart_data(self, base_date: datetime):
        """Update the yield curve chart using workflow data."""
        self.loading = True

        try:
            # Get yield curve data
            yield_result = await fetch_yield_curve_data(base_date)
            yield_data = yield_result.get("data")
            maturities = yield_result.get("maturities", [])
            latest_date = yield_result.get("latest_date")

            if yield_data is None or yield_data.empty:
                self.chart_figure = go.Figure()
                return

            # Create yield curve chart using chart utilities
            fig = go.Figure()

            # Get theme colors
            theme_colors = self.get_theme_colors()

            # Store main series data for adding later
            main_series_data = None
            if latest_date:
                latest_data = yield_data.loc[yield_data.index == latest_date]
                if not latest_data.empty:
                    main_series_data = {
                        "x": maturities,
                        "y": latest_data.iloc[0].values,
                        "date": latest_date,
                    }

            # Add historical curves with semantic colors (background elements first)
            if len(yield_data) > 1 and latest_date is not None:
                historical_dates = yield_data.index.sort_values(ascending=False)

                curves = []
                # Find dates approximately 6 months and 12 months ago
                for months_back, color, name in [
                    (6, MARKET_COLORS["warning"], "6M ago"),
                    (12, MARKET_COLORS["trend"], "12M ago"),
                ]:
                    target_date = latest_date - pd.DateOffset(months=months_back)

                    # Find closest date with more flexible approach
                    # First try exact target date or earlier
                    closest_dates = historical_dates[historical_dates <= target_date]

                    # If no dates found (edge case), try a bit more flexibility
                    if len(closest_dates) == 0:
                        # Look within ±1 week of target for edge cases
                        tolerance_start = target_date - pd.DateOffset(days=7)
                        tolerance_end = target_date + pd.DateOffset(days=7)
                        closest_dates = historical_dates[
                            (historical_dates >= tolerance_start)
                            & (historical_dates <= tolerance_end)
                        ]

                    # Debug logging to understand missing curves
                    logger.debug(
                        f"Yield curve {name}: target_date={target_date}, "
                        f"available_dates={len(closest_dates)}, "
                        f"latest_date={latest_date}, "
                        f"data_range=({yield_data.index.min()}, "
                        f"{yield_data.index.max()})"
                    )

                    if len(closest_dates) > 0:
                        hist_date = closest_dates[
                            0
                        ]  # Take the most recent within range
                        hist_data = yield_data.loc[yield_data.index == hist_date]
                        if not hist_data.empty:
                            logger.debug(f"Adding {name} curve from {hist_date}")
                            curves.append(
                                {
                                    "x": maturities,
                                    "y": hist_data.iloc[0].values,
                                    "name": f"{name} "
                                    f"({hist_date.strftime('%Y-%m-%d')})",
                                    "color": color,
                                    "opacity": 0.7,
                                }
                            )
                        else:
                            logger.warning(
                                f"Empty data for {name} curve at {hist_date}"
                            )
                    else:
                        logger.warning(
                            f"No historical data available for {name} "
                            f"(target: {target_date}, data range: "
                            f"{yield_data.index.min()} to {yield_data.index.max()})"
                        )

                add_historical_curves(fig, curves)

            # Add the current yield curve on top (drawn last, appears on top)
            if main_series_data:
                fig.add_trace(
                    go.Scatter(
                        x=main_series_data["x"],
                        y=main_series_data["y"],
                        mode="lines+markers",
                        name=f"Current "
                        f"({main_series_data['date'].strftime('%Y-%m-%d')})",
                        line={"color": MARKET_COLORS["primary"], "width": 3},
                        marker={"size": 6, "color": MARKET_COLORS["primary"]},
                        hovertemplate="<b>Current Yield Curve</b><br>"
                        + "Maturity: %{x}<br>"
                        + "Yield: %{y:.2f}%<br>"
                        + "<extra></extra>",
                    )
                )

            # Apply chart configuration and theme
            config = ChartConfig(
                title="Yield Curve",
                yaxis_title="Yield (%)",
                hover_format="Yield: %{y:.2f}%<br>",
                height=400,
            )

            # Apply theme manually since this is a custom chart type
            fig.update_layout(
                title=config.title,
                xaxis_title="Maturity",
                yaxis_title=config.yaxis_title,
                hovermode="x unified",
                showlegend=True,
                height=config.height,
                margin={"l": 50, "r": 50, "t": 80, "b": 50},
                plot_bgcolor=theme_colors["plot_bgcolor"],
                paper_bgcolor=theme_colors["paper_bgcolor"],
                hoverlabel={
                    "bgcolor": theme_colors["hover_bgcolor"],
                    "bordercolor": theme_colors["hover_bordercolor"],
                    "font_size": 14,
                    "font_color": "white",
                },
            )

            # Only add font_color if it's not None
            if theme_colors["text_color"] is not None:
                fig.update_layout(font_color=theme_colors["text_color"])

            # Update axes with theme colors and maturity ordering
            fig.update_xaxes(
                showgrid=True,
                gridwidth=1,
                gridcolor=theme_colors["grid_color"],
                showline=True,
                linewidth=1,
                linecolor=theme_colors["line_color"],
                categoryorder="array",
                categoryarray=maturities,
            )
            fig.update_yaxes(
                showgrid=True,
                gridwidth=1,
                gridcolor=theme_colors["grid_color"],
                showline=True,
                linewidth=1,
                linecolor=theme_colors["line_color"],
            )

            self.chart_figure = fig

        except Exception as e:
            # Chart generation error - wrap in PageOutputException
            raise PageOutputException(
                output_type="yield curve chart",
                message=f"Failed to generate yield curve chart: {e}",
                user_message=(
                    "Failed to generate yield curve chart. "
                    "Please try refreshing the data."
                ),
                context={"error": str(e)},
            ) from e
        finally:
            self.loading = False


@rx.event
async def update_yield_curve(state: YieldCurveState, base_date: datetime):
    """
    Decentralized event handler to update yield curve chart.
    Called from the main page when period changes.
    """
    base_date = fix_datetime(base_date)
    await state.update_yield_chart_data(base_date)


def yield_curve() -> rx.Component:
    """US Treasury yield curve chart component."""
    return rx.cond(
        YieldCurveState.loading,
        rx.center(rx.spinner(), height="400px"),
        rx.plotly(
            data=YieldCurveState.chart_figure,
            width="100%",
            height="400px",
        ),
    )
