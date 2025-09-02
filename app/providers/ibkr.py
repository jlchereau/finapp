"""
IBKR provider module
This module provides functionality to fetch data from Interactive Brokers API.

Requires IB Gateway which can be found at
https://www.interactivebrokers.com/en/trading/ibgateway-latest.php.
"""

import asyncio
import threading
from decimal import Decimal
from pandas import DataFrame, Index
from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract

from app.lib.logger import logger
from app.lib.settings import settings
from .base import (
    BaseProvider,
    ProviderType,
    ProviderConfig,
    NonRetriableProviderException,
    RetriableProviderException,
)
from .cache import cache


class IBKRApp(EWrapper, EClient):
    """
    IBKR application class for connecting to Interactive Brokers Gateway.
    Handles both positions and cash data collection.
    """

    def __init__(self):
        EClient.__init__(self, self)
        self.positions_data = []
        self.cash_data = []
        self.positions_ready = threading.Event()
        self.cash_ready = threading.Event()
        self.connected = threading.Event()
        self.next_order_id = None

    def error(self, reqId, errorCode, errorString, advancedOrderRejectJson=""):
        """Handle error messages from IBKR."""
        _ = advancedOrderRejectJson  # Unused parameter
        if errorCode in (2104, 2106, 2107, 2158):
            # These are informational messages, not actual errors
            logger.debug(f"IBKR info {errorCode}: {errorString}")
            return
        logger.warning(f"IBKR error {reqId} {errorCode}: {errorString}")

    def nextValidId(self, orderId):
        """Called when connection is established."""
        self.next_order_id = orderId
        logger.info(f"IBKR connected, next valid order ID: {orderId}")
        self.connected.set()

    def position(
        self, account: str, contract: Contract, position: Decimal, avgCost: float
    ):
        """Handle position updates."""
        self.positions_data.append(
            {
                "account": account,
                "symbol": contract.symbol,
                "secType": contract.secType,
                "currency": contract.currency,
                "exchange": contract.exchange,
                "position": position,
                "avgCost": avgCost,
            }
        )

    def positionEnd(self):
        """Called when all positions have been received."""
        logger.info(f"Received {len(self.positions_data)} positions from IBKR")
        self.positions_ready.set()

    def updateAccountValue(self, key, val, currency, accountName):
        """Handle account value updates."""
        if key == "CashBalance":
            self.cash_data.append(
                {"account": accountName, "currency": currency, "value": float(val)}
            )

    def accountDownloadEnd(self, accountName):
        """Called when account download is complete."""
        logger.info(f"Account download complete for: {accountName}")
        self.cash_ready.set()


def run_ibkr_loop(app):
    """Helper function to run IBKR app loop in a thread."""
    app.run()


class IBKRPositionsProvider(BaseProvider[DataFrame]):
    """
    Provider for fetching positions data from IBKR API
    """

    def _get_provider_type(self) -> ProviderType:
        """Return the provider type."""
        return ProviderType.IBKR_POSITIONS

    @cache
    # @cache loses pyrefly - no easy fix
    # pyrefly: ignore[bad-override]
    async def _fetch_data(self, query: str | None, *args, **kwargs) -> DataFrame:
        """
        Fetch positions data from the IBKR API

        Args:
            query: Not used for positions (set to None)
            **kwargs: Additional parameters (currently unused)

        Returns:
            DataFrame with position data

        Raises:
            NonRetriableProviderException: For connection failures
            RetriableProviderException: For temporary failures
        """
        _ = query, kwargs  # Unused parameters for positions

        def fetch_positions():
            """Inner function to run IBKR operations in a thread."""
            app = IBKRApp()

            try:
                # Connect to IB Gateway
                logger.info(
                    f"Connecting to IBKR at localhost:{settings.IB_GATEWAY_PORT} "
                    f"client_id={settings.IB_GATEWAY_CLIENT_ID}"
                )
                app.connect(
                    "127.0.0.1", settings.IB_GATEWAY_PORT, settings.IB_GATEWAY_CLIENT_ID
                )

                # Start the app in a separate thread
                app_thread = threading.Thread(
                    target=run_ibkr_loop, args=(app,), daemon=True
                )
                app_thread.start()

                # Wait for connection
                if not app.connected.wait(timeout=10):
                    app.disconnect()
                    raise ConnectionError(
                        "Failed to connect to IBKR Gateway within 10 seconds"
                    )

                # Request positions
                logger.info("Requesting positions from IBKR")
                app.reqPositions()

                # Wait for positions data
                if not app.positions_ready.wait(timeout=30):
                    app.disconnect()
                    raise TimeoutError("Timeout waiting for positions data from IBKR")

                # Convert to DataFrame
                if not app.positions_data:
                    logger.warning("No positions data received from IBKR")
                    df_positions = DataFrame(
                        columns=Index(
                            [
                                "account",
                                "symbol",
                                "secType",
                                "currency",
                                "exchange",
                                "position",
                                "avgCost",
                            ]
                        )
                    )
                else:
                    df_positions = DataFrame(app.positions_data)
                    logger.info(f"Successfully retrieved {len(df_positions)} positions")

                # Clean disconnect
                app.disconnect()
                return df_positions

            except Exception as e:
                try:
                    app.disconnect()
                except Exception:  # pylint: disable=broad-except
                    pass
                raise e

        try:
            # Run IBKR operations in a separate thread
            data = await asyncio.to_thread(fetch_positions)
            return data

        except ConnectionError as e:
            logger.error(f"Connection error in IBKRPositionsProvider: {e}")
            raise NonRetriableProviderException(
                f"Failed to connect to IBKR Gateway: {e}"
            ) from e
        except TimeoutError as e:
            logger.warning(f"Timeout error in IBKRPositionsProvider: {e}")
            raise RetriableProviderException(f"Timeout fetching positions: {e}") from e
        except Exception as e:
            logger.warning(
                f"Retriable error in IBKRPositionsProvider: {type(e).__name__}: {e}"
            )
            raise RetriableProviderException(str(e)) from e


class IBKRCashProvider(BaseProvider[DataFrame]):
    """
    Provider for fetching cash data from IBKR API
    """

    def _get_provider_type(self) -> ProviderType:
        """Return the provider type."""
        return ProviderType.IBKR_CASH

    @cache
    # @cache loses pyrefly - no easy fix
    # pyrefly: ignore[bad-override]
    async def _fetch_data(self, query: str | None, *args, **kwargs) -> DataFrame:
        """
        Fetch cash data from the IBKR API

        Args:
            query: Not used for cash data (set to None)
            **kwargs: Additional parameters (currently unused)

        Returns:
            DataFrame with cash balance data

        Raises:
            NonRetriableProviderException: For connection failures
            RetriableProviderException: For temporary failures
        """
        _ = query, kwargs  # Unused parameters for cash

        def fetch_cash():
            """Inner function to run IBKR operations in a thread."""
            app = IBKRApp()

            try:
                # Connect to IB Gateway
                logger.info(
                    f"Connecting to IBKR at localhost:{settings.IB_GATEWAY_PORT} "
                    f"client_id={settings.IB_GATEWAY_CLIENT_ID}"
                )
                app.connect(
                    "127.0.0.1", settings.IB_GATEWAY_PORT, settings.IB_GATEWAY_CLIENT_ID
                )

                # Start the app in a separate thread
                app_thread = threading.Thread(
                    target=run_ibkr_loop, args=(app,), daemon=True
                )
                app_thread.start()

                # Wait for connection
                if not app.connected.wait(timeout=10):
                    app.disconnect()
                    raise ConnectionError(
                        "Failed to connect to IBKR Gateway within 10 seconds"
                    )

                # Request account updates
                logger.info("Requesting account data from IBKR")
                app.reqAccountUpdates(True, "")

                # Wait for account data
                if not app.cash_ready.wait(timeout=30):
                    app.disconnect()
                    raise TimeoutError("Timeout waiting for account data from IBKR")

                # Stop account updates and prepare DataFrame
                app.reqAccountUpdates(False, "")

                # Convert to DataFrame
                if not app.cash_data:
                    logger.warning("No cash data received from IBKR")
                    df_cash = DataFrame(columns=Index(["account", "currency", "value"]))
                else:
                    df_cash = DataFrame(app.cash_data)
                    logger.info(f"Successfully retrieved {len(df_cash)} cash entries")

                # Clean disconnect
                app.disconnect()
                return df_cash

            except Exception as e:
                try:
                    app.disconnect()
                except Exception:  # pylint: disable=broad-except
                    pass
                raise e

        try:
            # Run IBKR operations in a separate thread
            data = await asyncio.to_thread(fetch_cash)
            return data

        except ConnectionError as e:
            logger.error(f"Connection error in IBKRCashProvider: {e}")
            raise NonRetriableProviderException(
                f"Failed to connect to IBKR Gateway: {e}"
            ) from e
        except TimeoutError as e:
            logger.warning(f"Timeout error in IBKRCashProvider: {e}")
            raise RetriableProviderException(f"Timeout fetching cash data: {e}") from e
        except Exception as e:
            logger.warning(
                f"Retriable error in IBKRCashProvider: {type(e).__name__}: {e}"
            )
            raise RetriableProviderException(str(e)) from e


# Factory functions for easy provider creation
def create_ibkr_positions_provider(
    timeout: float = 60.0,
    retries: int = 3,
) -> IBKRPositionsProvider:
    """
    Factory function to create an IBKR Positions provider with custom settings.

    Args:
        timeout: Request timeout in seconds (default: 60s for IBKR operations)
        retries: Number of retry attempts

    Returns:
        Configured IBKRPositionsProvider instance
    """
    config = ProviderConfig(timeout=timeout, retries=retries)
    return IBKRPositionsProvider(config)


def create_ibkr_cash_provider(
    timeout: float = 60.0,
    retries: int = 3,
) -> IBKRCashProvider:
    """
    Factory function to create an IBKR Cash provider with custom settings.

    Args:
        timeout: Request timeout in seconds (default: 60s for IBKR operations)
        retries: Number of retry attempts

    Returns:
        Configured IBKRCashProvider instance
    """
    config = ProviderConfig(timeout=timeout, retries=retries)
    return IBKRCashProvider(config)
