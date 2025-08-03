"""
llama-index-workflows for markets page.
"""

import asyncio
import random
from typing import Dict, Any

from workflows import Workflow, step
from workflows.events import Event, StartEvent, StopEvent


class SeedGeneratedEvent(Event):
    """Event emitted when seed is generated."""
    ticker: str
    seed: int


class StockPriceWorkflow(Workflow):
    """Workflow that generates a stock price for a ticker in two steps."""

    @step
    async def generate_seed(self, ev: StartEvent) -> SeedGeneratedEvent:
        """First step: sleep for 3 seconds and create a seed."""
        ticker = ev.ticker
        await asyncio.sleep(3)
        seed = random.randint(1, 1000000)
        return SeedGeneratedEvent(ticker=ticker, seed=seed)

    @step
    async def generate_stock_price(self, ev: SeedGeneratedEvent) -> StopEvent:
        """Second step: sleep for 3 seconds and create a random stock price."""
        await asyncio.sleep(3)
        random.seed(ev.seed)
        # Generate a realistic stock price between $10 - $500
        stock_price = random.uniform(10.0, 500.0)
        return StopEvent(result={
            "ticker": ev.ticker, 
            "price": round(stock_price, 2)
        })


async def process_ticker(ticker: str) -> Dict[str, Any]:
    """Process a ticker through the stock price workflow."""
    try:
        workflow = StockPriceWorkflow()
        start_event = StartEvent(ticker=ticker)
        handler = workflow.run(start_event=start_event)
        result = await handler
        
        # Check if result has a .result attribute
        if hasattr(result, 'result'):
            return result.result
        else:
            # If no .result attribute, return the result directly
            return result
            
    except Exception as e:
        return {"ticker": ticker, "error": str(e)}