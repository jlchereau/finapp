---
applyTo: "app/flows/**/*.py"
---

# Workflows Design and Concurrency Model

This document outlines the design principles for workflows in the FinApp project, particularly concerning concurrency and separation of concerns.

## Workflow Engine

We have considered several workflow brokers:
- [Airflow](https://github.com/apache/incubator-airflow)
- [Celery](https://github.com/celery/celery)
- [Prefect](https://github.com/PrefectHQ/Prefect)

Considering FinApp is currently a single-user application, we have opted for a lightweight, in-process, in-memory solution using [LlamaIndex Workflows (standalone module)](https://pypi.org/project/llama-index-workflows/). These workflows are launched as [Reflex background tasks](https://reflex.dev/docs/events/background-events/#background-tasks).

## Two-Tier Concurrency Model

A clear separation of concerns exists between workflow-level and provider-level concurrency.

### 1. Workflow-Level Concurrency (Orchestration)

- **Responsibility**: The `app/flows/` directory is responsible for orchestrating **what** high-level tasks run in parallel. For example, fetching data for 20 different tickers simultaneously.
- **Implementation**: This is achieved using the `@step(num_workers=N)` decorator from LlamaIndex Workflows.
- **Goal**: To maximize the application's local resource utilization (CPU, memory, network bandwidth) for overall throughput.
- **Principle**: The workflow layer should remain agnostic to the specific constraints of individual data providers. It orchestrates the "business logic" flow and trusts the provider layer to handle its own implementation details.

### 2. Provider-Level Concurrency (API Politeness)

- **Responsibility**: The `app/providers/` providers are responsible for managing **how** they interact with external APIs. This includes respecting API rate limits.
- **Implementation**: This is handled by an `asyncio.Semaphore` within the `BaseProvider` class.
- **Goal**: To prevent being rate-limited or blocked by external services (e.g., `429 Too Many Requests` errors) by controlling the number of concurrent requests to a specific API endpoint. This is a measure of safety and robustness.

### How They Interact

The two levels work together seamlessly. A workflow can attempt to run 20 tasks in parallel using `num_workers=20`. However, if all those tasks call a provider that has a semaphore limit of 10, the provider itself will act as a gatekeeper, ensuring only 10 requests are sent to the external API at any one time. The other 10 tasks will wait gracefully for a slot to open.

This design encapsulates provider-specific constraints within the provider itself, leading to a more resilient and maintainable system.

## Workflow Responsibilities

Workflows in `app/flows/` serve as the data assembly layer, orchestrating multiple providers to create comprehensive datasets:

### Core Functions
- **Data Aggregation**: Combine data from multiple providers (Yahoo, Zacks, TipRanks, etc.)
- **Error Coordination**: Handle scenarios where some providers succeed and others fail
- **Business Logic**: Implement domain-specific logic for financial data processing
- **Parallel Execution**: Maximize throughput by running provider calls concurrently
- **Result Assembly**: Structure data for consumption by Reflex pages

### Separation from Providers
- **Workflows**: Focus on *what* data to fetch and *how* to combine it
- **Providers**: Focus on *how* to fetch data from specific sources
- **Clear Interface**: Workflows consume `ProviderResult` objects with standardized error handling

## Implementation Examples

### Basic Workflow Pattern
```python
from llama_index.workflows import Workflow, step, Event
from app.providers.yahoo import create_yahoo_info_provider
from app.providers.zacks import create_zacks_provider

class FinancialDataWorkflow(Workflow):
    @step(num_workers=3)
    async def fetch_all_data(self, ticker: str):
        # Coordinate multiple providers concurrently
        yahoo_provider = create_yahoo_info_provider()
        zacks_provider = create_zacks_provider()
        
        results = await asyncio.gather(
            yahoo_provider.get_data(ticker),
            zacks_provider.get_data(ticker),
            return_exceptions=True
        )
        
        # Assemble results with error handling
        return self.process_results(results)
```