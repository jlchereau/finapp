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

- **Responsibility**: The `app/models/` providers are responsible for managing **how** they interact with external APIs. This includes respecting API rate limits.
- **Implementation**: This is handled by an `asyncio.Semaphore` within the `BaseProvider` class.
- **Goal**: To prevent being rate-limited or blocked by external services (e.g., `429 Too Many Requests` errors) by controlling the number of concurrent requests to a specific API endpoint. This is a measure of safety and robustness.

### How They Interact

The two levels work together seamlessly. A workflow can attempt to run 20 tasks in parallel using `num_workers=20`. However, if all those tasks call a provider that has a semaphore limit of 10, the provider itself will act as a gatekeeper, ensuring only 10 requests are sent to the external API at any one time. The other 10 tasks will wait gracefully for a slot to open.

This design encapsulates provider-specific constraints within the provider itself, leading to a more resilient and maintainable system.


# Workflows

We have considered the following workflow brokers:
    - [Airflow](https://github.com/apache/incubator-airflow)
    - [Celery](https://github.com/celery/celery)
    - [Prefect](https://github.com/PrefectHQ/Prefect)

Considering FinApp is currently a single-user application, we have opted for a more lightweight in-process in-memory solution with [LlamaIndex Workflows (standalone module)](https://pypi.org/project/llama-index-workflows/) launched by [reflex background tasks](https://reflex.dev/docs/events/background-events/#background-tasks).