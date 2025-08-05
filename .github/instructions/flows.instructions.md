---
applyTo: "app/flows/**/*.py"
---

# Workflows

We have considered the following workflow brokers:
    - [Airflow](https://github.com/apache/incubator-airflow)
    - [Celery](https://github.com/celery/celery)
    - [Prefect](https://github.com/PrefectHQ/Prefect)

Considering FinApp is currently a single-user application, we have opted for a more lightweight in-process in-memory solution with [LlamaIndex Workflows (standalone module)](https://pypi.org/project/llama-index-workflows/) launched by [reflex background tasks](https://reflex.dev/docs/events/background-events/#background-tasks).