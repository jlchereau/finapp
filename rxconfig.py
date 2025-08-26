"""The reflex (rx) app configuration."""

import reflex as rx

config = rx.Config(  # pylint: disable=not-callable
    app_name="app",
    # https://reflex.dev/docs/hosting/self-hosting/
    # api_url="http://localhost:8000",
    plugins=[
        rx.plugins.SitemapPlugin(),
        rx.plugins.TailwindV4Plugin(),
    ],
)
