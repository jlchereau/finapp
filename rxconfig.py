"""The reflex (rx) app configuration."""

import reflex as rx

config = rx.Config(  # pylint: disable=not-callable
    app_name="app",
    plugins=[
        rx.plugins.SitemapPlugin(),
        rx.plugins.TailwindV4Plugin(),
    ],
)
