"""Navbar component."""

import reflex as rx
from .theme_toggle import theme_toggle


def navbar() -> rx.Component:
    """
    Create a horizontal navbar with logo, navigation links,
    and dark mode toggle.
    """
    return rx.box(
        rx.hstack(
            # Logo section
            rx.hstack(
                rx.link(
                    rx.image(
                        src="/logo.svg",
                        height="2rem",
                        width="auto",
                    ),
                    href="/",
                    text_decoration="none",
                ),
                spacing="3",
                align="center",
            ),

            # Navigation links
            rx.hstack(
                rx.link(
                    "Markets",
                    href="/markets",
                    color=rx.color("gray", 11),
                    text_decoration="none",
                    _hover={
                        "color": rx.color("accent", 10),
                    },
                ),
                rx.link(
                    "Portfolio",
                    href="/portfolio",
                    color=rx.color("gray", 11),
                    text_decoration="none",
                    _hover={
                        "color": rx.color("accent", 10),
                    },
                ),
                rx.link(
                    "Screen",
                    href="/screen",
                    color=rx.color("gray", 11),
                    text_decoration="none",
                    _hover={
                        "color": rx.color("accent", 10),
                    },
                ),
                rx.link(
                    "Compare",
                    href="/compare",
                    color=rx.color("gray", 11),
                    text_decoration="none",
                    _hover={
                        "color": rx.color("accent", 10),
                    },
                ),
                rx.link(
                    "Optimize",
                    href="/optimize",
                    color=rx.color("gray", 11),
                    text_decoration="none",
                    _hover={
                        "color": rx.color("accent", 10),
                    },
                ),
                rx.link(
                    "Backtest",
                    href="/backtest",
                    color=rx.color("gray", 11),
                    text_decoration="none",
                    _hover={
                        "color": rx.color("accent", 10),
                    },
                ),
                rx.link(
                    "Workflows",
                    href="/workflows",
                    color=rx.color("gray", 11),
                    text_decoration="none",
                    _hover={
                        "color": rx.color("accent", 10),
                    },
                ),
                spacing="6",
                align="center",
            ),

            # Theme toggle
            theme_toggle(),

            justify="between",
            align="center",
            width="100%",
        ),
        bg=rx.color("accent", 3),
        padding="1em",
        position="sticky",
        top="0",
        z_index="1000",
        width="100%",
    )
