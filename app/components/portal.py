"""A portal component for rendering children into a different part of the DOM."""

from .headlessui import HeadlessUIComponent


class Portal(HeadlessUIComponent):
    """A portal component for rendering children into a different part of the DOM."""

    tag = "Portal"


# A convenience function to create a Portal component
portal = Portal.create