"""
Gauge Chart

The Reflex Gaugechart custom component is a Reflex wrapper for the popular react-gauge-chart React library.

It provides a visually appealing and highly customisable gauge chart component, ideal for displaying percentage values or progress in a sleek, modern format.

From: https://github.com/leogaggl/reflex_gaugechart
"""

import reflex as rx
from typing import Optional


class Gaugechart(rx.NoSSRComponent):
    """Gaugechart is a custom Reflex component that provides a visual representation of a single value within a range using a gauge chart."""

    library = "react-gauge-chart"
    lib_dependencies: list[str] = ["lodash", "d3"]
    tag = "GaugeChart"
    alias = "ReflexGrauge"
    is_default = True

    className: Optional[
        rx.Var[str]
    ]  # Add className to the div container
    marginInPercent: Optional[
        rx.Var[float]
    ]  # Margin for the chart inside the containing SVG element	0.05
    cornerRadius: Optional[rx.Var[int]]  # Corner radius for the elements in the chart	6
    nrOfLevels: Optional[rx.Var[int]]  # The number of elements displayed in the arc	3
    percent: Optional[
        rx.Var[float]
    ]  # The number where the pointer should point to (between 0 and 1)	0.4
    arcPadding: Optional[
        rx.Var[float]
    ]  # The distance between the elements in the arc	0.05
    arcWidth: Optional[rx.Var[float]]  # The thickness of the arc	0.2
    colors: Optional[
        rx.Var[list]
    ]  # An array of colors in HEX format displayed in the arc	["#00FF00", "#FF0000"]
    textColor: Optional[rx.Var[str]]  # The color of the text	"#FFFFFF"
    needleColor: Optional[rx.Var[str]]  # The color of the needle triangle	"#464A4F"
    needleBaseColor: Optional[
        rx.Var[str]
    ]  # The color of the circle at the base of the needle	"#464A4F"
    hideText: Optional[
        rx.Var[bool]
    ]  # Whether or not to hide the percentage display	false
    arcsLength: Optional[
        rx.Var[list]
    ]  # An array specifying the length of each individual arc. If this prop is set, the nrOfLevels prop will have no effect
    animate: Optional[
        rx.Var[bool]
    ]  # Whether or not to animate the needle when loaded	true
    animDelay: Optional[
        rx.Var[int]
    ]  # Delay in ms before starting the needle animation	500
    animateDuration: Optional[
        rx.Var[int]
    ]  # Duration in ms for the needle animation	3000
    formatTextValue: Optional[
        rx.Var[str]
    ]  # Format you own text value (example: value => value+'%')	Null
    textComponent: Optional[rx.Var[str]]  # Custom text label textComponent	Null
    textComponentContainerClassName: Optional[
        rx.Var[str]
    ]  # Add className to the text component container
    needleScale: Optional[rx.Var[float]]  # Needle arc cornerRadius	0.55
    customNeedleComponentClassName: Optional[
        rx.Var[str]
    ]  # Add className to the custom needle container
    customNeedleStyle: Optional[rx.Var[dict]]  # dd style to custom needle container div


gaugechart = Gaugechart.create
