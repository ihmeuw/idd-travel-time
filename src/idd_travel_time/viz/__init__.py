"""Visualisation for idd-travel-time.

Public surface
--------------
render        : (TravelTimeResult, layers?) -> folium.Map    (interactive)
save          : (TravelTimeResult, path, layers?) -> Path    (interactive HTML)
render_static : (TravelTimeResult, layers?, figsize?, dpi?) -> Figure  (static)
save_static   : (TravelTimeResult, path, layers?, figsize?, dpi?) -> Path (static)
"""

from idd_travel_time.viz.folium_viz import render, save
from idd_travel_time.viz.static_viz import render_static, save_static

__all__ = ["render", "render_static", "save", "save_static"]
