"""Travel-time surface computation and evaluation.

Public surface
--------------
BoundingBox             : geographic bounding box
TravelTimeSurface       : raster output (data + affine + CRS)
SurfaceRouter           : abstract backend interface
OsmnxSurfaceRouter      : Dijkstra on OSM street graph
FrictionSurfaceRouter   : MCP_Geometric on friction raster
EnsembleSurfaceRouter   : weighted combination of backends
score_at_points         : point-based metrics vs ground truth
compare_surfaces        : pixel-wise comparison of two surfaces
benchmark_table         : tidy metrics table across backends
run_benchmark           : full benchmark pipeline
make_synthetic_ground_truth : synthetic GT for testing/demos
"""

from idd_travel_time.access.surface import BoundingBox, TravelTimeSurface
from idd_travel_time.access.surface_router import SurfaceRouter
from idd_travel_time.access.backends.osmnx_surface import OsmnxSurfaceRouter
from idd_travel_time.access.backends.friction_surface import FrictionSurfaceRouter
from idd_travel_time.access.ensemble import EnsembleSurfaceRouter
from idd_travel_time.access.metrics import (
    score_at_points,
    compare_surfaces,
    benchmark_table,
    residual_geodataframe,
)
from idd_travel_time.access.benchmark import run_benchmark, make_synthetic_ground_truth

__all__ = [
    "BoundingBox",
    "EnsembleSurfaceRouter",
    "FrictionSurfaceRouter",
    "OsmnxSurfaceRouter",
    "SurfaceRouter",
    "TravelTimeSurface",
    "benchmark_table",
    "compare_surfaces",
    "make_synthetic_ground_truth",
    "residual_geodataframe",
    "run_benchmark",
    "score_at_points",
]
