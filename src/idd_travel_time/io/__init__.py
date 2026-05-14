"""Persistence layer for idd-travel-time.

Public surface
--------------
write_result         : (TravelTimeResult, output_dir) -> Path
read_result          : (result_dir) -> TravelTimeResult
result_id            : (TravelTimeRequest) -> str
GraphCache           : persistent osmnx graph cache
default_cache_from_env : () -> GraphCache | None
"""

from idd_travel_time.io.graph_cache import GraphCache, default_cache_from_env
from idd_travel_time.io.parquet import read_result, result_id, write_result

__all__ = [
    "GraphCache",
    "default_cache_from_env",
    "read_result",
    "result_id",
    "write_result",
]
