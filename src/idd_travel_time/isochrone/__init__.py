"""Isochrone computation for idd-travel-time.

Public surface
--------------
compute          : compute one isochrone (auto-selects router)
compute_batch    : compute many isochrones sequentially, shared router
compute_parallel : compute many isochrones in parallel worker processes
"""

from idd_travel_time.isochrone.compute import compute, compute_batch, compute_parallel

__all__ = ["compute", "compute_batch", "compute_parallel"]
