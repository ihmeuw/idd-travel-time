"""CLI entry point for idd-travel-time.

Usage — by coordinates
----------------------
    run-isochrone \\
        --lat -1.2864 --lon 36.8172 \\
        --mode walking \\
        --duration 30
    # No --output: the map goes to <maps node>/working/<result_id>.html, where the
    # maps node is <project_output_root>/maps (paths.yaml). --scratch LABEL writes
    # to maps/scratch/LABEL/ instead; --freeze / --current snapshot working/ on
    # success (idd-versions contract). --output PATH writes exactly there.

Usage — by place name
---------------------
    run-isochrone \\
        --place "Kenyatta National Hospital, Nairobi" \\
        --mode walking \\
        --duration 20 \\
        --output knh_walking.html

    run-isochrone \\
        --place "Lagos Island" \\
        --country-hint NG \\
        --mode transit \\
        --transit-preset informal_minibus \\
        --time-context rush_hour \\
        --duration 45 \\
        --scratch lagos-probe
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

import click
from idd_tools.versions import finish_if_requested, resolve_target, versions_options

from idd_travel_time.constants import TimeContext, TransportMode, VizLayer
from idd_travel_time.io.parquet import result_id
from idd_travel_time.isochrone import compute
from idd_travel_time.models import TransitProfile, TravelTimeRequest
from idd_travel_time.paths import NODE_MAPS, node_dir
from idd_travel_time.viz import save as save_map

if TYPE_CHECKING:
    from idd_tools.versions import VersionsOptions

    from idd_travel_time.routers.base import Router

_VALID_MODES = [m.value for m in TransportMode]
_VALID_CONTEXTS = [c.value for c in TimeContext]
_VALID_LAYERS = [lyr.value for lyr in VizLayer]
_VALID_ROUTERS = ["auto", "ors", "osmnx"]


def _build_router(router_name: str) -> Router | None:
    if router_name == "ors":
        from idd_travel_time.routers.openroute import (
            OpenRouteServiceRouter,
        )

        return OpenRouteServiceRouter()
    if router_name == "osmnx":
        from idd_travel_time.routers.osmnx_router import OsmnxRouter

        return OsmnxRouter()
    return None


@click.command(name="run-isochrone")
@click.option(
    "--place",
    default=None,
    type=str,
    help="Origin as a place name or address (geocoded via Nominatim). "
    "Mutually exclusive with --lat/--lon.",
)
@click.option(
    "--country-hint",
    "country_hint",
    default=None,
    type=str,
    help="ISO 3166-1 alpha-2 code to bias --place geocoding, e.g. 'NG'.",
)
@click.option(
    "--lat", default=None, type=float, help="Origin latitude (WGS-84). Requires --lon."
)
@click.option(
    "--lon", default=None, type=float, help="Origin longitude (WGS-84). Requires --lat."
)
@click.option(
    "--mode",
    required=True,
    type=click.Choice(_VALID_MODES, case_sensitive=False),
    help="Transport mode.",
)
@click.option(
    "--duration",
    default=30,
    show_default=True,
    type=int,
    help="Isochrone duration in minutes.",
)
@click.option(
    "--time-context",
    "time_context",
    default=TimeContext.OFF_PEAK.value,
    show_default=True,
    type=click.Choice(_VALID_CONTEXTS, case_sensitive=False),
    help="Time-of-day context (affects transit speed assumptions).",
)
@click.option(
    "--transit-preset",
    "transit_preset",
    default=None,
    type=str,
    help="Named TransitProfile preset (required when --mode=transit). "
    "Options: formal_bus, informal_minibus, mototaxi, metro, tuk_tuk.",
)
@click.option(
    "--router",
    default="auto",
    show_default=True,
    type=click.Choice(_VALID_ROUTERS, case_sensitive=False),
    help="Routing backend. 'auto' picks ORS if ORS_API_KEY is set, else osmnx.",
)
@click.option(
    "--layers",
    multiple=True,
    type=click.Choice(_VALID_LAYERS, case_sensitive=False),
    default=_VALID_LAYERS,
    show_default=True,
    help="Visualisation layers to include (repeat flag for multiple).",
)
@click.option(
    "--output",
    default=None,
    type=click.Path(dir_okay=False, writable=True),
    help="Output HTML file. Default: <maps node>/working/<result_id>.html "
    "(paths.yaml project_output_root). The node flags below apply only to the default.",
)
@versions_options
def main(
    place: str | None,
    country_hint: str | None,
    lat: float | None,
    lon: float | None,
    mode: str,
    duration: int,
    time_context: str,
    transit_preset: str | None,
    router: str,
    layers: tuple[str, ...],
    output: str | None,
    versions: VersionsOptions,
) -> None:
    """Compute a travel-time isochrone and save it as an interactive HTML map.

    Specify an origin either by place name (--place) or by coordinates
    (--lat + --lon). Exactly one of the two forms is required.
    """
    # --- origin resolution ---
    if place and (lat is not None or lon is not None):
        click.echo("Error: --place and --lat/--lon are mutually exclusive.", err=True)
        sys.exit(1)
    if not place and (lat is None or lon is None):
        click.echo("Error: provide either --place or both --lat and --lon.", err=True)
        sys.exit(1)
    if output is not None and (
        versions.scratch is not None or versions.finishes or versions.clean
    ):
        click.echo(
            "Error: --output names an explicit file; --scratch, --freeze, --current and "
            "--clean apply only to the default node location.",
            err=True,
        )
        sys.exit(1)

    if place:
        from idd_travel_time.locations import GeocodingError, geocode

        click.echo(f"Geocoding {place!r} …")
        try:
            lat, lon = geocode(place, country_hint=country_hint)
            click.echo(f"  → ({lat:.6f}, {lon:.6f})")
        except GeocodingError as exc:
            click.echo(f"Error: {exc}", err=True)
            sys.exit(1)

    assert lat is not None  # noqa: S101  # narrowed by --place / --lat-lon validation
    assert lon is not None  # noqa: S101  # narrowed by --place / --lat-lon validation

    # --- request construction ---
    transport_mode = TransportMode(mode)
    tc = TimeContext(time_context)
    viz_layers = [VizLayer(lyr) for lyr in layers]

    transit_profile: TransitProfile | None = None
    if transport_mode is TransportMode.TRANSIT:
        if not transit_preset:
            click.echo(
                "Error: --transit-preset is required when --mode=transit.\n"
                "Available presets: formal_bus, informal_minibus, mototaxi, metro, tuk_tuk.",
                err=True,
            )
            sys.exit(1)
        transit_profile = TransitProfile.from_preset(transit_preset, time_context=tc)

    request = TravelTimeRequest(
        origin_lat=lat,
        origin_lon=lon,
        mode=transport_mode,
        duration_minutes=duration,
        time_context=tc,
        transit_profile=transit_profile,
        viz_layers=viz_layers,
    )

    # --- output location: explicit file, or the maps node's slot ---
    if output is None:
        maps_node = node_dir(NODE_MAPS)
        # resolve_target runs the launch-time pre-flights, so a bad flag
        # combination is refused before any compute.
        out_path = resolve_target(maps_node, versions) / f"{result_id(request)}.html"
    else:
        out_path = Path(output)

    active_router = _build_router(router)
    click.echo(
        f"Computing {duration}-min {mode} isochrone from ({lat:.6f}, {lon:.6f}) "
        f"[{time_context}] using {active_router.name if active_router else 'auto'} …"
    )

    result = compute(request, router=active_router)
    saved = save_map(result, out_path, layers=viz_layers)
    click.echo(f"Saved → {saved}")
    if output is None:
        finish_if_requested(maps_node, versions)


if __name__ == "__main__":
    main()
