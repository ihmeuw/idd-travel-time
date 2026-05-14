"""Transit profile presets for idd-travel-time.

Public surface
--------------
PRESETS     : dict[str, TransitProfile] — full preset registry
list_presets: () -> list[str]
get_preset  : (str) -> TransitProfile
"""

from idd_travel_time.profiles.presets import PRESETS, get_preset, list_presets

__all__ = ["PRESETS", "get_preset", "list_presets"]
