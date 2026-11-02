# Friction-surface parameter tables

Every number here is traceable to a published table and was chosen by Bobby
(repo DECISIONS.md, 2026-10-21). The builder reads these files; nothing is
defaulted in code. Rows marked "V0 rule" or "V0 mapping" are provisional.

- land_cover_speeds_v0.csv: walking km/h per Overture land_cover subtype,
  Weiss et al. 2018 (Nature 553:333) survey medians, crosswalked.
- road_speeds_weiss2020_st1.csv: Weiss et al. 2020 (Nat Med 26:1835)
  Supplementary Table 1 verbatim: per-country median OSM speed limit per OSM
  class, plus the "Global Default" rows (iso3 "-"). Motorised speeds.
- road_class_crosswalk_v0.csv: Overture segment class -> Weiss OSM class.
- water_rule_v0.csv: per Overture water class, motorised and walking-only
  speeds (Weiss 2018 water layers; Weiss 2020 walking rule), or ignore.
- slope_rule_v0.yaml: Tobler's hiking function (capped at 5, divided by 5)
  and the altitude factor, Weiss 2018 eq. 1-3; off-road only.

Walking-only surface: every road at 5 km/h (Weiss 2020).
