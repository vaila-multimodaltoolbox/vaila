# soccerfield_vitruvian_dlt3d

## Module information

- **File:** `vaila/soccerfield_vitruvian_dlt3d.py`
- **Version:** 0.3.104 (11 August 2026)
- **Interface:** CLI and Python API
- **Purpose:** time-varying DLT3D calibration from pitch landmarks plus Vitruvian player-box verticals

## What it computes

The module combines two geometrically different control sources:

1. visible pitch landmarks with known world coordinates `(X,Y,0)`;
2. non-coplanar vertical controls.

For a tracked player, `sam_vaila_bottom.csv` supplies a foot/ground proxy. The
per-frame pitch homography maps that pixel to `(X,Y,0)`. The matching point in
`sam_vaila_top.csv` is assigned `(X,Y,h_i)`, where `h_i` comes from an explicit
track-height table, the existing transparent bbox-height ranking heuristic, or
a declared default height.

Known goalpost or scene verticals can be supplied with `--known-verticals` and
receive a larger default weight than bbox controls. This makes the player
vertical a fallback when the goal structure is not visible.

The eight DLT coefficients observable on `Z=0` are fitted first. The remaining
three coefficients `(L3,L7,L11)`, which multiply `Z`, are solved from the
vertical controls. At least **two spatially distinct verticals** are required,
and the code rejects a vertical design matrix whose rank is below three.

## Scientific limits

This operation calibrates a full projective camera; it is not single-camera
triangulation of arbitrary joints. A bbox top is affected by pose, jumping,
occlusion, and segmentation padding. It is therefore down-weighted by default
and its reprojection residual and height provenance are always exported.

The module estimates a DLT row independently for each frame that has enough
field and vertical evidence. It does **not** linearly interpolate raw DLT
coefficients. Frames that fail the point-count or rank checks are skipped and
recorded in the report.

## Inputs

| Argument | Input |
|---|---|
| `--field-pixels` | wide `frame,p1_x,p1_y,...` pitch-keypoint CSV |
| `--field-ref` | pitch reference CSV with `x,y` world coordinates |
| `--bbox-bottom` | SAM `sam_vaila_bottom.csv` (`frame,x1,y1,...`) |
| `--bbox-top` | SAM `sam_vaila_top.csv` |
| `--heights` | optional roster or explicit track-height CSV |
| `--known-verticals` | optional measured goalpost/scene verticals |

An explicit height table can contain `track`/`track_id`/`person_slot` and
`height_m` (or `height_cm`). Without a track column, player and bbox heights are
ranked exactly as a declared heuristic. Unassigned tracks use
`--default-height-m` (1.80 m by default).

Known verticals use the long columns:

```text
frame,name,world_x,world_y,height_m,top_u_px,top_v_px
```

## Example

```bash
uv run python -m vaila.soccerfield_vitruvian_dlt3d \
  --field-pixels runs/pitch/field_keypoints_getpixelvideo.csv \
  --bbox-bottom runs/sam/VIDEO/sam_vaila_bottom.csv \
  --bbox-top runs/sam/VIDEO/sam_vaila_top.csv \
  --heights anthropometry_match.csv \
  --output runs/vitruvian_dlt3d
```

Restrict calibration to selected anchor frames by repeating `--frame`:

```bash
uv run python -m vaila.soccerfield_vitruvian_dlt3d \
  --field-pixels field.csv --bbox-bottom bottom.csv --bbox-top top.csv \
  --frame 0 --frame 30 --frame 60 --output anchors/
```

## Outputs

```text
vitruvian_timevarying.dlt3d
vitruvian_timevarying_report.csv
vitruvian_timevarying_controls.csv
vitruvian_timevarying_height_assignments.csv
```

The `.dlt3d` file is compatible with vailá's multirow DLT consumers. The report
contains field, bbox-bottom, and vertical reprojection RMS, along with rank and
conditioning. The controls file makes every assumed world vertical auditable.

## Related modules

- `soccerfield_keypoints_ai.py` — 32 field landmarks
- `soccerfield_calib.py` — planar `Z=0` calibration only
- `sam_postprocess.py` — bbox bottom/top exports
- `fifa_anthropometry.py` — roster parsing and anthropometric scale helpers
- `fifa_to_dlt.py` — DLT from supplied physical camera parameters
- `monocular_dlt_align.py` — constrained world placement of a camera-frame body

