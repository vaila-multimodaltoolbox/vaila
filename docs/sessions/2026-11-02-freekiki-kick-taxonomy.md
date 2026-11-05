# FreeKiki: kick (gol) vs trajetória planar (2026-11-02)

## Taxonomy

| Kind | Ball to goal? | Report |
|------|---------------|--------|
| `penalty` / `free_kick` / `free_kick_wall` | yes | flight TV |
| `kick` (chute a gol) | yes | flight TV, no wall (was former `open_play` report path) |
| `open_play` | no | data only |
| `dribble` / `pass` | no | planar `openplay-report` |

## Rule change

- `classify()` with `pre_contact` in `rolling` / `unseen` → **`kick`** (not `open_play`).
- `report-tv` rejects `open_play` / `dribble` / `pass`.
- GUI radios: Auto | Penalty | Free kick | **Kick (chute a gol)** | Open play (só dados) | Drible | Passe.

## Helpers

`is_goal_kick(kind)`, `GOAL_KINDS` in `freekiki_setpiece.py`.
