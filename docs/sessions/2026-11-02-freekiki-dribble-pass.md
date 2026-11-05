# FreeKiki dribble / pass open-play reports (2026-11-02)

## What shipped

- Planar open-play analyses on existing FreeKiki tracks:
  - **Drible (A):** driblador + up to 3 marcadores (SAM `person_id`); ball + player path length, speed, accel/decel, COD, curvature (rad/m), interpersonal distance / relative velocity.
  - **Passe (B):** passador + recebedor; pairwise distances with ball; heuristic release→receive window.
  - **Open play:** unchanged data-only set-piece kind (no 1v1/pass analysis).
- GUI (section 5): radios *Drible* / *Passe*, ID entry fields, *Browse open-play folder…*, *Report drible/passe*.
- CLI: `freekiki.py openplay-report --folder … --kind dribble|pass …`
- Outputs: `data/dribble_analysis.json` / `data/pass_analysis.json`, `report_tv/<lang>/relatorio_openplay.html` (+ PDF when Chrome is available).

## Science framing (clip metrics, not lab batteries)

- Wilson et al. 2018 J Sports Sci (PMID 30470166) — curved-path dribbling speed.
- Wilson–Santiago predator–prey COD / agility reading.
- Corrêa / Duarte — interpersonal distance and relative velocity in 1v1.
- Wilson–Santiago 2026 IJSPP — dribbling and passing vs small-sided success (pass report cards).

## Modules

| File | Role |
|------|------|
| `vaila/freekiki_openplay_kinematics.py` | Pure planar kinematics |
| `vaila/freekiki_dribble.py` | 1vN analysis + JSON |
| `vaila/freekiki_pass.py` | Pass window + JSON |
| `vaila/freekiki_openplay_report.py` | TV HTML/PDF + figures |
| `vaila/freekiki_setpiece.py` | `dribble` / `pass` choices |

## Tests

`tests/test_freekiki_openplay_kinematics.py`, `test_freekiki_dribble.py`, `test_freekiki_pass.py`, `test_freekiki_openplay_report.py`, extended `test_freekiki_setpiece.py`.
