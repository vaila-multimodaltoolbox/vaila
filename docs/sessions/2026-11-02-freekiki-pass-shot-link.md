# FreeKiki pass report: timing, receiver run, shot link (2026-11-02)

## Play

`Lance_decifrado/pal_vas01nov`: pass id21 → id51 (`passe/passe_3d_20261102_003906/c1_passe_main`), then
51's shot (`chute/.../kick_analysis_20261102_022721/c2_chute_16mm_frame_6_to_20_3d_20261102_022721`).

Command that made the report:

```bash
uv run --no-sync vaila/freekiki.py openplay-report --folder <c1_passe_main> --kind pass \
  --passer 21 --receiver 51 --arrival-frame 24 --shot-folder <kick 3D folder> --lang both
```

## Bugs found in the first pass report (all fixed)

1. **fps 30 instead of 23.976**: `detect_fps` only read JSON keys. It now reads the README `video:` line, then
   `*_3d_overlay.mp4` (`detect_fps_source`). Old speeds were 1.25× too high.
2. **Ball under the pitch**: `ball_3d_on_field.csv` had z down to −0.81 m and ended ~5 m past the receiver.
   `ball_source(auto)` switches to `ball_3d.csv` (ground ray) when the flight file has z < −0.1 m; ground rows
   with score < 0.5 are dropped.
3. **Constant-velocity player tracks**: `ground_tracks.csv` `x_m/y_m` come from `PLAYER_MODEL` (accel 2 m/s²),
   so 51 ran at exactly 6.48 m/s. The pass now uses `raw_x_m/raw_y_m` + `smooth_track` (local-line Hampel,
   floor 0.3 m; Butterworth 2 Hz with **odd** padding: `filter_utils.butter_filter` pads with an even
   reflection, which forces the speed to zero at the clip edges, so it is not used here).
4. **Release/arrival**: release = flight-fit `kick_time_s` (−0.039 s; back-extrapolation −0.055 s) when the ball
   already moves in frame 0. Arrival by closest approach picked frame 21 (ball crosses in front of the box-base
   point); the video shows the first touch at frame 24, so `--arrival-frame 24` was used.

## Numbers (pt report)

Passer → arrival 15.2 m; release → first touch 1.04 s; ball 73 km/h at release (flight fit 20.4 m/s agrees),
31 km/h at arrival, 51 km/h mean; 51 already at 17 km/h in the first frame → run started ≥ 0.9 s before the
pass (estimate, 0.6–1.1 s for a_max 7–4 m/s²); 51 moved 5.0 m during the pass, peak 25 km/h. Shot: 92 km/h
launch (estimate, same code as the shot report), 10.2 m from the goal line, 2.9 m after the first touch;
time between clips not measured (0.3–0.9 s bracket).

## Not done

- `freekiki_player_ids._fps_of` still has its own README/cv2 fps code (not moved to the shared helper).
- The automatic arrival (closest approach on ground points) is ~3 frames early on this clip; a foot-based
  rule (SAM 3D Body foot keypoints) would remove the need for `--arrival-frame`.

## TV package (same deliverables as the dribble package)

`vaila/freekiki_pass_tv.py`, called by `openplay-report --kind pass --tv-package DIR` (GUI: *Pacote TV
(vídeos)*). Delivered to `Lance_decifrado/pal_vas01nov/passe/enviar/resultados_passe_tv_20261102/`
(17 MB): `explicacao_tv.mp4` (1800 frames, 60 s, 30 fps), `lance_sam3body.mp4` (26 frames, 991001/41333),
`chute_animacao_bola.mp4` (copied from the shot report), `relatorio_passe.pdf/html` (+ `_en`), captions,
four check images, `data/`, checksums and `validacao_entrega.json` (all three checks true). Built in 37 s.
Reuses `_text`, `encode_frames`, `frame_at_time`, `_srt_time` from `freekiki_openplay_video.py`.

## Rerun with the corrected ball + 16 mm replay (2026-11-02, ~04:00)

- The c1 ball was re-marked by hand (26/26 frames); new runs `passe/kick_analysis_20261102_033754` (c1) and
  `passe/kick_analysis_20261102_034754` (c2). Same people: tracks of 21 / 51 within 0.44 / 0.20 m of the old run.
- New numbers: passer → arrival 15.3 m, release (−0.016 s) → first touch 1.02 s, ball 76 / 33 / 51 km/h,
  51 at 17 km/h at release (run ≥ 0.9 s before, estimate), 51 moved 5.0 m.
- c2 is a slow replay (1.17–1.70×, ramped). `freekiki_pass_tv.sync_by_ball` uses the ball's along-path position as
  the clock: 37/39 frames, median 0.06 m, max 0.15 m. `match_ids`: 21 → c2 18 (0.56 m), 51 → c2 13 (0.67 m vs 1.54 m
  for 19, the marker). First touch confirmed in c2: frame 36 ↔ main 24.04.
- `kick_analysis` skipped the goal report for planar set pieces (`PLANAR_KINDS`), which used to mark pass runs failed.
- Delivery rebuilt in `passe/enviar/resultados_passe_tv_20261102`; the wrong-ball package moved to
  `passe/obsoleto/resultados_passe_tv_20261102_bola_errada`.

## Shot TV package review and rebuild (2026-11-02, ~04:30)

- Cursor's package (`chute/obsoleto/resultados_chute_tv_20261102_v1_cursor`) said "não dava para defender — só no
  tempo dos olhos (~120 ms)". The data contradict it: the keeper's right hand rises ≥ 0.30 m at 0.25 s (early 0.17 s)
  and at 0.46 s is 0.67 m from the ball (ball at 1.88 m, hand at 2.09 m); the "1.77 m" was ball → feet.
  The user chose the measured verdict: "Foi gol: faltou alcance, não reflexo" (gap rounded to ~70 cm for TV).
- The reconstructed ball crosses the goal line 0.24 m outside the near post (single-camera sideways error); the goal is
  from the footage. Stated in the limits section.
- Speed on air: "cerca de 90 km/h na saída do pé" in both packages (`freekiki_pass.speed_headline_kmh`: the launch
  92.3 km/h rounded to 5; the mean over the path, 77 km/h, is the second number). The old "pico 89 km/h" was the first
  full frame after the kick (the ball had already slowed); the per-frame series rose 0 → 64 → 89 only because the kick
  falls inside the first frame step. `monotone_flight_speed` makes the displayed speed fall from the kick on.
- New `vaila/freekiki_shot_tv.py` rebuilds the whole package (no hand-edited HTML, no debug logging, hashes written
  last). Delivered to `chute/enviar/resultados_chute_tv_20261102`; pass package rebuilt for the speed text.
