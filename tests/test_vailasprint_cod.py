"""COD 180° labelling and left/right comparison helpers."""

from vaila import vailasprint


def test_format_trial_label_cod_sides():
    assert vailasprint.format_trial_label("cod", 1) == "COD Left (01)"
    assert vailasprint.format_trial_label("cod", 2) == "COD Right (02)"
    assert vailasprint.format_trial_label("cod", 1, "pt") == "COD Esquerda (01)"
    assert vailasprint.format_trial_label("cod", 2, "pt") == "COD Direita (02)"


def test_cod_left_right_comparison_pairs_athletes():
    run_stats = [
        {
            "athlete_name": "Allan",
            "run_id": 1,
            "total_time_s": 5.1,
            "max_speed_kmh": 28.0,
            "report_path": "a_left.html",
        },
        {
            "athlete_name": "Allan",
            "run_id": 2,
            "total_time_s": 4.9,
            "max_speed_kmh": 29.0,
            "report_path": "a_right.html",
        },
    ]
    import pandas as pd

    html = vailasprint.generate_cod_left_right_comparison_html(pd.DataFrame(run_stats))
    assert "COD Left (01)" in html
    assert "COD Right (02)" in html
    assert "Allan" in html
    assert "COD Direita (02)" not in html
