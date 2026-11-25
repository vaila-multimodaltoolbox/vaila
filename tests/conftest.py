"""
Shared pytest fixtures: keep every test away from the user's real AI Track files.

The AI Track network lives in a user-chosen folder (``VAILA_AI_MODELS_HOME``,
``~/.vaila/ai_models.json`` or the legacy ``vaila/models/ai_tracker``). Without isolation a
test that saves a discriminator would write into the real folder. This autouse fixture

* points ``VAILA_AI_MODELS_HOME`` at a fresh temporary folder (it is an environment
  variable, so subprocesses started by a test, such as the getpixelvideo window tests,
  inherit it);
* redirects ``ai_tracker_store.CONFIG_FILE`` to a temporary file and clears the session home;
* checks, once per session, that the real legacy folder, ``~/.vaila/ai_models.json`` and
  ``~/vaila_ai_models`` were not touched.

A test that needs a legacy folder patches ``ai_tracker_store.LEGACY_DIR`` itself.

Author: Paulo Roberto Pereira Santiago
Update Date: 24 November 2026
Version: 0.4.19
"""

from __future__ import annotations

from pathlib import Path

import pytest

from vaila import ai_tracker_store as store


def _snapshot() -> dict[str, object]:
    legacy = Path(store.LEGACY_DIR)
    files = (
        sorted((str(p.relative_to(legacy)), p.stat().st_size) for p in legacy.rglob("*"))
        if legacy.is_dir()
        else []
    )
    return {
        "legacy": files,
        "config": Path(store.CONFIG_FILE).exists(),
        "suggested": Path(store.SUGGESTED_HOME).exists(),
    }


@pytest.fixture(autouse=True)
def _isolate_ai_tracker_home(tmp_path_factory, monkeypatch):
    home = tmp_path_factory.mktemp("ai_models_home")
    monkeypatch.setenv(store.ENV_HOME, str(home))
    monkeypatch.setattr(store, "CONFIG_FILE", home.parent / f"{home.name}_cfg" / "ai_models.json")
    monkeypatch.setattr(store, "_session_home", None)
    store._SHA_CACHE.clear()
    yield home


@pytest.fixture(scope="session", autouse=True)
def _real_ai_tracker_files_untouched():
    before = _snapshot()
    yield
    after = _snapshot()
    assert after == before, (
        "a test changed the real AI Track folders (legacy folder, ~/.vaila/ai_models.json or "
        f"~/vaila_ai_models): before={before} after={after}"
    )
