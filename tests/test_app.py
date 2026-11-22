from pathlib import Path

import pytest

from model import RELAYS

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest

APP = str(Path(__file__).resolve().parent.parent / "app.py")


@pytest.mark.parametrize("relay", list(RELAYS))
def test_app_runs_for_each_relay(relay):
    at = AppTest.from_file(APP, default_timeout=30)
    at.run()
    at.sidebar.selectbox[0].set_value(relay).run()
    assert not at.exception
    assert not at.error
    assert any("Fastest lineup" in h.value for h in at.subheader)


def test_app_compares_preferred_lineup():
    at = AppTest.from_file(APP, default_timeout=30)
    at.run()
    picks = ["Andre Lyons", "Marcus Hale", "Theo Okafor", "Devon Price"]
    pref_boxes = [sb for sb in at.selectbox if sb.label == "Your lineup"]
    for box, name in zip(pref_boxes, picks):
        box.set_value(name)
    at.run()
    assert not at.exception
    assert any("slower" in w.value for w in at.warning)
