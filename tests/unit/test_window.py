"""Access window of the AUS workstation: Thursday 18:00 to Monday 08:00, UAE time (UTC+4)."""

import importlib.util
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from pcdkd.experiments.window import UAE, StopGuard, window_close

# 2026-11-12 is a Thursday; 2026-11-16 the following Monday.
THU, MON = datetime(2026, 11, 12, tzinfo=UAE), datetime(2026, 11, 16, tzinfo=UAE)
CLOSE = MON.replace(hour=8)


@pytest.mark.parametrize(
    ("now", "expected"),
    [
        (THU.replace(hour=17, minute=59), None),
        (THU.replace(hour=18), CLOSE),
        (THU + timedelta(days=2, hours=3), CLOSE),  # Saturday 03:00
        (MON.replace(hour=7, minute=59), CLOSE),
        (MON.replace(hour=8), None),
        (MON + timedelta(days=2, hours=12), None),  # Wednesday noon
    ],
)
def test_window_close(now, expected):
    assert window_close(now) == expected


def test_window_uses_uae_time_whatever_the_machine_zone():
    utc_friday_noon = datetime(2026, 11, 13, 8, tzinfo=UTC)  # 12:00 in the UAE
    assert window_close(utc_friday_noon) == CLOSE
    utc_thursday = datetime(2026, 11, 12, 13, 59, tzinfo=UTC)  # 17:59 in the UAE
    assert window_close(utc_thursday) is None


def test_guard_leaves_margin_and_room_for_the_next_epoch():
    now = CLOSE - timedelta(minutes=30)
    guard = StopGuard(CLOSE, timedelta(minutes=20), clock=lambda: now)
    assert not guard(0)  # 30 min left > 20 min margin
    assert not guard(6 * 60)  # 20 + 9 = 29 min needed
    assert guard(8 * 60)  # 20 + 12 = 32 min needed
    assert not StopGuard(None)(10**9)


def _run_suite():
    path = Path(__file__).resolve().parents[2] / "scripts" / "run_suite.py"
    spec = importlib.util.spec_from_file_location("run_suite", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_cli_refuses_outside_window_and_guards_inside():
    rs = _run_suite()
    assert rs.make_guard("aus", 20, now=MON + timedelta(days=1)) is None
    inside = rs.make_guard("aus", 20, now=THU.replace(hour=19))
    assert inside.close == CLOSE and inside.margin == timedelta(minutes=20)
    assert rs.make_guard("none", 20).close is None


def test_cli_run_outside_window_exits_without_training(monkeypatch, capsys):
    rs = _run_suite()
    monkeypatch.setattr(rs, "make_guard", lambda *a, **k: None)
    monkeypatch.setattr(rs, "run_many", lambda *a, **k: pytest.fail("must not train"))
    assert rs.main(["run", "--profile", "smoke", "--window", "aus", "--ids", "pilot"]) == 2
    assert "outside the workstation access window" in capsys.readouterr().err
