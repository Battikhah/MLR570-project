"""Access-window guard for the AUS GPU workstation (ESB-1043-20).

Remote use is allowed from Thursday 18:00 to Monday 08:00, UAE time (UTC+4, no daylight saving),
and use outside it is monitored. The suite therefore refuses to start outside the window and stops
cleanly, after saving a checkpoint, early enough that the current epoch and the cloud sync finish
before Monday 08:00.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

UAE = timezone(timedelta(hours=4), "GST")
OPEN = (3, 18)  # Thursday (Monday = 0) 18:00
CLOSE = (0, 8)  # Monday 08:00


def window_close(now: datetime) -> datetime | None:
    """End of the access window containing ``now`` (timezone-aware), or None outside it."""
    t = now.astimezone(UAE)
    monday = (t - timedelta(days=t.weekday())).replace(hour=0, minute=0, second=0, microsecond=0)
    opened = monday + timedelta(days=OPEN[0], hours=OPEN[1])
    closes = monday + timedelta(days=7 + CLOSE[0], hours=CLOSE[1])  # the following Monday
    if t >= opened:
        return closes
    previous_close = monday + timedelta(hours=CLOSE[1])  # this week's Monday 08:00
    return previous_close if t < previous_close else None


class StopGuard:
    """``guard(next_epoch_seconds) -> True`` when there is no longer time for another epoch.

    Leaves ``margin`` before the window closes for the checkpoint to sync, and assumes the next
    epoch may take 1.5x the measured average.
    """

    def __init__(self, close: datetime | None, margin=timedelta(minutes=20), clock=None):
        self.close, self.margin = close, margin
        self.clock = clock or (lambda: datetime.now(UAE))

    def __call__(self, next_epoch_seconds: float = 0.0) -> bool:
        if self.close is None:
            return False
        needed = self.margin + timedelta(seconds=1.5 * next_epoch_seconds)
        return self.clock() + needed >= self.close

    def describe(self) -> str:
        if self.close is None:
            return "no access window enforced"
        return f"stopping before {(self.close - self.margin):%a %Y-%m-%d %H:%M} GST"
