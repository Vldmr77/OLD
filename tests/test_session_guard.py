from datetime import datetime

from scalp_system.config.base import SessionScheduleConfig
from scalp_system.control.session import SessionGuard


def test_session_guard_disabled_is_always_active():
    guard = SessionGuard(SessionScheduleConfig(enabled=False))

    state = guard.evaluate(datetime(2024, 1, 1, 9, 0, 0))

    assert state.active is True
    assert state.reason is None
    assert state.changed is True


def test_session_guard_pre_open_reports_resume_time():
    config = SessionScheduleConfig(
        enabled=True,
        start_time="10:00",
        end_time="18:45",
        pre_open_minutes=15,
        post_close_minutes=10,
        allowed_weekdays=[0, 1, 2, 3, 4],
    )
    config.ensure()
    guard = SessionGuard(config)

    now = datetime(2024, 1, 2, 9, 30, 0)
    state = guard.evaluate(now)

    assert state.active is False
    assert state.reason == "pre_open"
    assert state.next_transition is not None
    assert state.next_transition.hour == 9 and state.next_transition.minute == 45

    repeat = guard.evaluate(now)
    assert repeat.changed is False


def test_session_guard_handles_post_close_and_weekend():
    config = SessionScheduleConfig(
        enabled=True,
        start_time="10:00",
        end_time="18:30",
        pre_open_minutes=10,
        post_close_minutes=5,
        allowed_weekdays=[0, 1, 2, 3, 4],
    )
    config.ensure()
    guard = SessionGuard(config)

    after_close = datetime(2024, 1, 2, 20, 0, 0)
    state = guard.evaluate(after_close)
    assert state.active is False
    assert state.reason == "post_close"
    assert state.next_transition is not None

    weekend = datetime(2024, 1, 6, 12, 0, 0)
    state_weekend = guard.evaluate(weekend)
    assert state_weekend.reason == "weekday_blocked"
    assert state_weekend.active is False
    assert state_weekend.next_transition is not None
