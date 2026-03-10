from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llava_video_surveillance import (
    FrameClassification,
    PresenceEvent,
    detect_presence_events,
    format_timestamp,
    parse_person_present_from_text,
)


def make_classification(
    timestamp_seconds: float, person_present: bool
) -> FrameClassification:
    return FrameClassification(
        timestamp_seconds=timestamp_seconds,
        frame_index=int(timestamp_seconds),
        person_present=person_present,
        raw_text="",
        source="structured",
    )


def test_detect_presence_events_single_enter_and_exit() -> None:
    classifications = [
        make_classification(0.0, False),
        make_classification(2.0, False),
        make_classification(4.0, True),
        make_classification(6.0, True),
        make_classification(8.0, False),
    ]

    events, start_present, end_present = detect_presence_events(classifications)

    assert events == [PresenceEvent("enter", 4.0), PresenceEvent("exit", 8.0)]
    assert start_present is False
    assert end_present is False


def test_detect_presence_events_when_person_present_at_start_and_end() -> None:
    classifications = [
        make_classification(0.0, True),
        make_classification(2.0, True),
        make_classification(4.0, False),
        make_classification(6.0, True),
    ]

    events, start_present, end_present = detect_presence_events(classifications)

    assert events == [PresenceEvent("exit", 4.0), PresenceEvent("enter", 6.0)]
    assert start_present is True
    assert end_present is True


def test_parse_person_present_from_text_handles_strict_yes_no() -> None:
    assert parse_person_present_from_text("YES") is True
    assert parse_person_present_from_text("NO") is False
    assert parse_person_present_from_text("Yes, a person is visible.") is True
    assert parse_person_present_from_text("No person is visible.") is False
    assert parse_person_present_from_text("MAYBE") is None


def test_format_timestamp_rounds_to_nearest_second() -> None:
    assert format_timestamp(0.0) == "00:00:00"
    assert format_timestamp(61.2) == "00:01:01"
    assert format_timestamp(3661.6) == "01:01:02"
