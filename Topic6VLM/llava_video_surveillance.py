from __future__ import annotations

import argparse
import base64
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from typing import cast

import cv2
from langchain_core.messages import AIMessage, HumanMessage, content as msg_content
from pydantic import BaseModel

from agent.ollama_client import OllamaSettings, get_chatollama
from util.imgUtils import scale_to_max_dimension


class PersonPresence(BaseModel):
    person_present: bool


@dataclass(frozen=True)
class FrameSample:
    timestamp_seconds: float
    frame_index: int
    image_b64: str


@dataclass(frozen=True)
class FrameClassification:
    timestamp_seconds: float
    frame_index: int
    person_present: bool
    raw_text: str
    source: Literal["structured", "raw_fallback"]


@dataclass(frozen=True)
class PresenceEvent:
    event_type: Literal["enter", "exit"]
    timestamp_seconds: float


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze a surveillance-style video with LLaVA and report person enter/exit times."
    )
    parser.add_argument("video_path", help="Path to the video file to analyze.")
    parser.add_argument(
        "--sample-seconds",
        type=float,
        default=2.0,
        help="Seconds between sampled frames (default: 2.0).",
    )
    parser.add_argument(
        "--max-dimension",
        type=int,
        default=720,
        help="Maximum frame width/height before downscaling (default: 720).",
    )
    parser.add_argument(
        "--model",
        default="llava",
        help="Ollama model name to use (default: llava).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print one classification line per sampled frame.",
    )
    return parser


def encode_frame_to_base64(frame: cv2.typing.MatLike) -> str:
    ok, buffer = cv2.imencode(".jpg", frame)
    if not ok:
        raise ValueError("Frame could not be JPEG-encoded.")
    return base64.b64encode(buffer.tobytes()).decode("ascii")


def extract_sampled_frames(
    video_path: Path,
    sample_every_seconds: float = 2.0,
    max_dimension: int = 720,
) -> list[FrameSample]:
    if sample_every_seconds <= 0:
        raise ValueError("sample_every_seconds must be greater than 0.")
    if max_dimension <= 0:
        raise ValueError("max_dimension must be greater than 0.")

    resolved_path = video_path.resolve()
    if not resolved_path.is_file():
        raise FileNotFoundError(f"Video file not found: {resolved_path}")

    cap = cv2.VideoCapture(str(resolved_path))
    if not cap.isOpened():
        cap.release()
        raise ValueError(f"Could not open video: {resolved_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        cap.release()
        raise ValueError(f"Video FPS is invalid for {resolved_path}: {fps}")

    sample_interval_frames = max(1, int(round(fps * sample_every_seconds)))
    samples: list[FrameSample] = []
    frame_index = 0

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        if frame_index % sample_interval_frames == 0:
            resized = scale_to_max_dimension(frame, max_dimension)
            samples.append(
                FrameSample(
                    timestamp_seconds=frame_index / fps,
                    frame_index=frame_index,
                    image_b64=encode_frame_to_base64(resized),
                )
            )
        frame_index += 1

    cap.release()

    if not samples:
        raise ValueError(f"No frames could be sampled from {resolved_path}")

    return samples


def build_presence_message(image_b64: str) -> HumanMessage:
    blocks = [
        msg_content.create_text_block(
            "Inspect this surveillance frame and determine whether at least one real person is visible anywhere in the scene. Ignore posters, screens, reflections, drawings, and statues."
        ),
        msg_content.create_image_block(base64=image_b64, mime_type="image/jpeg"),
    ]
    return HumanMessage(content_blocks=blocks)


def build_raw_fallback_message(image_b64: str) -> HumanMessage:
    blocks = [
        msg_content.create_text_block(
            "Answer with exactly YES or NO. Is at least one real person visible anywhere in this surveillance frame? Ignore posters, screens, reflections, drawings, and statues."
        ),
        msg_content.create_image_block(base64=image_b64, mime_type="image/jpeg"),
    ]
    return HumanMessage(content_blocks=blocks)


def extract_ai_text(message: AIMessage) -> str:
    if isinstance(message.content, str):
        return message.content.strip()
    parts: list[str] = []
    for item in message.content:
        if isinstance(item, str):
            parts.append(item)
            continue
        if item.get("type") == "text" and isinstance(item.get("text"), str):
            parts.append(item["text"])
    return "\n".join(part.strip() for part in parts if part.strip())


def parse_person_present_from_text(text: str) -> bool | None:
    normalized = text.strip().upper()
    if not normalized:
        return None
    if normalized == "YES":
        return True
    if normalized == "NO":
        return False
    if normalized.startswith("YES"):
        return True
    if normalized.startswith("NO"):
        return False
    return None


def classify_sample(
    sample: FrameSample, settings: OllamaSettings
) -> FrameClassification:
    chat_model = get_chatollama(settings)
    structured_model = chat_model.with_structured_output(
        PersonPresence,
        method="json_schema",
        include_raw=True,
    )

    response = cast(
        dict[str, object],
        structured_model.invoke([build_presence_message(sample.image_b64)]),
    )
    parsed = response.get("parsed")
    raw_message = response.get("raw")
    raw_text = (
        extract_ai_text(raw_message) if isinstance(raw_message, AIMessage) else ""
    )

    if isinstance(parsed, PersonPresence):
        return FrameClassification(
            timestamp_seconds=sample.timestamp_seconds,
            frame_index=sample.frame_index,
            person_present=parsed.person_present,
            raw_text=raw_text,
            source="structured",
        )

    fallback_message = chat_model.invoke([build_raw_fallback_message(sample.image_b64)])
    fallback_text = extract_ai_text(fallback_message)
    fallback_value = parse_person_present_from_text(fallback_text)
    if fallback_value is None:
        raise ValueError(
            f"Model response could not be parsed as person presence. Raw response: {fallback_text or raw_text}"
        )
    return FrameClassification(
        timestamp_seconds=sample.timestamp_seconds,
        frame_index=sample.frame_index,
        person_present=fallback_value,
        raw_text=fallback_text,
        source="raw_fallback",
    )


def detect_presence_events(
    classifications: list[FrameClassification],
) -> tuple[list[PresenceEvent], bool, bool]:
    if not classifications:
        return [], False, False

    events: list[PresenceEvent] = []
    start_present = classifications[0].person_present

    for previous, current in zip(classifications, classifications[1:]):
        if not previous.person_present and current.person_present:
            events.append(PresenceEvent("enter", current.timestamp_seconds))
        elif previous.person_present and not current.person_present:
            events.append(PresenceEvent("exit", current.timestamp_seconds))

    end_present = classifications[-1].person_present
    return events, start_present, end_present


def format_timestamp(seconds: float) -> str:
    total_seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def analyze_video(
    video_path: Path,
    sample_every_seconds: float,
    max_dimension: int,
    settings: OllamaSettings,
) -> tuple[list[FrameClassification], list[PresenceEvent], bool, bool]:
    samples = extract_sampled_frames(
        video_path=video_path,
        sample_every_seconds=sample_every_seconds,
        max_dimension=max_dimension,
    )
    classifications = [classify_sample(sample, settings) for sample in samples]
    events, start_present, end_present = detect_presence_events(classifications)
    return classifications, events, start_present, end_present


def print_report(
    video_path: Path,
    classifications: list[FrameClassification],
    events: list[PresenceEvent],
    start_present: bool,
    end_present: bool,
    verbose: bool,
) -> None:
    print(f"Video: {video_path.resolve()}")
    print(f"Sampled frames analyzed: {len(classifications)}")
    if classifications:
        print(
            f"Analyzed time range: {format_timestamp(classifications[0].timestamp_seconds)} to {format_timestamp(classifications[-1].timestamp_seconds)}"
        )

    if verbose:
        print("\nPer-frame results:")
        for item in classifications:
            verdict = "present" if item.person_present else "absent"
            print(
                f"- {format_timestamp(item.timestamp_seconds)} | frame {item.frame_index} | {verdict} | {item.source} | {item.raw_text or '(no raw text)'}"
            )

    print("\nDetected events:")
    if start_present:
        print(
            f"- Person already present in the first sampled frame at {format_timestamp(classifications[0].timestamp_seconds)}"
        )
    if events:
        for event in events:
            print(
                f"- {event.event_type} at {format_timestamp(event.timestamp_seconds)}"
            )
    else:
        print("- No enter/exit transitions detected in sampled frames")
    if end_present:
        print(
            f"- Person still present in the last sampled frame at {format_timestamp(classifications[-1].timestamp_seconds)}"
        )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    settings = OllamaSettings(model=args.model)
    video_path = Path(args.video_path)

    classifications, events, start_present, end_present = analyze_video(
        video_path=video_path,
        sample_every_seconds=args.sample_seconds,
        max_dimension=args.max_dimension,
        settings=settings,
    )
    print_report(
        video_path=video_path,
        classifications=classifications,
        events=events,
        start_present=start_present,
        end_present=end_present,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
