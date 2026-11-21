import json
from pathlib import Path

from video import VideoSummary


def write_summary(summary: VideoSummary, output_path: Path) -> None:
    content = json.dumps(summary, indent=2, allow_nan=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(f"{content}\n", encoding="utf-8")