import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import cv2

from output import write_summary
from video import read_video


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Read an MP4 video and write a JSON summary."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to the MP4 video, usually under data/input/.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/output/summary.json"),
        help="JSON output path (default: data/output/summary.json).",
    )
    arguments = parser.parse_args(argv)

    try:
        if arguments.output.resolve() == arguments.input.resolve():
            raise ValueError("Output path must not overwrite the input video.")
        summary = read_video(arguments.input)
        write_summary(summary, arguments.output)
    except (OSError, ValueError, cv2.error) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    print(f"Video summary written to: {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())