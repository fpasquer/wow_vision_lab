# WoW Vision Lab

WoW Vision Lab is a local computer vision learning project focused on detecting objects in recorded World of Warcraft gameplay videos.

## V1 Scope

- Process a local MP4 video.
- Detect objects using YOLO11n with Ultralytics and PyTorch.
- Run inference on CPU inside Docker Compose.
- Produce an annotated MP4 video and JSON detection results.
- Detect one frame every configurable number of decoded frames (default: 5).
- Use a configurable detection confidence threshold (default: 0.25).

## Constraints

- CPU only, with a limit of 2 CPUs and 2 GB RAM.
- No GPU requirement.
- No network access during processing.
- One on-demand processing container.
- No unnecessary architectural complexity.

## Development

The developer owns architecture and requirements. AI tools assist with implementation under explicit developer supervision.

## Status

Sequential MP4 reading, JSON summaries, and T05 CPU object detection are
implemented. Detection is opt-in. Existing summary-only commands and their
output format remain unchanged.

## Docker Setup

Install Docker Engine with Docker Compose. Building the image requires network
access to download Python and the pinned dependencies:

- Python 3.15.2 (`python:3.15.2-slim-trixie`).
- `numpy==2.5.5`.
- `opencv-python-headless==5.1.0.95`.
- `torch==2.14.1+cpu`.
- `torchvision==0.29.1+cpu`.
- `ultralytics-opencv-headless==8.3.253`.

CPU-only PyTorch and torchvision wheels come from the official PyTorch CPU
index. The headless Ultralytics distribution provides the `ultralytics` import
without installing a conflicting GUI version of OpenCV. Its required runtime
dependencies are installed, but no optional extras or CUDA packages are added.

The image creates a non-root `app` account using `HOST_UID` and `HOST_GID`.
Both default to 1000. Compose passes the same IDs at build time and runtime.
Set them from your non-root host account before building so the `app` account
matches your host and can write to `data/output`.

Run commands from the project root:

```sh
export HOST_UID="$(id -u)"
export HOST_GID="$(id -g)"
docker compose config
docker compose build
docker compose run --rm app python -m pip check
docker compose run --rm app python --version
docker compose run --rm app python -c "import cv2, numpy; print(cv2.__version__, numpy.__version__)"
```

Rebuild the image after changing these IDs. Use a non-root host account; the
build rejects `HOST_UID=0`. The runtime user must have read access to
`data/input` and `data/models`, and write access to `data/output`.

T03 runs one on-demand container with these restrictions:

- Maximum 2 CPUs and 2 GB RAM.
- No runtime network access and no exposed ports.
- Read-only root filesystem, with a temporary `/tmp` limited to 128 MB.
- All Linux capabilities dropped and privilege escalation disabled.
- `data/input` mounted read-only at `/app/data/input`.
- `data/models` mounted read-only at `/app/data/models`.
- `data/output` mounted read-write at `/app/data/output`.

Use `docker compose run --rm app` for the default Python version check. It exits
after printing the version, and Compose removes the container. Override the
command after `app` for processing or other one-shot checks. There is no
continuously running service.

## MP4 Video Reading

Place an MP4 video in `data/input/`. Rebuild the image after changing source
files, then run the CLI:

```sh
docker compose build
docker compose run --rm app python -m main --input data/input/sample.mp4
```

The default output is `data/output/summary.json`. Set a different output path
with `--output`:

```sh
docker compose run --rm app python -m main \
  --input data/input/sample.mp4 \
  --output data/output/sample-summary.json
```

The output directory is created if necessary. A successful summary contains:

```json
{
  "video": "sample.mp4",
  "width": 1920,
  "height": 1080,
  "fps": 30.0,
  "frames_read": 1800,
  "frames_expected": 1800,
  "duration_seconds": 60.0,
  "status": "completed"
}
```

`video` is the input filename, without its directory. Frames are decoded
sequentially and are not collected in memory. `frames_read` counts frames
actually decoded, not the container's reported frame count. Duration is
estimated as `frames_read / fps`.

`frames_expected` comes from OpenCV's frame-count metadata. It is used only
when finite, positive, and integer-valued; otherwise it is `null`. Unknown
counts do not trigger an incomplete-reading warning.

If reading stops before a valid expected count, a summary is still written:

```json
{
  "video": "sample.mp4",
  "width": 1920,
  "height": 1080,
  "fps": 30.0,
  "frames_read": 850,
  "frames_expected": 1800,
  "duration_seconds": 28.333333333333332,
  "status": "incomplete",
  "warning": "Video reading stopped before the expected frame count."
}
```

The CLI prints the warning to standard error. Successfully writing a summary
exits with code 0 for both `completed` and `incomplete` results. Completed
results omit `warning`, including when the expected count is unavailable or
lower than the decoded count.

Frame-count metadata may be inaccurate. An incomplete result indicates a
potentially incomplete reading, not proof of corruption. Unknown or
underreported counts cannot reliably reveal truncation. No automatic recovery
or frame skipping is attempted.

Missing files, unreadable videos, invalid dimensions or FPS, and videos without
readable frames produce a clear error on standard error and exit with code 1.
Zero readable frames remain an error even when the expected count is valid. Invalid
FPS values (zero, negative, or non-finite) are rejected without a fallback.
Input errors do not write a summary. Output errors also exit with code 1.
The video capture is always released, including after errors. The output path
cannot overwrite the input video.

## Object Detection

Place trusted pretrained YOLO11n weights at `data/models/yolo11n.pt`. The model
is not included in the image. Obtain the weights before processing; neither the
application nor the container downloads weights at runtime. The existing
`data/models` mount is read-only, and the application does not modify weights.
Use only trusted `.pt` files because PyTorch checkpoints can contain executable
Python objects.

Rebuild after changing source files, then enable detection with `--model`:

```sh
docker compose build
docker compose run --rm app python -m main \
  --input data/input/sample.mp4 \
  --model \
  --output data/output/sample-summary.json
```

A bare `--model` uses `data/models/yolo11n.pt`. Override the local weights path,
confidence, or sampling step if needed:

```sh
docker compose run --rm app python -m main \
  --input data/input/sample.mp4 \
  --model data/models/yolo11n.pt \
  --confidence 0.5 \
  --frame-step 10 \
  --output data/output/sample-summary.json
```

Without `--model`, the CLI only writes the T04 summary and does not load
Ultralytics or require weights. Detection options have these defaults:

- `--confidence 0.25`: a finite number from 0 to 1.
- `--frame-step 5`: a positive integer.

Frame indexes are zero-based. The default step detects frames 0, 5, 10, and so
on, not 5 frames per second. The first decoded frame is always processed, even
when the video contains fewer frames than the step.

Detection mode requires all output paths to stay under `data/output`. Existing
summary-only custom output paths remain supported. For the first command above,
a completed run produces:

- `data/output/sample-summary.json`: the unchanged T04 summary.
- `data/output/sample-summary-detections.json`: detection metadata.
- `data/output/sample-summary-annotated.mp4`: every decoded frame at the
  original dimensions and FPS.

Only sampled frames receive boxes and class/confidence labels. Other frames
are kept without annotations; detections are not reused or tracked. The MP4
uses OpenCV's `mp4v` writer and does not preserve audio.

Detection JSON contains the selected weights filename and one record per
sampled frame, including frames without detections:

```json
{
  "model": "yolo11n.pt",
  "frames": [
    {
      "frame_index": 0,
      "timestamp_seconds": 0.0,
      "detections": [
        {
          "class_id": 0,
          "class_name": "person",
          "confidence": 0.82,
          "bbox": [10.0, 20.0, 100.0, 120.0]
        }
      ]
    },
    {
      "frame_index": 5,
      "timestamp_seconds": 0.16666666666666666,
      "detections": []
    }
  ]
}
```

`timestamp_seconds` estimates how far into the source video a frame appears.
As in T04, divide the frame's zero-based index in the source video
(`frame_index`) by the source video's frames per second (`source_fps`). For
example, at 30 fps, frame 0 is at 0 seconds and frame 5 is at `5 / 30`, or
about 0.167 seconds. Skipping frames does not change their source indices.
This calculation assumes frames are evenly spaced, so it does not recover
the actual timestamps in a variable-frame-rate video. Bounding boxes are
`[x1, y1, x2, y2]` in original-frame pixels, from top-left to bottom-right.

Frames and detection records are processed sequentially without collecting
the entire video in memory. Detection JSON and MP4 are staged under
`data/output` and published only after completed processing and a successful
summary write. Failures remove unfinished detection files and exit with code 1.
Missing, unreadable, corrupt, or non-detection model files produce clear errors
without downloading or falling back to another model.

Incomplete reading retains the existing T04 summary, stderr warning, and exit
code 0, but does not publish new detection JSON or MP4 files. Any detection
files from earlier runs remain unchanged.

Ultralytics online checks and automatic dependency installation are disabled.
Library settings and caches are stored under `data/output/.cache`. Inference
always uses CPU, and all existing Docker security and resource limits remain
in place.

Pretrained YOLO11n recognizes general COCO classes, not WoW-specific classes.
Its accuracy on gameplay is not guaranteed. Training, tracking, GPU inference,
and live capture are not included.

## Tests

Tests use standard-library `unittest` and a small MP4 generated with OpenCV.
Mocked captures cover incomplete reading, unavailable or invalid frame counts,
JSON output, stderr warnings, and resource cleanup. Mocked model results cover
CPU/confidence arguments, detection metadata, sampling, annotations, output
validation, and cleanup after failures. The original T04 tests remain intact.
No additional dependencies are needed. Mount the tests read-only because they
are intentionally excluded from the application image:

```sh
docker compose build
docker compose run --rm \
  -v "$(pwd)/tests:/app/tests:ro" \
  app python -m unittest discover -s tests -v
```

Real-weight tests are skipped in the default unit command. Run them separately
after placing the weights in `data/models/yolo11n.pt`:

```sh
docker compose run --rm \
  -e RUN_INTEGRATION_TESTS=1 \
  -v "$(pwd)/tests:/app/tests:ro" \
  app python -m unittest discover -s tests \
    -p test_detector_integration.py -v
```

When explicitly enabled, missing or invalid weights fail the integration tests
instead of skipping them. Integration tests run real CPU inference, verify
that weights are unchanged, check JSON sampling, and reopen the annotated MP4
to check every frame, dimensions, and FPS.

Test fixtures are created in temporary directories and removed after each test.
Detection test outputs use scoped temporary directories under `data/output`.
Library caches may remain under `data/output/.cache`.
The command keeps the T03 user, network, filesystem, CPU, memory, and security
restrictions.