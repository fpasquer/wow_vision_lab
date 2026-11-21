# WoW Vision Lab

WoW Vision Lab is a local computer vision learning project focused on detecting objects in recorded World of Warcraft gameplay videos.

## V1 Scope

- Process a local MP4 video.
- Detect objects using YOLO11n with Ultralytics and PyTorch.
- Run inference on CPU inside Docker Compose.
- Produce an annotated MP4 video and JSON detection results.
- Use a configurable frame processing rate (default: 5 FPS).
- Use a configurable detection confidence threshold (default: 0.25).

## Constraints

- CPU only, with a limit of 2 CPUs and 4 GB RAM.
- No GPU requirement.
- No network access during processing.
- One on-demand processing container.
- No unnecessary architectural complexity.

## Development

The developer owns architecture and requirements. AI tools assist with implementation under explicit developer supervision.

## Status

Repository initialization. Application implementation has not started.

## Docker Setup

Install Docker Engine with Docker Compose. Building the image requires network
access to download Python and the pinned dependencies:

- Python 3.15.2 (`python:3.15.2-slim-trixie`).
- `numpy==2.5.5`.
- `opencv-python-headless==5.1.0.95`.

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
command after `app` for other one-shot checks. No continuously running service
or image-processing functionality is included.