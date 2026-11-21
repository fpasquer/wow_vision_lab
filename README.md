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