# Splatlight Pipeline

A complete pipeline for converting videos into 3D Gaussian splats with proper world-space alignment.

## Overview

Splatlight processes video files to create aligned 3D Gaussian splats that can be loaded into viewers like MetalSplatter. The pipeline combines:

- **COLMAP**: Structure-from-Motion for camera pose estimation
- **SHARP**: Single-image to 3D Gaussian splat generation
- **Alignment**: Transforms splats from camera space to world space

## Features

- Automatic frame extraction from video with EXIF metadata
- Camera pose computation using COLMAP
- Per-frame Gaussian splat generation with SHARP
- World-space alignment of all splats
- Progress tracking and resumability (survives interruptions)
- Detailed failure reporting
- Manual focal length and sensor width configuration

## Requirements

### System Dependencies

- **Python 3.8+**
- **ffmpeg**: For video frame extraction
- **COLMAP**: For camera pose estimation
- **SHARP**: For Gaussian splat generation

### Python Dependencies

The pipeline automatically installs required Python packages:
- `tqdm`: Progress bars
- `piexif`: EXIF metadata handling
- `pycolmap`: COLMAP Python bindings
- `plyfile`: PLY file reading/writing
- `scipy`: Spatial transformations
- `Pillow`: Image processing

## Installation

### 1. Install System Dependencies

**macOS** (using Homebrew):
```bash
brew install ffmpeg colmap
```

**Linux** (Ubuntu/Debian):
```bash
sudo apt install ffmpeg colmap
```

### 2. Install SHARP

Follow the SHARP installation instructions from Apple:
```bash
# Install SHARP (see Apple's documentation)
pip install sharp
```

### 3. Clone/Download this Pipeline

```bash
git clone <repository-url>
cd splatlight
```

## Usage

### Quick Start

Place a video file (`.MOV`, `.mp4`, etc.) in the splatlight directory and run:

```bash
python pipeline.py
```

The pipeline will:
1. Extract frames from the video
2. Ask for focal length (or detect it automatically)
3. Compute camera poses with COLMAP
4. Generate splats with SHARP
5. Align splats to world space

### Command Reference

#### Run Pipeline

```bash
# Run with defaults (interactive focal length prompt)
python pipeline.py

# Specify focal length (35mm)
python pipeline.py run 35

# Specify focal length and sensor width
python pipeline.py run 35 36

# Use custom SHARP checkpoint
python pipeline.py run 35 36 /path/to/checkpoint.pt

# Run from a different directory
python pipeline.py /path/to/video/dir run 35
```

#### Check Status

```bash
python pipeline.py status
```

Shows which steps have been completed.

#### Reset Progress

```bash
# Reset all progress
python pipeline.py reset

# Reset from a specific step onwards
python pipeline.py reset camera_poses

# Reset but preserve SHARP frame progress
python pipeline.py reset camera_poses --keep-sharp
```

Available steps:
- `extract_frames`
- `camera_poses`
- `generate_splats`
- `align_splats`

### Parameters

#### Focal Length (`focal_length_mm`)

The focal length of the camera in millimeters. Critical for accurate camera pose estimation.

Common values:
- **24mm**: Wide angle / action cameras
- **26-27mm**: Most smartphones (iPhone/Android)
- **35mm**: Standard lens
- **50mm**: Normal/portrait lens

If not provided, the pipeline attempts to extract it from video metadata or prompts you interactively.

#### Sensor Width (`sensor_width_mm`)

The physical width of the camera sensor in millimeters. Used to convert focal length to pixels.

Common values:
- **36mm**: Full-frame DSLR/mirrorless (default)
- **23.8mm**: APS-C sensor
- **17.3mm**: Micro 4/3 (Panasonic)
- **7mm**: iPhone/Android smartphone main camera
- **6mm**: Ultra-wide smartphone

Default is 36mm (full-frame).

#### SHARP Checkpoint (`checkpoint_path`)

Path to a custom SHARP model checkpoint (`.pt` file). If not provided, the pipeline automatically downloads and caches the default model to `~/.cache/sharp/`.

## Pipeline Steps

### Step 1: Extract Frames

Extracts all frames from the input video as high-quality JPEG images and optionally adds EXIF focal length metadata.

**Output**: `frames/frame_0001.jpg`, `frames/frame_0002.jpg`, ...

### Step 2: Compute Camera Poses

Uses COLMAP to:
1. Extract features from frames
2. Match features between frames
3. Build sparse 3D reconstruction
4. Export camera poses as `transforms.json`

**Output**: `colmap/transforms.json`, `colmap/sparse/`

### Step 3: Generate Splats

Runs SHARP on each frame to generate per-frame Gaussian splat PLY files.

**Output**: `ply/frame_0001.ply`, `ply/frame_0002.ply`, ...

### Step 4: Align Splats

Transforms each splat from camera space to world space using the COLMAP camera poses. This aligns all splats into a consistent world coordinate system.

**Output**: `ply/aligned/frame_0001.ply`, `ply/aligned/frame_0002.ply`, ...

## Output Files

```
splatlight/
├── video.MOV                   # Input video
├── frames/                     # Extracted frames
│   ├── frame_0001.jpg
│   ├── frame_0002.jpg
│   └── ...
├── colmap/                     # COLMAP output
│   ├── database.db            # Feature database
│   ├── sparse/                # Sparse reconstruction
│   └── transforms.json        # Camera poses
├── ply/                        # Generated splats
│   ├── frame_0001.ply
│   ├── frame_0002.ply
│   ├── ...
│   └── aligned/               # World-aligned splats (FINAL OUTPUT)
│       ├── frame_0001.ply
│       ├── frame_0002.ply
│       └── ...
├── progress.txt               # Pipeline progress tracker
└── sharp_progress.json        # Per-frame SHARP progress
```

**Final output**: `ply/aligned/*.ply` - Load these into MetalSplatter or your renderer.

## Progress Tracking & Resumability

The pipeline saves progress after each step:

- **progress.txt**: Tracks completed pipeline steps
- **sharp_progress.json**: Tracks which frames SHARP has processed

If interrupted (Ctrl+C, crash, etc.), simply run the pipeline again. It will resume from where it left off.

### Preserving SHARP Progress

SHARP processing can take a long time. Use `--keep-sharp` when resetting:

```bash
# Re-run COLMAP but keep all SHARP progress
python pipeline.py reset camera_poses --keep-sharp
```

This is useful when:
- COLMAP failed but SHARP succeeded
- You want to adjust camera parameters without re-processing frames
- Experimenting with alignment without re-running SHARP

## Troubleshooting

### COLMAP Registration Failures

If COLMAP fails to register some frames:

**Symptoms**: Alignment step reports "no camera pose - COLMAP failed to register"

**Solutions**:
- Ensure video has sufficient motion and texture
- Avoid blurry frames or fast motion
- Verify focal length setting is correct
- Consider extracting fewer frames (modify ffmpeg command in code)
- Increase COLMAP's `--SequentialMatching.overlap` parameter

### SHARP Failures

If SHARP fails on specific frames:

**Symptoms**: Processing step shows "✗" failures

**Solutions**:
- Check frame quality (not too blurry, proper exposure)
- Verify SHARP installation: `sharp --version`
- Ensure MPS (Metal Performance Shaders) is available on macOS
- Try with CPU: Modify `--device mps` to `--device cpu` in code (pipeline.py:681)

### Missing Focal Length Metadata

If the pipeline can't extract focal length from video:

**Solution**: Manually provide it as a command argument:
```bash
python pipeline.py run 26  # for smartphone video
```

Check your camera specs or use a common default (26mm for phones, 35mm for cameras).

## Utility Scripts

### Add EXIF to Existing Frames

If you already extracted frames but forgot to add focal length metadata:

```bash
python add_exif_to_frames.py 26
python add_exif_to_frames.py 26 /path/to/frames
```

This adds EXIF focal length to existing frame images.

## Examples

### Smartphone Video (iPhone)

```bash
python pipeline.py run 26 7
```

### Action Camera (GoPro)

```bash
python pipeline.py run 24 6
```

### Full-Frame Camera (35mm lens)

```bash
python pipeline.py run 35 36
```

### Process and Reset

```bash
# Initial run
python pipeline.py

# COLMAP failed, try different focal length
python pipeline.py reset camera_poses --keep-sharp
python pipeline.py run 28
```

## Advanced Usage

### Custom Working Directory

```bash
python pipeline.py /path/to/video/directory run 35
```

### Custom SHARP Model

```bash
python pipeline.py run 35 36 /path/to/custom_checkpoint.pt
```

## Next Steps

After running the pipeline:

1. **Load into MetalSplatter**: Open the aligned PLY files
2. **Render with Metal**: Use the PLY files in your own Metal renderer
3. **Combine splats**: Merge multiple PLY files if needed
4. **Export/compress**: Convert to optimized formats for web viewing

## License

See project license file.

## Credits

- **COLMAP**: Structure-from-Motion system
- **SHARP**: Apple's single-image Gaussian splat generator
- **MetalSplatter**: 3D Gaussian splat viewer for macOS

## Support

For issues, questions, or contributions, please open an issue on the project repository.
