#!/usr/bin/env python3
"""
Add EXIF focal length to existing frames
Run this if you already extracted frames but forgot to add EXIF data
"""

import sys
from pathlib import Path
from tqdm import tqdm

def add_exif_focal_length(image_path, focal_length_mm):
    """Add focal length to image EXIF data using piexif"""
    try:
        import piexif
    except ImportError:
        print("Installing piexif for EXIF metadata...")
        import subprocess
        subprocess.run(["pip", "install", "piexif"], check=True, capture_output=True)
        import piexif

    from PIL import Image

    # Load image
    img = Image.open(image_path)

    # Try to load existing EXIF or create new
    try:
        exif_dict = piexif.load(str(image_path))
    except:
        exif_dict = {"0th": {}, "Exif": {}, "GPS": {}, "1st": {}, "thumbnail": None}

    # Add focal length to EXIF
    # FocalLength is stored as a rational (numerator, denominator)
    focal_length_rational = (int(focal_length_mm * 100), 100)  # e.g., 26.00mm = (2600, 100)
    exif_dict["Exif"][piexif.ExifIFD.FocalLength] = focal_length_rational

    # Also add FocalLengthIn35mmFilm for compatibility
    exif_dict["Exif"][piexif.ExifIFD.FocalLengthIn35mmFilm] = int(focal_length_mm)

    # Convert back to bytes
    exif_bytes = piexif.dump(exif_dict)

    # Save image with EXIF
    img.save(str(image_path), "jpeg", exif=exif_bytes, quality=95)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Add EXIF focal length metadata to existing frame images',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  %(prog)s 26
    Add 26mm focal length to frames in ./frames directory

  %(prog)s 35 /path/to/frames
    Add 35mm focal length to frames in custom directory

  %(prog)s 26 --sensor-width 7
    Add focal length with sensor width info (for reference)

Common Focal Lengths:
  24mm    - Wide angle / action cameras
  26-27mm - Most smartphones
  35mm    - Standard lens
  50mm    - Normal/portrait lens

Use Cases:
  - You extracted frames without EXIF metadata
  - You need to re-apply different focal length values
  - COLMAP requires focal length in EXIF for better results
        '''
    )

    parser.add_argument(
        'focal_length',
        type=float,
        metavar='FOCAL_LENGTH_MM',
        help='Focal length in millimeters (e.g., 26 for smartphone)'
    )

    parser.add_argument(
        'frames_dir',
        nargs='?',
        type=str,
        default=None,
        metavar='FRAMES_DIR',
        help='Directory containing frame_*.jpg files (default: ./frames)'
    )

    parser.add_argument(
        '--sensor-width',
        type=float,
        metavar='MM',
        help='Sensor width in millimeters (informational only, not written to EXIF)'
    )

    args = parser.parse_args()

    # Get frames directory
    if args.frames_dir:
        frames_dir = Path(args.frames_dir)
    else:
        # Use script's directory / frames
        frames_dir = Path(__file__).parent / "frames"

    if not frames_dir.exists():
        parser.error(f"Frames directory not found: {frames_dir}")

    # Find all frame images
    frames = sorted(frames_dir.glob("frame_*.jpg"))

    if len(frames) == 0:
        parser.error(f"No frame_*.jpg files found in {frames_dir}")

    print(f"Found {len(frames)} frames in {frames_dir}")
    print(f"Adding EXIF focal length: {args.focal_length}mm")
    if args.sensor_width:
        print(f"Sensor width (reference): {args.sensor_width}mm")

    # Process frames
    for frame in tqdm(frames, desc="Adding EXIF metadata"):
        add_exif_focal_length(frame, args.focal_length)

    print(f"\n✓ Successfully added EXIF focal length to {len(frames)} frames")


if __name__ == "__main__":
    main()
