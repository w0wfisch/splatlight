#!/usr/bin/env python3
"""
Splat Flashlight Pipeline with Progress Tracking
Resumes from last successful step if interrupted
"""

import os
import subprocess
import json
from pathlib import Path
from tqdm import tqdm
import shutil
import sys

class SplatPipeline:
    def __init__(self, base_dir=None):
        # If no base_dir provided, use the script's directory
        if base_dir is None:
            base_dir = Path(__file__).parent.resolve()
        self.base_dir = Path(base_dir)

        # Input paths - look for any .MOV or .mp4 video in the script's directory
        self.video_path = self._find_video()
        self.frames_dir = self.base_dir / "frames"

        # Output paths
        self.colmap_dir = self.base_dir / "colmap"
        self.ply_dir = self.base_dir / "ply"
        self.aligned_ply_dir = self.base_dir / "ply/aligned"

        # Progress tracking
        self.progress_file = self.base_dir / "progress.txt"
        self.sharp_progress_file = self.base_dir / "sharp_progress.json"
        self.failed_frames = []  # Track failed frames

        # Create directories
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.colmap_dir.mkdir(parents=True, exist_ok=True)
        self.ply_dir.mkdir(parents=True, exist_ok=True)
        self.aligned_ply_dir.mkdir(parents=True, exist_ok=True)

    def _find_video(self):
        """Find the first video file in the base directory"""
        video_extensions = ['.MOV', '.mov', '.mp4', '.MP4', '.m4v', '.M4V']
        for ext in video_extensions:
            videos = list(self.base_dir.glob(f'*{ext}'))
            if videos:
                return videos[0]
        # Return a placeholder path if no video found (will error later with helpful message)
        return self.base_dir / "video.MOV"

    def load_progress(self):
        """Load progress from file"""
        if self.progress_file.exists():
            with open(self.progress_file, 'r') as f:
                return set(line.strip() for line in f)
        return set()

    def load_sharp_progress(self):
        """Load Sharp frame progress from JSON file"""
        if self.sharp_progress_file.exists():
            with open(self.sharp_progress_file, 'r') as f:
                return json.load(f)
        return {"completed": [], "failed": []}

    def save_sharp_progress(self, frame_name, success=True):
        """Save Sharp frame progress"""
        progress = self.load_sharp_progress()
        if success:
            if frame_name not in progress["completed"]:
                progress["completed"].append(frame_name)
        else:
            if frame_name not in progress["failed"]:
                progress["failed"].append(frame_name)

        with open(self.sharp_progress_file, 'w') as f:
            json.dump(progress, f, indent=2)

    def mark_complete(self, step_name):
        """Mark a step as complete"""
        with open(self.progress_file, 'a') as f:
            f.write(f"{step_name}\n")
        self._print_success(f"Marked '{step_name}' as complete")

    def is_complete(self, step_name):
        """Check if a step is already complete"""
        progress = self.load_progress()
        return step_name in progress

    def _print_header(self, text):
        """Print a formatted header"""
        print("\n" + "=" * 70)
        print(f"  {text}")
        print("=" * 70)

    def _print_step(self, step_num, text):
        """Print a step header"""
        print(f"\n{'─' * 70}")
        print(f"  STEP {step_num}: {text}")
        print(f"{'─' * 70}")

    def _print_info(self, text):
        """Print info message"""
        print(f"  ℹ {text}")

    def _print_success(self, text):
        """Print success message"""
        print(f"  ✓ {text}")

    def _print_warning(self, text):
        """Print warning message"""
        print(f"  ⚠ {text}")

    def _print_error(self, text):
        """Print error message"""
        print(f"  ✗ {text}")

    def _print_skip(self, text):
        """Print skip message"""
        print(f"  ⏭ {text}")

    def _add_exif_focal_length(self, image_path, focal_length_mm):
        """Add focal length to image EXIF data using piexif"""
        try:
            import piexif
        except ImportError:
            self._print_info("Installing piexif for EXIF metadata...")
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

    def step1_extract_frames(self, focal_length_mm=None):
        """Extract frames from video and optionally add EXIF focal length

        Args:
            focal_length_mm: Focal length to embed in EXIF data (optional)
        """
        step_name = "extract_frames"

        if self.is_complete(step_name):
            self._print_skip("Step 1: Frame extraction already complete")
            return

        self._print_step(1, "Extracting frames from video")

        if not self.video_path.exists():
            raise FileNotFoundError(f"Video not found: {self.video_path}")

        # Get video info
        self._print_info(f"Video: {self.video_path.name}")

        # Extract frames
        output_pattern = str(self.frames_dir / "frame_%04d.jpg")

        cmd = [
            "ffmpeg",
            "-i", str(self.video_path),
            "-qscale:v", "1",  # Highest quality
            "-progress", "pipe:1",  # Progress to stdout
            "-loglevel", "error",  # Only errors to stderr
            output_pattern
        ]

        self._print_info("Extracting frames...")

        # Run ffmpeg with progress bar
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                  universal_newlines=True, bufsize=1)

        # Create progress bar (we'll update it as we go)
        pbar = tqdm(desc="  Extracting", unit=" frames", bar_format='{desc}: {n} frames | {elapsed}')

        for line in process.stdout:
            if line.startswith("frame="):
                try:
                    frame_num = int(line.split("=")[1].strip())
                    pbar.n = frame_num
                    pbar.refresh()
                except:
                    pass

        process.wait()
        pbar.close()

        if process.returncode != 0:
            error = process.stderr.read()
            self._print_error(f"ffmpeg failed: {error}")
            raise subprocess.CalledProcessError(process.returncode, cmd)

        # Count frames
        frames = sorted(self.frames_dir.glob("frame_*.jpg"))
        self._print_success(f"Extracted {len(frames)} frames")

        # Add EXIF focal length if provided
        if focal_length_mm:
            self._print_info(f"Adding EXIF focal length ({focal_length_mm}mm) to frames...")
            for frame in tqdm(frames, desc="  Adding EXIF",
                            bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}'):
                self._add_exif_focal_length(frame, focal_length_mm)
            self._print_success(f"Added EXIF metadata to {len(frames)} frames")

        self.mark_complete(step_name)

    def _run_colmap_with_progress(self, cmd, description, progress_pattern=None):
        """Run a COLMAP command with progress tracking

        Args:
            cmd: Command list to run
            description: Description for progress bar
            progress_pattern: Regex pattern to extract progress (optional)
        """
        import re

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )

        # Create progress bar
        pbar = tqdm(desc=f"  {description}",
                   bar_format='{desc}: {elapsed} | {postfix}',
                   postfix={'status': 'initializing'})

        output_lines = []
        for line in process.stdout:
            output_lines.append(line)

            # Update progress bar with status
            if progress_pattern:
                match = re.search(progress_pattern, line)
                if match:
                    pbar.set_postfix({'status': match.group(1)})
            else:
                # Show last meaningful line
                if line.strip() and not line.startswith('  '):
                    status = line.strip()[:50]
                    pbar.set_postfix({'status': status})

            pbar.update(0)

        process.wait()
        pbar.close()

        if process.returncode != 0:
            self._print_error(f"COLMAP failed")
            print("Last output lines:")
            for line in output_lines[-20:]:
                print(f"  {line.rstrip()}")
            raise subprocess.CalledProcessError(process.returncode, cmd)

        return process.returncode

    def step2_camera_poses(self, focal_length_mm=None, sensor_width_mm=None):
        """Get camera poses using COLMAP only

        Args:
            focal_length_mm: Manual focal length in mm (e.g., 35, 50)
                           If None, will try to extract from video metadata
            sensor_width_mm: Sensor width in mm (e.g., 36 for full-frame, 7 for smartphone)
                           If None, uses 36mm (full-frame default)
        """
        step_name = "camera_poses"

        if self.is_complete(step_name):
            self._print_skip("Step 2: Camera poses already computed")
            return

        self._print_step(2, "Computing camera poses with COLMAP")

        # Try to get focal length if not provided
        if focal_length_mm is None:
            focal_length_mm = self._get_focal_length()

        # Use frames we already extracted in step 1
        images_dir = self.frames_dir
        db_path = self.colmap_dir / "database.db"
        sparse_dir = self.colmap_dir / "sparse"

        # Create dirs
        db_path.parent.mkdir(parents=True, exist_ok=True)
        sparse_dir.mkdir(parents=True, exist_ok=True)

        # Get frame count for progress info
        frames = list(self.frames_dir.glob("frame_*.jpg"))
        num_frames = len(frames)
        self._print_info(f"Processing {num_frames} frames")

        # 1. Feature extraction
        cmd1 = [
            "colmap", "feature_extractor",
            "--database_path", str(db_path),
            "--image_path", str(images_dir),
            "--ImageReader.single_camera", "1",
            "--ImageReader.camera_model", "OPENCV"
        ]

        # Add focal length if available
        if focal_length_mm:
            focal_length_px = self._mm_to_pixels(focal_length_mm, sensor_width_mm=sensor_width_mm)
            cmd1.extend([
                "--ImageReader.camera_params", f"{focal_length_px},{focal_length_px},0,0,0,0,0,0"
            ])
            sensor_used = sensor_width_mm if sensor_width_mm else 36.0
            self._print_info(f"Using focal length: {focal_length_mm}mm (sensor: {sensor_used}mm, {focal_length_px:.0f}px)")

        self._run_colmap_with_progress(cmd1, "Extracting features", r"Processed (\d+/\d+)")
        self._print_success("Features extracted")

        # 2. Feature matching
        cmd2 = [
            "colmap", "sequential_matcher",
            "--database_path", str(db_path),
            "--SequentialMatching.overlap", "10"
        ]
        self._run_colmap_with_progress(cmd2, "Matching features", r"Matching (\d+/\d+)")
        self._print_success("Features matched")

        # 3. Sparse reconstruction
        cmd3 = [
            "colmap", "mapper",
            "--database_path", str(db_path),
            "--image_path", str(images_dir),
            "--output_path", str(sparse_dir)
        ]
        self._run_colmap_with_progress(cmd3, "Building reconstruction", r"Registering image #(\d+)")
        self._print_success("Sparse reconstruction complete")

        # 4. Convert to transforms.json
        self._print_info("Converting to transforms.json...")

        # Install pycolmap if needed
        try:
            import pycolmap
        except ImportError:
            self._print_info("Installing pycolmap...")
            subprocess.run(["pip", "install", "pycolmap"],
                         check=True, capture_output=True, text=True)
            import pycolmap

        # Convert
        self._colmap_to_transforms(sparse_dir, images_dir)

        # Verify output
        transforms_file = self.colmap_dir / "transforms.json"
        if not transforms_file.exists():
            raise FileNotFoundError("transforms.json not created")

        # Load and check
        with open(transforms_file) as f:
            transforms = json.load(f)

        num_poses = len(transforms.get("frames", []))
        self._print_success(f"Found {num_poses} camera poses")
        self._print_success(f"Saved to: {transforms_file.name}")

        self.mark_complete(step_name)

    def _mm_to_pixels(self, focal_length_mm, sensor_width_mm=None):
        """
        Convert focal length from mm to pixels based on sensor width.

        Formula: focal_length_px = focal_length_mm * (image_width_px / sensor_width_mm)

        Args:
            focal_length_mm: Focal length in millimeters
            sensor_width_mm: Sensor width in mm. If None, defaults to 36mm (full-frame).
                           Common values:
                           - 36mm: Full-frame DSLR/mirrorless
                           - 23.8mm: APS-C sensor
                           - 17.3mm: Micro 4/3 (Panasonic)
                           - 7mm: iPhone/Android smartphone main camera
                           - 6mm: Ultra-wide smartphone

        Returns:
            Focal length in pixels (for use with COLMAP)
        """
        if sensor_width_mm is None:
            sensor_width_mm = 36.0  # Default: full-frame

        # Get first frame to determine image width
        frames = sorted(self.frames_dir.glob("frame_*.jpg"))
        if frames:
            try:
                from PIL import Image
                img = Image.open(frames[0])
                image_width_px = img.width
                pixels_per_mm = image_width_px / sensor_width_mm
                return focal_length_mm * pixels_per_mm
            except Exception:
                pass

        # Fallback: assume 1920px width (common for full-frame)
        return focal_length_mm * (1920.0 / sensor_width_mm)

    def _get_focal_length(self):
        """
        Try to extract focal length from video metadata.
        Falls back to user input if extraction fails.

        Returns focal length in mm, or None if not found.
        """
        # Try to extract from video metadata
        try:
            import subprocess

            video_path = self.video_path
            if not video_path.exists():
                return None

            # Use ffprobe to extract camera metadata
            cmd = [
                "ffprobe",
                "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream_side_data_list,stream_side_data",
                "-of", "json",
                str(video_path)
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)

            if result.returncode == 0:
                try:
                    data = json.loads(result.stdout)
                    if "streams" in data and len(data["streams"]) > 0:
                        stream = data["streams"][0]

                        # Check for focal length in side data
                        if "side_data_list" in stream:
                            for sd in stream["side_data_list"]:
                                if "camera_intrinsics" in sd:
                                    focal = sd.get("focal_length")
                                    if focal:
                                        self._print_success(f"Found focal length in video metadata: {focal}mm")
                                        return float(focal)
                except (json.JSONDecodeError, KeyError, TypeError):
                    pass
        except Exception as e:
            pass  # Silently fail, we'll prompt user instead

        # If extraction failed, ask user
        print()
        self._print_info("Focal length not found in video metadata")
        print("     Common values:")
        print("       - 24mm: Wide angle / action cameras")
        print("       - 26-27mm: Most smartphones (iPhone/Android)")
        print("       - 35mm: Standard lens")
        print("       - 50mm: Normal/portrait")
        print()

        while True:
            try:
                user_input = input("     Enter focal length in mm (or press Enter for 35mm default): ").strip()
                if not user_input:
                    self._print_info("Using default: 35mm")
                    return 35.0
                focal_length = float(user_input)
                if focal_length > 0:
                    return focal_length
                else:
                    self._print_warning("Please enter a positive number")
            except ValueError:
                self._print_warning("Please enter a valid number")

    def _ensure_sharp_checkpoint(self):
        """
        Ensure SHARP checkpoint is downloaded and cached.
        Uses a standard cache location to avoid re-downloading.

        Returns path to checkpoint file.
        """
        import urllib.request

        # Use a standard cache location (~/.cache/sharp/)
        cache_dir = Path.home() / ".cache" / "sharp"
        cache_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_path = cache_dir / "sharp_2572gikvuh.pt"

        # If already cached, return it
        if checkpoint_path.exists():
            self._print_info(f"Using cached SHARP checkpoint")
            return str(checkpoint_path)

        # Download the model
        model_url = "https://ml-site.cdn-apple.com/models/sharp/sharp_2572gikvuh.pt"

        self._print_info("Downloading SHARP model (first time only)...")
        self._print_info(f"URL: {model_url}")
        self._print_info(f"Cache: {checkpoint_path}")

        try:
            # Download with progress bar
            def download_progress(block_num, block_size, total_size):
                downloaded = block_num * block_size
                percent = min(100, int(100 * downloaded / total_size))
                print(f"\r     Progress: {percent}% ({downloaded}/{total_size} bytes)", end="")

            urllib.request.urlretrieve(model_url, checkpoint_path, reporthook=download_progress)
            print()  # New line after progress
            self._print_success(f"Downloaded SHARP model")

            return str(checkpoint_path)

        except Exception as e:
            self._print_error(f"Failed to download SHARP checkpoint: {e}")
            self._print_info(f"Try manually downloading from: {model_url}")
            self._print_info(f"And place at: {checkpoint_path}")
            raise

    def _colmap_to_transforms(self, sparse_dir, images_dir):
        """Fallback: manually convert COLMAP to transforms.json"""
        import pycolmap
        import numpy as np

        # Read COLMAP reconstruction
        reconstruction_path = sparse_dir / "0"
        if not reconstruction_path.exists():
            raise FileNotFoundError(f"COLMAP reconstruction not found at {reconstruction_path}")

        reconstruction = pycolmap.Reconstruction(str(reconstruction_path))

        if len(reconstruction.cameras) == 0:
            raise ValueError("No cameras found in COLMAP reconstruction")

        if len(reconstruction.images) == 0:
            raise ValueError("No images found in COLMAP reconstruction")

        # Get camera parameters (assumes single camera)
        camera = list(reconstruction.cameras.values())[0]

        transforms = {
            "camera_model": "OPENCV",
            "fl_x": camera.focal_length_x if hasattr(camera, 'focal_length_x') else camera.focal_length,
            "fl_y": camera.focal_length_y if hasattr(camera, 'focal_length_y') else camera.focal_length,
            "cx": camera.principal_point_x if hasattr(camera, 'principal_point_x') else camera.width / 2,
            "cy": camera.principal_point_y if hasattr(camera, 'principal_point_y') else camera.height / 2,
            "w": camera.width,
            "h": camera.height,
            "frames": []
        }

        # Convert each image
        for img_id, image in reconstruction.images.items():
            # Get rotation and translation
            # COLMAP stores world-to-camera, we need camera-to-world
            qpose = image.cam_from_world()
            qvec = qpose.rotation.quat  # (w, x, y, z)
            tvec = qpose.translation   # (x, y, z)


            # Convert quaternion to rotation matrix
            R = qpose.rotation.matrix()

            # Convert world-to-camera to camera-to-world
            # c2w = [R^T | -R^T @ t]
            R_inv = R.T
            t_inv = -R_inv @ tvec

            # Build 4x4 transform matrix
            transform_matrix = np.eye(4)
            transform_matrix[:3, :3] = R_inv
            transform_matrix[:3, 3] = t_inv

            # Add frame
            transforms["frames"].append({
                "file_path": f"images/{image.name}",
                "transform_matrix": transform_matrix.tolist()
            })

        # Sort frames by filename
        transforms["frames"].sort(key=lambda x: x["file_path"])

        # Save transforms.json
        output_file = self.colmap_dir / "transforms.json"
        with open(output_file, 'w') as f:
            json.dump(transforms, f, indent=2)

        return output_file

    def step3_generate_splats(self, checkpoint_path=None):
        """Run SHARP on each frame

        Args:
            checkpoint_path: Path to SHARP checkpoint (.pt file).
                           If None, will download/cache default model.
        """
        step_name = "generate_splats"

        if self.is_complete(step_name):
            self._print_skip("Step 3: Splat generation already complete")
            return

        self._print_step(3, "Generating splats with SHARP")

        # Get frames to process
        frames = sorted(self.frames_dir.glob("frame_*.jpg"))
        self._print_info(f"Found {len(frames)} frames to process")

        if len(frames) == 0:
            raise FileNotFoundError("No frames found in frames/")

        # Ensure checkpoint is available
        checkpoint = checkpoint_path
        if not checkpoint:
            checkpoint = self._ensure_sharp_checkpoint()

        # Load previous progress
        sharp_progress = self.load_sharp_progress()
        completed_frames = set(sharp_progress.get("completed", []))
        failed_frames = set(sharp_progress.get("failed", []))

        # Calculate what needs processing
        total_frames = len(frames)
        already_done = len(completed_frames)
        previously_failed = len(failed_frames)

        if already_done > 0:
            self._print_info(f"Resuming: {already_done} frames already completed")
        if previously_failed > 0:
            self._print_warning(f"Previously failed: {previously_failed} frames will be retried")

        # Process each frame
        self._print_info("Processing frames with SHARP...")

        # Use tqdm with custom format
        pbar = tqdm(frames,
                   desc="  Processing",
                   bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')

        current_failed = []
        current_success = 0

        for i, frame_path in enumerate(pbar):
            frame_name = frame_path.stem
            output_dir = self.ply_dir / frame_name
            final_ply = self.ply_dir / f"{frame_name}.ply"

            # Skip if already successfully completed
            if frame_name in completed_frames and final_ply.exists():
                current_success += 1
                continue

            # Clean up any existing output
            if output_dir.exists():
                shutil.rmtree(output_dir)
            if final_ply.exists():
                final_ply.unlink()

            cmd = [
                "sharp", "predict",
                "-i", str(frame_path),
                "-o", str(output_dir),
                "--device", "mps"
            ]

            # Add checkpoint if available
            if checkpoint and Path(checkpoint).exists():
                cmd.extend(["--checkpoint-path", str(checkpoint)])

            try:
                # Run Sharp with output suppressed
                result = subprocess.run(cmd, check=True, capture_output=True, text=True)

                # Move the generated PLY to the expected location
                ply_files = list(output_dir.glob("*.ply"))
                if ply_files:
                    shutil.move(str(ply_files[0]), str(final_ply))
                    shutil.rmtree(output_dir)
                    self.save_sharp_progress(frame_name, success=True)
                    current_success += 1
                else:
                    # No PLY generated
                    current_failed.append(f"{frame_name} (no output)")
                    self.save_sharp_progress(frame_name, success=False)

            except subprocess.CalledProcessError as e:
                # Sharp failed on this frame
                error_msg = e.stderr if e.stderr else "unknown error"
                current_failed.append(f"{frame_name} ({error_msg[:50]}...)")
                self.save_sharp_progress(frame_name, success=False)

                # Clean up failed output
                if output_dir.exists():
                    shutil.rmtree(output_dir)

            # Update progress bar description with current stats
            pbar.set_description(f"  Processing (✓{current_success} ✗{len(current_failed)})")

        pbar.close()

        # Final summary
        print()
        self._print_success(f"Successfully processed: {current_success}/{total_frames} frames")

        if len(current_failed) > 0:
            self._print_warning(f"Failed to process: {len(current_failed)} frames")
            self.failed_frames.extend([("Sharp", frame) for frame in current_failed])

        # Verify output
        ply_files = sorted(self.ply_dir.glob("frame_*.ply"))
        ply_files = [p for p in ply_files if p.is_file()]
        self._print_info(f"Total PLY files ready: {len(ply_files)}")

        self.mark_complete(step_name)

    def step4_align_splats(self):
        """Align splats to world space using camera poses"""
        step_name = "align_splats"

        if self.is_complete(step_name):
            self._print_skip("Step 4: Splat alignment already complete")
            return

        self._print_step(4, "Aligning splats to world space")

        # Import required libraries
        import numpy as np
        try:
            from plyfile import PlyData, PlyElement
        except ImportError:
            self._print_info("Installing plyfile...")
            subprocess.run(["pip", "install", "plyfile"], check=True, capture_output=True)
            from plyfile import PlyData, PlyElement

        try:
            from scipy.spatial.transform import Rotation
        except ImportError:
            self._print_info("Installing scipy...")
            subprocess.run(["pip", "install", "scipy"], check=True, capture_output=True)
            from scipy.spatial.transform import Rotation

        # Load camera poses
        transforms_file = self.colmap_dir / "transforms.json"
        with open(transforms_file) as f:
            transforms = json.load(f)

        # Create a mapping from frame filename to transform
        frame_to_transform = {}
        for frame_data in transforms["frames"]:
            # Extract filename from path like "images/frame_0001.jpg"
            file_path = frame_data["file_path"]
            frame_name = Path(file_path).stem  # Get "frame_0001" from "images/frame_0001.jpg"
            frame_to_transform[frame_name] = frame_data["transform_matrix"]

        self._print_info(f"Found {len(frame_to_transform)} camera poses from COLMAP")

        # Get splat files
        ply_files = sorted(self.ply_dir.glob("frame_*.ply"))
        ply_files = [p for p in ply_files if p.is_file()]  # Skip directories

        self._print_info(f"Aligning {len(ply_files)} splats using camera poses...")

        # Process each splat with progress bar
        pbar = tqdm(ply_files,
                   desc="  Aligning",
                   bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')

        aligned_count = 0
        alignment_failures = []

        for ply_file in pbar:
            # Find corresponding camera pose by matching frame name
            frame_name = ply_file.stem  # Get "frame_0001" from "frame_0001.ply"

            if frame_name not in frame_to_transform:
                alignment_failures.append(f"{ply_file.name} (no camera pose - COLMAP failed to register)")
                continue

            try:
                c2w_matrix = np.array(frame_to_transform[frame_name])

                # Load PLY file
                plydata = PlyData.read(str(ply_file))
                vertex_element = plydata['vertex']
                vertex = vertex_element.data  # Get underlying numpy array

                # Extract Gaussian data
                positions = np.stack([vertex['x'], vertex['y'], vertex['z']], axis=-1)

                # Transform positions from camera space to world space
                positions_h = np.concatenate([
                    positions,
                    np.ones((len(positions), 1))
                ], axis=1)  # [N, 4]

                # Apply camera-to-world transform
                positions_world = (c2w_matrix @ positions_h.T).T[:, :3]

                # Transform rotations (quaternions)
                if 'rot_0' in vertex.dtype.names:
                    quats = np.stack([
                        vertex['rot_0'],  # w
                        vertex['rot_1'],  # x
                        vertex['rot_2'],  # y
                        vertex['rot_3']   # z
                    ], axis=-1)

                    # Get rotation from c2w matrix
                    R = c2w_matrix[:3, :3]
                    rot_matrix_scipy = Rotation.from_matrix(R)
                    rot_quat_xyzw = rot_matrix_scipy.as_quat()  # [x, y, z, w]
                    rot_quat_wxyz = np.array([rot_quat_xyzw[3], rot_quat_xyzw[0],
                                             rot_quat_xyzw[1], rot_quat_xyzw[2]])

                    # Multiply quaternions
                    quats_norm = quats / np.linalg.norm(quats, axis=-1, keepdims=True)

                    w1, x1, y1, z1 = rot_quat_wxyz
                    w2, x2, y2, z2 = quats_norm.T

                    quats_world = np.stack([
                        w1*w2 - x1*x2 - y1*y2 - z1*z2,  # w
                        w1*x2 + x1*w2 + y1*z2 - z1*y2,  # x
                        w1*y2 - x1*z2 + y1*w2 + z1*x2,  # y
                        w1*z2 + x1*y2 - y1*x2 + z1*w2   # z
                    ], axis=-1)

                    quats_world = quats_world / np.linalg.norm(quats_world, axis=-1, keepdims=True)

                # Transform scales
                if 'scale_0' in vertex.dtype.names:
                    scales = np.stack([vertex['scale_0'], vertex['scale_1'], vertex['scale_2']], axis=-1)
                    scale_factor = np.cbrt(np.abs(np.linalg.det(c2w_matrix[:3, :3])))
                    scales_world = scales + np.log(scale_factor)

                # Create new vertex array
                vertex_new = np.empty(len(vertex), dtype=vertex.dtype)

                # Copy all original attributes
                for name in vertex.dtype.names:
                    vertex_new[name] = vertex[name]

                # Update transformed attributes
                vertex_new['x'] = positions_world[:, 0]
                vertex_new['y'] = positions_world[:, 1]
                vertex_new['z'] = positions_world[:, 2]

                if 'rot_0' in vertex.dtype.names:
                    vertex_new['rot_0'] = quats_world[:, 0]
                    vertex_new['rot_1'] = quats_world[:, 1]
                    vertex_new['rot_2'] = quats_world[:, 2]
                    vertex_new['rot_3'] = quats_world[:, 3]

                if 'scale_0' in vertex.dtype.names:
                    vertex_new['scale_0'] = scales_world[:, 0]
                    vertex_new['scale_1'] = scales_world[:, 1]
                    vertex_new['scale_2'] = scales_world[:, 2]

                # Save aligned PLY
                aligned_path = self.aligned_ply_dir / ply_file.name
                el = PlyElement.describe(vertex_new, 'vertex')
                PlyData([el], text=False).write(str(aligned_path))
                aligned_count += 1

            except Exception as e:
                alignment_failures.append(f"{ply_file.name} ({str(e)[:50]}...)")
                continue

        pbar.close()

        # Summary
        print()
        self._print_success(f"Aligned {aligned_count} splats to world space")

        if len(alignment_failures) > 0:
            self._print_warning(f"Failed to align: {len(alignment_failures)} splats")
            self.failed_frames.extend([("Alignment", frame) for frame in alignment_failures])

        self._print_success(f"Aligned splats saved to: {self.aligned_ply_dir.name}/")

        self.mark_complete(step_name)

    def reset_progress(self, from_step=None, keep_sharp=False):
        """Reset progress from a specific step onwards

        Args:
            from_step: Step to reset from (None = reset all)
            keep_sharp: If True, preserve Sharp progress even when resetting generate_splats
        """
        if from_step is None:
            # Clear all progress
            if self.progress_file.exists():
                self.progress_file.unlink()
            if not keep_sharp and self.sharp_progress_file.exists():
                self.sharp_progress_file.unlink()
            self._print_success("Reset all progress" + (" (kept Sharp progress)" if keep_sharp else ""))
        else:
            # Clear progress from specific step onwards
            steps = ["extract_frames", "camera_poses", "generate_splats", "align_splats"]
            if from_step not in steps:
                self._print_error(f"Unknown step: {from_step}")
                return

            step_idx = steps.index(from_step)
            steps_to_clear = steps[step_idx:]

            progress = self.load_progress()
            new_progress = progress - set(steps_to_clear)

            with open(self.progress_file, 'w') as f:
                for step in new_progress:
                    f.write(f"{step}\n")

            # Clear Sharp progress if generate_splats is being reset (unless keep_sharp=True)
            if "generate_splats" in steps_to_clear and not keep_sharp and self.sharp_progress_file.exists():
                self.sharp_progress_file.unlink()
                self._print_info("Cleared Sharp progress")
            elif "generate_splats" in steps_to_clear and keep_sharp:
                self._print_info("Kept Sharp progress - will resume from where it left off")

            self._print_success(f"Reset progress from '{from_step}' onwards")

    def show_status(self):
        """Show current progress status"""
        self._print_header("PIPELINE STATUS")

        progress = self.load_progress()
        steps = [
            ("extract_frames", "Extract frames from video"),
            ("camera_poses", "Compute camera poses (COLMAP)"),
            ("generate_splats", "Generate splats with SHARP"),
            ("align_splats", "Align splats to world space")
        ]

        for step_name, description in steps:
            status = "✓" if step_name in progress else "○"
            status_text = "COMPLETE" if step_name in progress else "PENDING"
            print(f"  {status} {status_text:12} │ {description}")

        # Show Sharp progress if available
        if self.sharp_progress_file.exists():
            sharp_progress = self.load_sharp_progress()
            completed = len(sharp_progress.get("completed", []))
            failed = len(sharp_progress.get("failed", []))
            if completed > 0 or failed > 0:
                print(f"\n  Sharp Progress: {completed} completed, {failed} failed")

        print("=" * 70)

    def print_failure_report(self):
        """Print a summary of all failures encountered during the run"""
        if len(self.failed_frames) == 0:
            return

        self._print_header("FAILURE REPORT")

        # Group failures by stage and reason
        failures_by_stage = {}
        colmap_registration_failures = 0

        for stage, frame in self.failed_frames:
            if stage not in failures_by_stage:
                failures_by_stage[stage] = []
            failures_by_stage[stage].append(frame)

            # Count COLMAP registration failures specifically
            if "no camera pose - COLMAP failed to register" in frame:
                colmap_registration_failures += 1

        for stage, frames in failures_by_stage.items():
            self._print_warning(f"{stage}: {len(frames)} failures")

            # For alignment failures, separate COLMAP issues from other errors
            if stage == "Alignment":
                colmap_failures = [f for f in frames if "no camera pose" in f]
                other_failures = [f for f in frames if "no camera pose" not in f]

                if colmap_failures:
                    print(f"     └─ {len(colmap_failures)} frames not registered by COLMAP")
                    print(f"        (COLMAP couldn't find enough features/matches)")

                if other_failures:
                    print(f"     └─ {len(other_failures)} processing errors:")
                    for frame in other_failures[:5]:
                        print(f"        - {frame}")
                    if len(other_failures) > 5:
                        print(f"        ... and {len(other_failures) - 5} more")
            else:
                # Show first few failures
                for frame in frames[:10]:
                    print(f"     - {frame}")
                if len(frames) > 10:
                    print(f"     ... and {len(frames) - 10} more")

        # Add helpful tips
        if colmap_registration_failures > 0:
            print()
            self._print_info("Tips for improving COLMAP registration:")
            print("     - Ensure video has enough motion and texture")
            print("     - Avoid blurry frames or fast motion")
            print("     - Check that focal length setting is correct")
            print("     - Consider using fewer frames (extract every Nth frame)")

        print("=" * 70)

    def run(self, focal_length_mm=None, sensor_width_mm=None, checkpoint_path=None):
        """Run the complete pipeline

        Args:
            focal_length_mm: Optional manual focal length in mm
            sensor_width_mm: Optional sensor width in mm (default: 36 for full-frame)
            checkpoint_path: Optional path to SHARP checkpoint file
        """
        self._print_header("SPLAT FLASHLIGHT PIPELINE")
        print(f"  Working directory: {self.base_dir}")

        # Get focal length early so it can be used in step 1 for EXIF
        if focal_length_mm is None:
            focal_length_mm = self._get_focal_length()

        if focal_length_mm:
            self._print_info(f"Focal length: {focal_length_mm}mm")
        if sensor_width_mm:
            self._print_info(f"Sensor width: {sensor_width_mm}mm")
        else:
            self._print_info(f"Sensor width: 36mm (full-frame default)")
        if checkpoint_path:
            self._print_info(f"Checkpoint: {checkpoint_path}")
        print("=" * 70)

        self.show_status()

        try:
            self.step1_extract_frames(focal_length_mm=focal_length_mm)
            self.step2_camera_poses(focal_length_mm=focal_length_mm, sensor_width_mm=sensor_width_mm)
            self.step3_generate_splats(checkpoint_path=checkpoint_path)
            self.step4_align_splats()

            self._print_header("✓ PIPELINE COMPLETE")
            print(f"  Aligned splats: {self.aligned_ply_dir}")
            print()
            print("  Next steps:")
            print("    - Load splats into MetalSplatter viewer")
            print("    - Or render programmatically with Metal")
            print("=" * 70)

            # Show failure report if any
            if len(self.failed_frames) > 0:
                print()
                self.print_failure_report()

        except KeyboardInterrupt:
            print("\n")
            self._print_warning("Pipeline interrupted by user")
            self._print_info("Progress has been saved. Run again to resume.")
            self.show_status()
            if len(self.failed_frames) > 0:
                print()
                self.print_failure_report()
            sys.exit(1)

        except Exception as e:
            print("\n")
            self._print_header("✗ PIPELINE FAILED")
            print(f"  Error: {e}")
            print()
            print("  Progress has been saved. Fix the error and run again to resume.")
            print("=" * 70)
            self.show_status()
            if len(self.failed_frames) > 0:
                print()
                self.print_failure_report()
            raise


def main():
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description='Splat Flashlight Pipeline - Convert videos to aligned 3D Gaussian splats',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  %(prog)s
    Run pipeline with interactive focal length prompt

  %(prog)s run --focal-length 26 --sensor-width 7
    Run pipeline for smartphone video (iPhone)

  %(prog)s run -f 35
    Run pipeline with 35mm focal length

  %(prog)s run -f 35 -s 36 --checkpoint /path/to/model.pt
    Run with custom SHARP checkpoint

  %(prog)s status
    Show current progress

  %(prog)s reset camera_poses --keep-sharp
    Reset COLMAP and alignment, preserve SHARP progress

  %(prog)s reset
    Reset all progress

  %(prog)s --base-dir /path/to/video run -f 26
    Process video in specific directory

Pipeline Steps:
  extract_frames    - Extract frames from video with EXIF
  camera_poses      - Compute camera poses using COLMAP
  generate_splats   - Generate splats with SHARP
  align_splats      - Align splats to world space

Common Focal Lengths:
  24mm    - Wide angle / action cameras (GoPro)
  26-27mm - Most smartphones (iPhone/Android)
  35mm    - Standard lens
  50mm    - Normal/portrait lens

Common Sensor Widths:
  36mm    - Full-frame DSLR/mirrorless (default)
  23.8mm  - APS-C sensor
  17.3mm  - Micro 4/3
  7mm     - iPhone/Android main camera
  6mm     - Ultra-wide smartphone camera
        '''
    )

    parser.add_argument(
        '--base-dir',
        type=str,
        default=None,
        metavar='DIR',
        help='Base directory containing video (default: current directory)'
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to execute')

    # Run command
    run_parser = subparsers.add_parser('run', help='Run the pipeline')
    run_parser.add_argument(
        '-f', '--focal-length',
        type=float,
        metavar='MM',
        help='Focal length in millimeters (e.g., 26 for smartphone, 35 for standard lens)'
    )
    run_parser.add_argument(
        '-s', '--sensor-width',
        type=float,
        metavar='MM',
        help='Sensor width in millimeters (default: 36 for full-frame)'
    )
    run_parser.add_argument(
        '-c', '--checkpoint',
        type=str,
        metavar='PATH',
        help='Path to SHARP checkpoint file (.pt)'
    )

    # Status command
    subparsers.add_parser('status', help='Show current pipeline progress')

    # Reset command
    reset_parser = subparsers.add_parser('reset', help='Reset progress from a specific step')
    reset_parser.add_argument(
        'step',
        nargs='?',
        choices=['extract_frames', 'camera_poses', 'generate_splats', 'align_splats'],
        help='Step to reset from (omit to reset all)'
    )
    reset_parser.add_argument(
        '--keep-sharp',
        action='store_true',
        help='Preserve SHARP frame progress when resetting generate_splats'
    )

    args = parser.parse_args()

    # Create pipeline
    pipeline = SplatPipeline(args.base_dir)

    # Execute command
    if args.command == 'status':
        pipeline.show_status()
    elif args.command == 'reset':
        pipeline.reset_progress(args.step, keep_sharp=args.keep_sharp)
    elif args.command == 'run':
        # Validate checkpoint path if provided
        if args.checkpoint and not Path(args.checkpoint).exists():
            parser.error(f"Checkpoint file not found: {args.checkpoint}")

        pipeline.run(
            focal_length_mm=args.focal_length,
            sensor_width_mm=args.sensor_width,
            checkpoint_path=args.checkpoint
        )
    else:
        # Default: run pipeline
        pipeline.run()


if __name__ == "__main__":
    main()
