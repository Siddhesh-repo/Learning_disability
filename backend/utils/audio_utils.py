"""
Audio format conversion utilities.
"""
import os
import subprocess
import logging

from config import Config

logger = logging.getLogger(__name__)


def find_ffmpeg() -> str | None:
    """Locate an FFmpeg binary on the system."""
    paths = [Config.FFMPEG_PATH, "ffmpeg", "/usr/bin/ffmpeg", "/usr/local/bin/ffmpeg"]
    for p in paths:
        try:
            subprocess.run(
                [p, "-version"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
            )
            return p
        except (subprocess.CalledProcessError, FileNotFoundError):
            continue
    return None


def convert_to_wav(input_path: str, output_path: str) -> bool:
    """Convert any supported audio format to 16 kHz mono WAV."""
    ffmpeg = find_ffmpeg()
    if ffmpeg is None:
        raise RuntimeError(
            "FFmpeg not found. Install it with: brew install ffmpeg (macOS) "
            "or apt-get install ffmpeg (Linux)."
        )

    cmd = [
        ffmpeg, "-i", input_path,
        "-acodec", "pcm_s16le",
        "-ar", str(Config.AUDIO_SAMPLE_RATE),
        "-ac", "1",
        "-y", output_path,
    ]

    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        error = result.stderr.decode(errors="replace")
        raise RuntimeError(f"FFmpeg conversion failed: {error[:500]}")

    if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
        raise RuntimeError("Conversion produced an empty file.")

    logger.info("Converted %s -> %s (%d bytes)", input_path, output_path, os.path.getsize(output_path))
    return True
