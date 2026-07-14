"""Ingest step 1 — extract audio from each video with ffmpeg.

Fixes the old `01.mp4.mp3` double-extension bug (so rename.py is no longer needed)
and skips files that are already extracted so re-runs are cheap.

Requires ffmpeg on PATH.  Run:  python -m courselens.ingest.extract_audio
"""
import subprocess

from courselens.config import VIDEO_DIR, AUDIO_DIR


def main():
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    for video in sorted(VIDEO_DIR.glob("*.mp4")):
        out = AUDIO_DIR / f"{video.stem}.mp3"   # 01.mp4 -> 01.mp3
        if out.exists():
            print(f"skip {out.name} (already extracted)")
            continue
        print(f"extracting {video.name} -> {out.name}")
        subprocess.run(["ffmpeg", "-i", str(video), str(out)], check=True)
    print("Audio extraction complete.")


if __name__ == "__main__":
    main()
