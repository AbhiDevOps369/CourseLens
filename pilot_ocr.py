"""One-off OCR PILOT (temporary, delete after use) — NOT the real Phase 4B build.

Question we're actually answering: on THIS video content (live coding, lots of
narration), does OCR extract anything real and non-redundant, or is it mostly
noise/duplicate of the transcript? Look at the raw output honestly before
deciding whether the full OCR pipeline (all 8 videos, second Chroma
collection, fusion, extended golden set) is worth building at all.

Prereqs (run yourself, not in this script):
    brew install tesseract              # the actual OCR engine (not a pip package)
    pip install pytesseract Pillow

Step 1 — extract a handful of frames from ONE video yourself:
    mkdir -p pilot_frames
    ffmpeg -i data/videos/03.mp4 -vf fps=1/12 -t 300 pilot_frames/frame_%04d.jpg
    (the -t 300 caps it to the first 5 minutes — plenty for a pilot, no need
    to process a whole video just to eyeball quality)

Step 2 — run this script:
    python pilot_ocr.py

It OCRs every frame in pilot_frames/, skips near-blank ones (talking-head
shots with no on-screen text), and prints what it found so you can read it
and judge for yourself: real signal, or noise?
"""
from pathlib import Path

import pytesseract
from PIL import Image

FRAMES_DIR = Path(__file__).parent / "pilot_frames"
MIN_CHARS = 15  # roadmap's own cutoff — filters blank/near-blank frames


def main():
    frame_paths = sorted(FRAMES_DIR.glob("*.jpg"))
    if not frame_paths:
        print(f"No frames found in {FRAMES_DIR}. Run the ffmpeg command in this file's docstring first.")
        return

    print(f"Found {len(frame_paths)} frames. OCR-ing each one...\n")

    kept = 0
    for frame_path in frame_paths:
        text = pytesseract.image_to_string(Image.open(frame_path)).strip()
        alnum_chars = sum(c.isalnum() for c in text)

        if alnum_chars < MIN_CHARS:
            continue  # near-blank frame, skip printing it

        kept += 1
        print(f"--- {frame_path.name} ({alnum_chars} alnum chars) ---")
        print(text)
        print()

    print(f"\n{kept}/{len(frame_paths)} frames had meaningful text (≥{MIN_CHARS} alnum chars).")
    print("Now the real question: read the text above — is any of it something")
    print("that ISN'T already covered by what the instructor says out loud in")
    print("this section of the video? That's what decides if OCR is worth building out.")


if __name__ == "__main__":
    main()
