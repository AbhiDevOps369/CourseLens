"""Ingest step 2 — Whisper Hindi->English transcription -> data/jsons/*.json.

Each output file: {Video_id, chunks:[{_id, Video_id, start, end, text}], full_text}.

NOTE: large-v2 is slow and this compute is already spent for the processed videos.
This script SKIPS any video that already has a transcript, so you can safely run it
to process the remaining videos without re-doing finished ones.

Run:  python -m courselens.ingest.transcribe
"""
import json
import time

import whisper

from courselens.config import (
    AUDIO_DIR,
    SEGMENTS_DIR,
    WHISPER_MODEL,
    WHISPER_LANGUAGE,
    WHISPER_TASK,
)


def main():
    SEGMENTS_DIR.mkdir(parents=True, exist_ok=True)
    model = whisper.load_model(WHISPER_MODEL)

    global_id = 1
    start_time = time.time()

    for audio in sorted(AUDIO_DIR.glob("*.mp3")):
        video_id = audio.stem
        out = SEGMENTS_DIR / f"{video_id}.json"
        if out.exists():
            print(f"skip {video_id} (already transcribed)")
            continue

        print(f"transcribing {audio.name} ...")
        result = model.transcribe(
            audio=str(audio),
            language=WHISPER_LANGUAGE,
            task=WHISPER_TASK,
            word_timestamps=False,
        )

        chunks = []
        for segment in result["segments"]:
            chunks.append({
                "_id": global_id,
                "Video_id": video_id,
                "start": segment["start"],
                "end": segment["end"],
                "text": segment["text"].strip(),
            })
            global_id += 1

        out.write_text(json.dumps(
            {"Video_id": video_id, "chunks": chunks, "full_text": result["text"].strip()},
            indent=4,
        ))

    elapsed = int(time.time() - start_time)
    print("\nAll files processed.")
    print(f"Total time: {elapsed // 3600}h {(elapsed % 3600) // 60}m {elapsed % 60}s")


if __name__ == "__main__":
    main()
