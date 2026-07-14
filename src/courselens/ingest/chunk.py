"""Ingest step 3 — merge tiny Whisper segments into ~400-word overlapping chunks.

Reads data/jsons/*.json -> writes data/merged_jsons/*.json.
Each output file: {Video_id, merged_chunks:[{_id, Video_id, start, end, text}], full_text}.

Run:  python -m courselens.ingest.chunk

KNOWN ISSUE (see CURRENT_STATE / roadmap Phase 1): after a flush, `buffer_start`
is reset to the current segment's start, but the retained OVERLAP_WORDS actually
began earlier — so timestamps on post-first chunks can be slightly off. Left as-is
here to preserve current behaviour; fix it when you get to retrieval quality work.
"""
import json

from courselens.config import SEGMENTS_DIR, CHUNKS_DIR, MAX_WORDS, OVERLAP_WORDS


def merge_chunks(video_data):
    merged = []
    buffer_words = []
    buffer_start = None
    buffer_end = None
    new_id = 1

    for chunk in video_data["chunks"]:
        words = chunk["text"].split()

        if buffer_start is None:
            buffer_start = chunk["start"]

        buffer_words.extend(words)
        buffer_end = chunk["end"]

        if len(buffer_words) >= MAX_WORDS:
            merged.append({
                "_id": new_id,
                "Video_id": video_data["Video_id"],
                "start": buffer_start,
                "end": buffer_end,
                "text": " ".join(buffer_words),
            })
            buffer_words = buffer_words[-OVERLAP_WORDS:]   # keep overlap
            buffer_start = chunk["start"]
            new_id += 1

    if buffer_words:
        merged.append({
            "_id": new_id,
            "Video_id": video_data["Video_id"],
            "start": buffer_start,
            "end": buffer_end,
            "text": " ".join(buffer_words),
        })

    return merged


def main():
    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
    for file in sorted(SEGMENTS_DIR.glob("*.json")):
        print(f"merging {file.name} ...")
        content = json.loads(file.read_text())
        output = {
            "Video_id": content["Video_id"],
            "merged_chunks": merge_chunks(content),
            "full_text": content["full_text"],
        }
        (CHUNKS_DIR / file.name).write_text(json.dumps(output, indent=4))
    print("Merging complete.")


if __name__ == "__main__":
    main()
