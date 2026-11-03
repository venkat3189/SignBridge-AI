"""Extracts MediaPipe landmarks from the selected INCLUDE videos.

For every video: sample frames -> MediaPipe Hands + Pose -> keep the span where a
hand is raised -> resample to 30 frames -> save (30, 182) to data/processed/landmarks.
Resumable: videos that already have an output file are skipped.

Run (from the project folder, environment active):
    python scripts/extract_landmarks.py --limit 5      # quick test
    python scripts/extract_landmarks.py --workers 4    # full run
"""
import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

from src.config import RAW_DIR, LANDMARK_DIR, SEQ_LEN, MAX_PROCESSED_FRAMES, SELECTED_SIGNS
from src.data_preprocessing import clean_name, sanitize
from src.feature_extraction import resample_sequence

VIDEO_EXT = {".mov", ".mp4", ".avi", ".mkv", ".m4v"}
_extractor = None


def _get_extractor():
    global _extractor
    if _extractor is None:
        from src.hand_tracking import LandmarkExtractor
        _extractor = LandmarkExtractor(static_image_mode=True, hands_complexity=1, pose_complexity=0)
    return _extractor


def read_frames(path):
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        if w > 640:
            frame = cv2.resize(frame, (640, int(h * 640 / w)))
        frames.append(frame)
    cap.release()
    return frames


def video_to_sequence(path):
    frames = read_frames(path)
    if not frames:
        raise RuntimeError("no frames could be read")
    n = len(frames)
    idxs = np.unique(np.linspace(0, n - 1, min(n, MAX_PROCESSED_FRAMES)).astype(int))
    ext = _get_extractor()
    feats, active = [], []
    for i in idxs:
        rgb = cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB)
        vec, info = ext.process(rgb)
        feats.append(vec)
        active.append(info["active"])
    feats = np.stack(feats)
    active = np.array(active)
    if active.any():
        pos = np.where(active)[0]
        start, end = pos[0], pos[-1] + 1
        used_active = True
    else:
        start, end = 0, len(feats)
        used_active = False
    seq = resample_sequence(feats[start:end], SEQ_LEN)
    hand_frac = float((np.abs(seq[:, 14:]).sum(axis=1) > 0).mean())
    return seq, used_active, hand_frac


def work(job):
    src, dst = job
    try:
        seq, used_active, hand_frac = video_to_sequence(Path(src))
        np.save(dst, seq)
        return src, "ok", f"active={used_active} hands_in_frames={hand_frac:.2f}"
    except Exception as exc:
        return src, "error", f"{type(exc).__name__}: {exc}"


def collect_jobs():
    jobs, missing = [], []
    for category, signs in SELECTED_SIGNS.items():
        cat_dir = RAW_DIR / category
        if not cat_dir.exists():
            missing.append(f"category folder {cat_dir}")
            continue
        folders = {clean_name(p.name): p for p in cat_dir.iterdir() if p.is_dir()}
        for sign in signs:
            folder = folders.get(sign)
            if folder is None:
                missing.append(f"{category}/{sign}")
                continue
            for video in sorted(folder.iterdir()):
                if video.suffix.lower() in VIDEO_EXT:
                    name = f"{sanitize(category)}__{sanitize(sign)}__{sanitize(video.stem)}.npy"
                    jobs.append((str(video), str(LANDMARK_DIR / name)))
    return jobs, missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="process only N videos (test run)")
    args = ap.parse_args()

    LANDMARK_DIR.mkdir(parents=True, exist_ok=True)
    jobs, missing = collect_jobs()
    if missing:
        print("WARNING - not found in data/raw/INCLUDE:", *missing, sep="\n  ")
    todo = [j for j in jobs if not Path(j[1]).exists()]
    print(f"{len(jobs)} videos selected, {len(jobs) - len(todo)} already done, {len(todo)} to process.")
    if args.limit:
        todo = todo[: args.limit]
    if not todo:
        print("Nothing to do.")
        return

    t0, done, errors, no_active = time.time(), 0, [], 0
    if args.workers <= 1:
        results = (work(j) for j in todo)
        pool = None
    else:
        pool = ProcessPoolExecutor(max_workers=args.workers)
        results = (f.result() for f in as_completed([pool.submit(work, j) for j in todo]))

    for src, status, note in results:
        done += 1
        if status == "error":
            errors.append((src, note))
        elif "active=False" in note:
            no_active += 1
        if done % 10 == 0 or done == len(todo):
            rate = (time.time() - t0) / done
            print(f"  {done}/{len(todo)}  ~{rate * (len(todo) - done) / 60:.1f} min left")
        if args.limit:
            print(f"  {Path(src).name}: {status} {note}")
    if pool:
        pool.shutdown()

    print(f"\nFinished in {(time.time() - t0) / 60:.1f} min. "
          f"{done - len(errors)} ok, {len(errors)} errors, "
          f"{no_active} videos without a detected raised hand (used the full clip).")
    for src, note in errors[:15]:
        print("  ERROR", src, note)


if __name__ == "__main__":
    main()
