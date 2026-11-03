"""Dataset loading and the train/val/test split."""
import re

import numpy as np

from src.config import LANDMARK_DIR, SELECTED_SIGNS


def clean_name(folder_name):
    """'49. How are you' -> 'How are you'."""
    return re.sub(r"^\s*\d+\s*[._\-)]\s*", "", folder_name).strip()


def sanitize(text):
    return re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_")


def selected_lookup():
    """sanitized sign name -> (category, sign name)."""
    return {sanitize(s): (cat, s) for cat, signs in SELECTED_SIGNS.items() for s in signs}


def video_number(stem):
    m = re.search(r"MVI_(\d+)", stem)
    return int(m.group(1)) if m else -1


def assign_groups(numbers, gap=3):
    """Cluster clips of one sign into recording runs: consecutive file numbers
    (gap <= 3) belong to the same run. Heuristic based on file numbering only."""
    numbers = np.asarray(numbers)
    order = np.argsort(numbers, kind="stable")
    groups = np.zeros(len(numbers), dtype=int)
    g, prev = 0, None
    for i in order:
        if prev is not None and numbers[i] - prev > gap:
            g += 1
        groups[i] = g
        prev = numbers[i]
    return groups


def split_by_group(labels, numbers):
    """Return an array of 'train' / 'val' / 'test' per sample.

    Whole recording runs go to one split so near-identical neighbouring takes
    never appear in both train and test. NOT verified to be signer-independent.
    """
    labels = np.asarray(labels)
    numbers = np.asarray(numbers)
    subset = np.full(len(labels), "train", dtype=object)
    for lab in np.unique(labels):
        idx = np.where(labels == lab)[0]
        groups = assign_groups(numbers[idx])
        ng = int(groups.max()) + 1
        if ng >= 4:
            role = lambda k: {1: "val", 3: "test"}.get(k % 4, "train")
        elif ng == 3:
            role = lambda k: {1: "val", 2: "test"}.get(k, "train")
        elif ng == 2:
            role = lambda k: "test" if k == 1 else "train"
        else:  # single run: fall back to the last clips by number
            order = idx[np.argsort(numbers[idx], kind="stable")]
            n_test = max(1, round(0.2 * len(order)))
            subset[order[-n_test:]] = "test"
            continue
        for k, i in zip(groups, idx):
            subset[i] = role(k)
    return subset


def load_landmark_dataset():
    """Load every saved landmark file. Returns X (N,T,F), labels, numbers, names."""
    lookup = selected_lookup()
    X, labels, numbers, names = [], [], [], []
    for path in sorted(LANDMARK_DIR.glob("*.npy")):
        parts = path.stem.split("__")
        if len(parts) != 3 or parts[1] not in lookup:
            continue
        X.append(np.load(path))
        labels.append(lookup[parts[1]][1])
        numbers.append(video_number(parts[2]))
        names.append(path.stem)
    if not X:
        raise FileNotFoundError(f"No landmark files found in {LANDMARK_DIR}. Run extract_landmarks.py first.")
    return np.stack(X).astype(np.float32), np.array(labels), np.array(numbers), names
