"""Central configuration for SignBridge AI (paths, vocabulary, constants)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw" / "INCLUDE"
PROC_DIR = ROOT / "data" / "processed"
LANDMARK_DIR = PROC_DIR / "landmarks"
SELF_DIR = PROC_DIR / "self"          # your own webcam recordings
MODELS_DIR = ROOT / "models"
DYNAMIC_DIR = MODELS_DIR / "dynamic_model"
DB_PATH = ROOT / "database" / "signbridge.db"

# ---- sequence / feature settings -------------------------------------------
SEQ_LEN = 30                 # frames per sign sequence
MAX_PROCESSED_FRAMES = 60    # max frames MediaPipe is run on per video
POSE_IDS = [0, 11, 12, 13, 14, 15, 16]   # nose, shoulders, elbows, wrists
N_POSE_FEATURES = len(POSE_IDS) * 2      # 14  (x, y)
HAND_FEATURES = 84                       # 21 landmarks x (x, y) x (body-relative + hand-shape)
FEATURE_DIM = N_POSE_FEATURES + 2 * HAND_FEATURES   # 182
# A hand counts as "raised/active" when its wrist is higher than this many
# shoulder-widths below the shoulder line (rest position hangs lower).
ACTIVE_WRIST_Y = 1.2

# ---- vocabulary: 30 signs taken from the INCLUDE dataset --------------------
SELECTED_SIGNS = {
    "Greetings": [
        "Hello", "How are you", "Alright", "Good Morning", "Good afternoon",
        "Good evening", "Good night", "Thank you", "Pleased",
    ],
    "Pronouns": [
        "I", "you", "he", "she", "it", "we", "you (plural)", "they",
    ],
    "Society": [
        "Medicine", "Money", "Bill", "Price", "Sign", "Exercise", "Sport",
        "Ball", "Team", "Technology", "Science", "Peace", "Marriage",
    ],
}
ALL_SIGNS = [s for signs in SELECTED_SIGNS.values() for s in signs]

# Extra sentence-making words. INCLUDE (as downloaded) has no verbs, so these are
# recorded with your own webcam via scripts/record_samples.py and used only by the
# Personalised model. Verify each sign against the ISLRTC dictionary / Sign Learn app
# or a qualified signer before presenting it as valid ISL.
CUSTOM_SIGNS = ["Help", "Need", "Want", "Water", "Food", "Doctor", "Yes", "No"]
RECORDABLE_SIGNS = ALL_SIGNS + CUSTOM_SIGNS
