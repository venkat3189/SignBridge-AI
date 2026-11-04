"""Record your own webcam samples of signs (uses the SAME pipeline as the live app).

Run (example):
    python scripts/record_samples.py --signs "Hello,How are you,Thank you,Good night,I,you" --clips 12

For every sign: press SPACE to begin, then repeat:
    raise your hands -> sign -> lower your hands fully -> wait for 'Saved'.
Keys:  SPACE start  |  U undo last clip  |  S skip sign  |  Q quit
Clips are saved to data/processed/self/.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
from mediapipe.python.solutions import drawing_utils, hands, pose

from src.config import RECORDABLE_SIGNS, SELF_DIR
from src.confidence_filter import SignSegmenter
from src.data_preprocessing import sanitize
from src.hand_tracking import LandmarkExtractor

BLUE = (255, 99, 37)  # BGR


def draw_landmarks(frame, info):
    draw = drawing_utils
    if info["pose"]:
        draw.draw_landmarks(frame, info["pose"], list(pose.POSE_CONNECTIONS),
                            draw.DrawingSpec(color=(190, 190, 190), thickness=1, circle_radius=1),
                            draw.DrawingSpec(color=(210, 210, 210), thickness=1))
    for hl in info["hands"] or []:
        draw.draw_landmarks(frame, hl, list(hands.HAND_CONNECTIONS),
                            draw.DrawingSpec(color=BLUE, thickness=2, circle_radius=3),
                            draw.DrawingSpec(color=(255, 255, 255), thickness=2))


def put(img, text, y, scale=0.9, color=(255, 255, 255), thick=2):
    cv2.putText(img, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 3, cv2.LINE_AA)
    cv2.putText(img, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--signs", required=True, help='comma separated, e.g. "Hello,Thank you,I"')
    ap.add_argument("--clips", type=int, default=12)
    ap.add_argument("--camera", type=int, default=0)
    args = ap.parse_args()

    signs = [s.strip() for s in args.signs.split(",") if s.strip()]
    unknown = [s for s in signs if s not in RECORDABLE_SIGNS]
    if unknown:
        print("Unknown sign name(s):", unknown)
        print("Valid names:", RECORDABLE_SIGNS)
        return 1

    SELF_DIR.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        print("Could not open the camera. Close other apps that use it.")
        return 1
    extractor = LandmarkExtractor(static_image_mode=False, hands_complexity=1, pose_complexity=0)
    seg = SignSegmenter()
    window = "SignBridge recorder"

    try:
        for si, sign in enumerate(signs, 1):
            saved, msg, msg_until, started = [], "", 0.0, False
            seg.reset()
            while len(saved) < args.clips:
                ok, frame = cap.read()
                if not ok:
                    print("Camera stopped sending frames.")
                    return 1
                vec, info = extractor.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                seq = seg.update(vec, info["active"]) if started else None
                draw_landmarks(frame, info)
                disp = cv2.flip(frame, 1)

                if seq is not None:
                    hand_frac = float((np.abs(seq[:, 14:]).sum(axis=1) > 0).mean())
                    if hand_frac < 0.6:
                        msg, msg_until = "Hands were lost - repeat the sign", time.time() + 1.5
                    else:
                        path = SELF_DIR / f"{sanitize(sign)}__{int(time.time() * 1000)}.npy"
                        np.save(path, seq)
                        saved.append(path)
                        msg, msg_until = f"Saved clip {len(saved)}", time.time() + 1.0

                put(disp, f"Sign {si}/{len(signs)}: {sign}", 40, 1.0, (255, 200, 120))
                if not started:
                    put(disp, "Press SPACE when ready (Q = quit)", 80, 0.8)
                else:
                    put(disp, f"Clips: {len(saved)}/{args.clips}   state: {seg.state}", 80, 0.8)
                    if time.time() < msg_until:
                        put(disp, msg, 120, 0.9, (120, 255, 160))
                put(disp, "Raise hands -> sign -> lower hands fully", 465, 0.6)
                cv2.imshow(window, disp)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    return 0
                if key == ord(" ") and not started:
                    started = True
                    seg.reset()
                if key == ord("s"):
                    break
                if key == ord("u") and saved:
                    saved.pop().unlink(missing_ok=True)
                    msg, msg_until = "Last clip removed", time.time() + 1.0
            print(f"{sign}: {len(saved)} clips saved")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        extractor.close()
    print("Done. Next: python scripts/train_demo_model.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
