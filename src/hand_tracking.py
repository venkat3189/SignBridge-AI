"""MediaPipe wrapper: one RGB frame -> one fixed-length feature vector (182 values).

Layout of the vector:
    [0:14]    upper-body pose (nose, shoulders, elbows, wrists), x/y relative to
              the shoulder midpoint and scaled by shoulder width
    [14:98]   hand in slot 0 (hand on the image-left of the body centre)
    [98:182]  hand in slot 1 (hand on the image-right)
Each hand block = 42 body-relative coords + 42 hand-shape coords (wrist-relative).
Missing hands are zeros. Frames must NOT be mirrored (training videos are not).
"""
from collections.abc import Sequence
from typing import Protocol, cast

import numpy as np
from mediapipe.python.solutions import hands as mp_hands, pose as mp_pose

from src.config import POSE_IDS, N_POSE_FEATURES, HAND_FEATURES, FEATURE_DIM, ACTIVE_WRIST_Y


class _Landmark(Protocol):
    x: float
    y: float
    visibility: float


class _LandmarkList(Protocol):
    landmark: Sequence[_Landmark]


class _HandsResult(Protocol):
    multi_hand_landmarks: Sequence[_LandmarkList] | None


class _PoseResult(Protocol):
    pose_landmarks: _LandmarkList | None


class LandmarkExtractor:
    def __init__(self, static_image_mode=True, hands_complexity=1, pose_complexity=0):
        self._hands = mp_hands.Hands(
            static_image_mode=static_image_mode, max_num_hands=2,
            model_complexity=hands_complexity,
            min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self._pose = mp_pose.Pose(
            static_image_mode=static_image_mode, model_complexity=pose_complexity,
            min_detection_confidence=0.5, min_tracking_confidence=0.5)

    def process(self, rgb):
        """Returns (features[182], info). info has raw results for drawing."""
        pose_res = cast(_PoseResult, self._pose.process(rgb))
        hands_res = cast(_HandsResult, self._hands.process(rgb))

        center = np.array([0.5, 0.5])
        scale = 0.3
        pose_feat = np.zeros(N_POSE_FEATURES, dtype=np.float32)
        wrist_ys = []

        if pose_res.pose_landmarks:
            lm = pose_res.pose_landmarks.landmark
            ls, rs = lm[11], lm[12]
            if ls.visibility > 0.3 and rs.visibility > 0.3:
                center = np.array([(ls.x + rs.x) / 2, (ls.y + rs.y) / 2])
                scale = max(float(np.hypot(ls.x - rs.x, ls.y - rs.y)), 1e-3)
            pts = np.array([[lm[i].x, lm[i].y] for i in POSE_IDS])
            norm = (pts - center) / scale
            pose_feat = norm.ravel().astype(np.float32)
            wrist_ys += [norm[5, 1], norm[6, 1]]   # POSE_IDS[5], [6] = wrists 15, 16

        slots = [np.zeros(HAND_FEATURES, np.float32), np.zeros(HAND_FEATURES, np.float32)]
        filled = [False, False]
        n_hands = 0
        if hands_res.multi_hand_landmarks:
            for hl in hands_res.multi_hand_landmarks[:2]:
                pts = np.array([[p.x, p.y] for p in hl.landmark])
                body_rel = (pts - center) / scale
                rel = pts - pts[0]
                span = max(float(np.abs(rel).max()), 1e-3)
                feat = np.concatenate([body_rel.ravel(), (rel / span).ravel()]).astype(np.float32)
                slot = 0 if pts[0, 0] < center[0] else 1
                if filled[slot]:
                    slot = 1 - slot
                if not filled[slot]:
                    slots[slot] = feat
                    filled[slot] = True
                    n_hands += 1
                wrist_ys.append(body_rel[0, 1])

        active = bool(wrist_ys) and min(wrist_ys) < ACTIVE_WRIST_Y and n_hands > 0
        vec = np.concatenate([pose_feat, slots[0], slots[1]]).astype(np.float32)
        assert vec.shape[0] == FEATURE_DIM
        info = {"pose": pose_res.pose_landmarks, "hands": hands_res.multi_hand_landmarks,
                "n_hands": n_hands, "active": active}
        return vec, info

    def close(self):
        self._hands.close()
        self._pose.close()
