# vision/video_utils.py

import cv2
from PIL import Image
from typing import List, Tuple, Optional

def sample_frames(video_path: str, step_seconds: int = 2, max_frames: int = 60) -> Tuple[List[Image.Image], List[float]]:
    """
    Extract frames from a video at fixed time intervals.

    Args:
        video_path (str): Path to the video file.
        step_seconds (int): Interval in seconds between sampled frames.
        max_frames (int): Maximum number of frames to extract.

    Returns:
        Tuple[List[Image.Image], List[float]]:
            - List of frames as PIL Images.
            - List of timestamps (in seconds) for each frame.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return [], []

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25  # fallback default

    step = max(int(fps * step_seconds), 1)
    frames, times, idx = [], [], 0

    while True:
        ok, frame = cap.read()
        if not ok or len(frames) >= max_frames:
            break
        if idx % step == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(rgb))
            times.append(idx / fps)
        idx += 1

    cap.release()
    return frames, times

def majority_vote(labels: List[str]) -> Tuple[Optional[str], int]:
    """
    Return the most frequent label from a list.

    Args:
        labels (List[str]): List of predicted labels.

    Returns:
        Tuple[str, int]: Most common label and its count.
    """
    if not labels:
        return None, 0

    counts = {}
    for lbl in labels:
        counts[lbl] = counts.get(lbl, 0) + 1

    return max(counts.items(), key=lambda x: x[1])