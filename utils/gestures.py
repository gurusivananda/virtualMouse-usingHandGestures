# utils/gestures.py
import math

def normalized_distance(lm1, lm2):
    """
    lm1, lm2 are MediaPipe normalized landmarks with .x and .y attributes.
    Returns Euclidean distance in normalized coordinates (0..1).
    """
    dx = lm1.x - lm2.x
    dy = lm1.y - lm2.y
    return math.hypot(dx, dy)
