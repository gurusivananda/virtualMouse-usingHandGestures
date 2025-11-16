# main.py (fixed + slower cursor + robust gestures)
import cv2
import mediapipe as mp
import pyautogui
import time
import math
from collections import deque

# ----------------- Config (tweak these if needed) -----------------
CAMERA_ID = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

SMOOTHING_WINDOW = 5         # moving average window for target cursor position (frames)
MARGIN = 0.10                # crop 10% border from camera frame for UX

# Relative thresholds (fractions of detected hand height)
THUMB_INDEX_CLICK_RATIO = 0.14     # pinch threshold for click
THUMB_INDEX_DRAG_RATIO = 0.09      # sustained pinch starts drag
THUMB_INDEX_DRAG_RELEASE_RATIO = 0.20

INDEX_MIDDLE_RIGHT_CLICK_RATIO = 0.14

MIDDLE_BEND_TAP_RATIO = 0.14       # tip-to-PIP distance ratio to detect middle finger tap

CLICK_COOLDOWN = 0.30              # seconds between allowed clicks
TAP_MAX_DURATION = 0.25            # max time between bend start and end for a tap

# Cursor smoothing / speed control
CURSOR_SMOOTHING = 7   # higher => slower & smoother (typical 5-12)
CURSOR_SCALE = 0.70    # <1 slows cursor movement (0.5 very slow) - multiplies movement fraction

# ------------------------------------------------------------------

pyautogui.PAUSE = 0
pyautogui.FAILSAFE = True

screen_w, screen_h = pyautogui.size()

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils


def px_dist(p1, p2, w, h):
    """Pixel Euclidean distance between normalized landmarks p1/p2."""
    return math.hypot((p1.x - p2.x) * w, (p1.y - p2.y) * h)


def hand_bbox_size(landmarks, w, h):
    """Return hand bbox (width, height) in pixels from landmarks list."""
    xs = [lm.x for lm in landmarks]
    ys = [lm.y for lm in landmarks]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    return (max_x - min_x) * w, (max_y - min_y) * h


def norm_to_screen(x_norm, y_norm):
    """Convert normalized camera coords to screen coords, with margin cropping."""
    def clamp(v, lo=0.0, hi=1.0):
        return max(lo, min(hi, v))
    x = clamp((x_norm - MARGIN) / (1 - 2 * MARGIN))
    y = clamp((y_norm - MARGIN) / (1 - 2 * MARGIN))
    return int(x * screen_w), int(y * screen_h)


def main():
    cap = cv2.VideoCapture(CAMERA_ID)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    # state
    last_click_time = 0
    last_right_time = 0
    dragging = False
    last_seen = time.time()
    pos_buffer = deque(maxlen=SMOOTHING_WINDOW)  # smoothing queue for target pos

    # cursor prev for smoothing & speed control
    prev_x, prev_y = screen_w // 2, screen_h // 2

    # tap detection state for middle finger
    middle_prev_dist = None
    middle_bend_start = None
    last_middle_tap = 0

    with mp_hands.Hands(max_num_hands=1,
                        model_complexity=1,
                        min_detection_confidence=0.6,
                        min_tracking_confidence=0.6) as hands:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Camera read failed.")
                break

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = hands.process(rgb)

            if res.multi_hand_landmarks:
                last_seen = time.time()
                hand = res.multi_hand_landmarks[0]
                lm = hand.landmark

                # compute hand bbox height (pixel) for scale-adaptive thresholds
                _, hand_h_px = hand_bbox_size(lm, w, h)
                if hand_h_px < 20:
                    hand_h_px = 120  # fallback if detection tiny

                # dynamic thresholds in pixels
                click_thresh_px = max(6, hand_h_px * THUMB_INDEX_CLICK_RATIO)
                drag_thresh_px = max(5, hand_h_px * THUMB_INDEX_DRAG_RATIO)
                drag_rel_thresh_px = max(10, hand_h_px * THUMB_INDEX_DRAG_RELEASE_RATIO)
                right_click_thresh_px = max(6, hand_h_px * INDEX_MIDDLE_RIGHT_CLICK_RATIO)
                middle_bend_thresh_px = max(5, hand_h_px * MIDDLE_BEND_TAP_RATIO)

                # important landmarks
                thumb = lm[4]
                index_tip = lm[8]
                index_pip = lm[6]
                middle_tip = lm[12]
                middle_pip = lm[10]

                # pixel distances
                d_thumb_index = px_dist(thumb, index_tip, w, h)
                d_index_middle = px_dist(index_tip, middle_tip, w, h)
                d_middle_tip_pip = px_dist(middle_tip, middle_pip, w, h)

                # Cursor mapping and smoothing (moving average)
                tx, ty = norm_to_screen(index_tip.x, index_tip.y)
                pos_buffer.append((tx, ty))
                avg_x = int(sum(p[0] for p in pos_buffer) / len(pos_buffer))
                avg_y = int(sum(p[1] for p in pos_buffer) / len(pos_buffer))

                # Smooth + slow cursor: compute movement fraction toward averaged target
                # Move a fraction of the distance each frame, scaled by CURSOR_SCALE and CURSOR_SMOOTHING
                try:
                    move_frac = (1.0 / float(CURSOR_SMOOTHING)) * float(CURSOR_SCALE)
                except Exception:
                    move_frac = 0.1
                cursor_x = prev_x + (avg_x - prev_x) * move_frac
                cursor_y = prev_y + (avg_y - prev_y) * move_frac

                # Clip to screen bounds (safety)
                cursor_x = max(0, min(screen_w - 1, int(cursor_x)))
                cursor_y = max(0, min(screen_h - 1, int(cursor_y)))

                # Move mouse
                try:
                    pyautogui.moveTo(cursor_x, cursor_y, _pause=False)
                except pyautogui.FailSafeException:
                    print("FailSafe triggered, stopping.")
                    break

                # update prev
                prev_x, prev_y = cursor_x, cursor_y

                now = time.time()

                # ---- Pinch-based left click (thumb + index) ----
                if d_thumb_index < click_thresh_px and (now - last_click_time) > CLICK_COOLDOWN:
                    # avoid accidental clicks while moving; require index finger fairly extended:
                    finger_up = (index_tip.y < index_pip.y)
                    if finger_up:
                        pyautogui.click()
                        last_click_time = now

                # ---- Pinch-based right click (index + middle) ----
                if d_index_middle < right_click_thresh_px and (now - last_right_time) > CLICK_COOLDOWN:
                    pyautogui.rightClick()
                    last_right_time = now

                # ---- Drag: sustained pinch (thumb-index) ----
                if d_thumb_index < drag_thresh_px and not dragging:
                    dragging = True
                    pyautogui.mouseDown()
                elif d_thumb_index > drag_rel_thresh_px and dragging:
                    dragging = False
                    pyautogui.mouseUp()

                # ---- Middle finger tap detection (bend) ----
                # detect a quick bend (tip -> pip) and release within TAP_MAX_DURATION -> tap
                if middle_prev_dist is None:
                    middle_prev_dist = d_middle_tip_pip

                if d_middle_tip_pip < middle_bend_thresh_px and middle_prev_dist >= middle_bend_thresh_px:
                    middle_bend_start = now

                if 'middle_bend_start' in locals() and middle_bend_start is not None and d_middle_tip_pip >= middle_bend_thresh_px:
                    duration = now - middle_bend_start
                    if duration <= TAP_MAX_DURATION and (now - last_middle_tap) > CLICK_COOLDOWN:
                        pyautogui.click()
                        last_middle_tap = now
                    middle_bend_start = None

                middle_prev_dist = d_middle_tip_pip

                # ---- Drawing & debug overlay ----
                # draw fingertips
                for idx_id in (4, 8, 12):
                    cx, cy = int(lm[idx_id].x * w), int(lm[idx_id].y * h)
                    cv2.circle(frame, (cx, cy), 8, (0, 255, 255), cv2.FILLED)

                # thresholds visualization (text)
                cv2.putText(frame, f"hand_h_px: {int(hand_h_px)}", (10, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 2)
                cv2.putText(frame, f"thumb-idx: {int(d_thumb_index)} (th:{int(click_thresh_px)})", (10, 45),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(frame, f"idx-mid: {int(d_index_middle)} (th:{int(right_click_thresh_px)})", (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
                cv2.putText(frame, f"mid-tip-pip: {int(d_middle_tip_pip)} (tap_th:{int(middle_bend_thresh_px)})", (10, 95),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 180, 0), 2)

            else:
                # lost hand -> reset drag state if necessary
                if time.time() - last_seen > 1.0:
                    if dragging:
                        try:
                            pyautogui.mouseUp()
                        except Exception:
                            pass
                        dragging = False
                    pos_buffer.clear()

            cv2.imshow("Virtual Mouse (press q to quit)", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord('c'):
                # quick debug print to help tune values in your environment
                print(f"CURSOR_SMOOTHING={CURSOR_SMOOTHING}, CURSOR_SCALE={CURSOR_SCALE}")
                print(f"SMOOTHING_WINDOW={SMOOTHING_WINDOW}")
                # show last buffer if any
                if pos_buffer:
                    print("pos_buffer sample:", list(pos_buffer)[:5])

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()