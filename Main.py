import cv2
from pathlib import Path
import numpy as np
import math

# constants for tuning
ALPHA = 0.3          # how much area vs distance matters in scoring
HUE_TOL = 12
SAT_MIN = 140
VAL_MIN = 50
MIN_AREA = 40        # ignore tiny blobs
MAX_STEP = 15000  # max pixels an object can move per frame
LOST_FRAMES = 10     # how many frames before we reset tracking
SMOOTH_WEIGHT = 0.4  # for smoothing the bounding box
DEBUG = False


def user_inputs():
    rgb_wanted = input("Please input what rgb you want in this format r g b : ")
    rgb_list = list(map(int, rgb_wanted.split()))
    rgb_color = np.uint8([[rgb_list]])
    hsv_color = cv2.cvtColor(rgb_color, cv2.COLOR_RGB2HSV)
    hsv_value = hsv_color[0][0]

    # figure out if hue wraps around
    if hsv_value[0] < HUE_TOL:
        filter_type = 2
    elif hsv_value[0] > (179 - HUE_TOL):
        filter_type = 3
    else:
        filter_type = 1

    return hsv_value, filter_type


def build_mask(hsv, target_hue, filter_type):
    # normal case, no wrapping needed
    if filter_type == 1:
        lower = np.array([target_hue - HUE_TOL, SAT_MIN, VAL_MIN])
        upper = np.array([target_hue + HUE_TOL, 255, 255])
        return cv2.inRange(hsv, lower, upper)

    # hue wraps around so we need two ranges
    if filter_type == 2:  # near 0
        low1 = np.array([(target_hue - HUE_TOL) % 180, SAT_MIN, VAL_MIN])
        high1 = np.array([179, 255, 255])
        low2 = np.array([0, SAT_MIN, VAL_MIN])
        high2 = np.array([target_hue + HUE_TOL, 255, 255])
    else:  # near 179
        low1 = np.array([target_hue - HUE_TOL, SAT_MIN, VAL_MIN])
        high1 = np.array([179, 255, 255])
        low2 = np.array([0, SAT_MIN, VAL_MIN])
        high2 = np.array([(target_hue + HUE_TOL) % 180, 255, 255])

    return cv2.bitwise_or(cv2.inRange(hsv, low1, high1),
                          cv2.inRange(hsv, low2, high2))


def get_centroid(contour):
    M = cv2.moments(contour)
    if M["m00"] == 0:
        return None
    return int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])


def score_candidates(candidates, target_i, target_j, alpha=ALPHA):
    # find best candidate based on combo of area and distance to predicted pos
    max_area = max(area for _, _, _, area in candidates)
    max_dist = max(
        math.sqrt((cx - target_i) ** 2 + (cy - target_j) ** 2)
        for _, cx, cy, _ in candidates
    ) or 1

    def calc_score(item):
        _, cx, cy, area = item
        dist = math.sqrt((cx - target_i) ** 2 + (cy - target_j) ** 2)
        norm_area = area / max_area
        norm_dist = 1 - dist / max_dist
        score = alpha * norm_area + (1 - alpha) * norm_dist
        if DEBUG:
            print(f"[{cx},{cy}] score={score:.3f}")
        return score

    return max(candidates, key=calc_score)


def HSV_Conversion(image_to_convert, hsv_value, filter_type,
                   last_i, last_j, prev_i, prev_j, smooth_box,
                   frame=None):
    # process one frame or image and return tracking info
    if frame is not None:
        image = frame
    else:
        IMG_DIR = Path(__file__).resolve().parent / str(image_to_convert)
        image = cv2.imread(str(IMG_DIR))

    if image is None:
        print("Error: bad path to image in HSV conversion.")
        return None, None, None, None, smooth_box

    low_res = cv2.resize(image, (640, 480), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(low_res, cv2.COLOR_BGR2HSV)

    target_hue = int(hsv_value[0])
    mask = build_mask(hsv, target_hue, filter_type)
    mask = noise_filter(mask, method="gaussian")

    # clean up the mask with morphology
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kern)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kern)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, low_res, mask, None, smooth_box

    # get all valid contours with their centroids and areas
    candidates = []
    MAX_AREA = 6000
    for c in contours:
        area = cv2.contourArea(c)
        if area < MIN_AREA or area > MAX_AREA:
            continue
        x, y, w, h = cv2.boundingRect(c)
        ar = w / h if h > 0 else 0
        if not (0.4 <= ar <= 2.5):
            continue
        cent = get_centroid(c)
        if cent is None:
            continue
        candidates.append((c, cent[0], cent[1], area))

    if not candidates:
        return None, low_res, mask, None, smooth_box

    # pick which contour to track
    if last_i is not None and last_j is not None:
        # predict where object should be based on velocity
        if prev_i is not None and prev_j is not None:
            pred_i = last_i + (last_i - prev_i)
            pred_j = last_j + (last_j - prev_j)
        else:
            pred_i, pred_j = last_i, last_j

        # only consider candidates that are close enough to prediction
        close_enough = [
            item for item in candidates
            if math.hypot(item[1] - pred_i, item[2] - pred_j) <= MAX_STEP
        ]

        if close_enough:
            best = score_candidates(close_enough, pred_i, pred_j)[0]
        else:
            # lost the object this frame
            return None, low_res, mask, None, smooth_box
    else:
        # first frame or after reset, just pick the biggest blob
        best = max(candidates, key=lambda item: item[3])[0]

    cent = get_centroid(best)
    if cent is None:
        return None, low_res, mask, None, smooth_box
    cx, cy = cent

    # smooth bounding box so it doesn't jump around
    raw_box = np.array(cv2.boundingRect(best), dtype=np.float32)
    if smooth_box is None:
        smooth_box = raw_box
    else:
        smooth_box = SMOOTH_WEIGHT * raw_box + (1 - SMOOTH_WEIGHT) * smooth_box

    x, y, w, h = smooth_box.astype(int)
    cv2.rectangle(low_res, (x, y), (x + w, y + h), (0, 255, 0), 2)

    zoomed = digital_zoom(low_res, x, y, w, h, zoom_factor=2.0)

    return (cx, cy), low_res, mask, zoomed, smooth_box


def noise_filter(mask, method="gaussian"):
    if method == "gaussian":
        blurred = cv2.GaussianBlur(mask, (5, 5), 0)
        _, cleaned = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
    elif method == "median":
        cleaned = cv2.medianBlur(mask, 5)
    else:
        cleaned = mask
    return cleaned


def digital_zoom(image, x, y, w, h, zoom_factor=2.0, padding=30):
    img_h, img_w = image.shape[:2]
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(img_w, x + w + padding)
    y2 = min(img_h, y + h + padding)

    cropped = image[y1:y2, x1:x2]
    if cropped.size == 0:
        return None

    new_w = int(cropped.shape[1] * zoom_factor)
    new_h = int(cropped.shape[0] * zoom_factor)
    return cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_LINEAR)


def show_image_result(image_path, hsv_value, filter_type, last_i, last_j):
    # for showing a single image result (level 1)
    centroid, display, mask, zoomed, _ = HSV_Conversion(
        image_path, hsv_value, filter_type,
        last_i, last_j, None, None, None
    )
    if display is not None:
        cv2.imshow("Tracking", display)
        cv2.imshow("Mask", mask)
        if zoomed is not None:
            cv2.imshow("Zoomed", zoomed)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return centroid


def process_frame(frame, hsv_value, filter_type, state):
    # run one frame through the pipeline, update state dict
    centroid, display, mask, zoomed, smooth_box = HSV_Conversion(
        None, hsv_value, filter_type,
        state["last_i"], state["last_j"],
        state["prev_i"], state["prev_j"],
        state["smooth_box"],
        frame=frame,
    )

    if centroid is not None:
        state["prev_i"], state["prev_j"] = state["last_i"], state["last_j"]
        state["last_i"], state["last_j"] = centroid
        state["smooth_box"] = smooth_box
        state["lost_count"] = 0
    else:
        state["lost_count"] += 1
        if state["lost_count"] >= LOST_FRAMES:
            # lost track for too long, reset everything
            state["last_i"] = state["last_j"] = None
            state["prev_i"] = state["prev_j"] = None
            state["smooth_box"] = None

    return display, mask, zoomed


def make_state():
    return {
        "last_i": None, "last_j": None,
        "prev_i": None, "prev_j": None,
        "smooth_box": None,
        "lost_count": 0,
    }


def track_video(video_src, hsv_value, filter_type, mode="realtime"):
    # main video tracking loop
    cap = cv2.VideoCapture(video_src)
    if not cap.isOpened():
        print(f"Error: cannot open video source '{video_src}'")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30
    delay = int(1000 / fps)

    state = make_state()

    if mode == "realtime":
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            display, mask, zoomed = process_frame(frame, hsv_value, filter_type, state)

            if display is not None:
                cv2.imshow("Tracking", display)
                cv2.imshow("Mask", mask)
                if zoomed is not None:
                    cv2.imshow("Zoomed", zoomed)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    elif mode == "postprocess":
        print("Processing video... please wait.")
        all_frames, all_masks, all_zooms = [], [], []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            count += 1

            display, mask, zoomed = process_frame(frame, hsv_value, filter_type, state)
            all_frames.append(display)
            all_masks.append(mask)
            all_zooms.append(zoomed)

            if count % 30 == 0:
                print(f"  Processed {count}/{total_frames} frames...")

        print(f"Done processing {count} frames. Playing back at {fps:.1f} FPS...")

        for i in range(len(all_frames)):
            if all_frames[i] is None:
                continue
            cv2.imshow("Tracking", all_frames[i])
            cv2.imshow("Mask", all_masks[i])
            if all_zooms[i] is not None:
                cv2.imshow("Zoomed", all_zooms[i])

            if cv2.waitKey(delay) & 0xFF == ord('q'):
                break
    else:
        print(f"Unknown mode '{mode}'. Use 'realtime' or 'postprocess'.")

    cap.release()
    cv2.destroyAllWindows()


def save_video(video_src, output_path, hsv_value, filter_type):
    # process video and save annotated output
    cap = cv2.VideoCapture(video_src)
    if not cap.isOpened():
        print(f"Error: cannot open video source '{video_src}'")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (640, 480))

    state = make_state()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        display, _, _ = process_frame(frame, hsv_value, filter_type, state)
        if display is not None:
            writer.write(display)

    cap.release()
    writer.release()
    print(f"Saved output video to {output_path}")


if __name__ == "__main__":
    hsv_value, filter_type = user_inputs()
    print(hsv_value)

    source = input("Enter video path (or 'cam' for webcam, or 'img' for images): ").strip()

    if source == "img":
        # image mode
        last_values = show_image_result("roll_1.jpg", hsv_value, filter_type, None, None)
        if last_values:
           show_image_result("roll_2.jpg", hsv_value, filter_type, last_values[0], last_values[1])
           show_image_result("roll_3.jpg", hsv_value, filter_type, last_values[0], last_values[1])
          #show_image_result("volleyball.webp",hsv_value, filter_type, None, None)
    else:
        # video mode
        if source == "cam":
            source = 0

        mode = input("Mode? (realtime / postprocess): ").strip() or "realtime"
        track_video(source, hsv_value, filter_type, mode=mode)
