import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import math

base_options = python.BaseOptions(model_asset_path="../model/hand_landmarker.task")
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
detector = vision.HandLandmarker.create_from_options(options)

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]

# Load memes
memes = {
    "peace": cv2.imread("../images/hamster_peace.webp"),
    "fingergun": cv2.imread("../images/hamster_gun.webp"),
    "fist": cv2.imread("../images/hamster_scream.jpg"),
    "flat": cv2.imread("../images/hamster_flat.webp"),
    "thumbsup": cv2.imread("../images/hamster_thumbs_up.jpg"),  # swap for your thumbs up image
    "ball": cv2.imread("../images/hamster_ball.jpeg"),
    "rock": cv2.imread("../images/spiderham.webp"),

}

def angle(a, b, c):
    # angle at point b, between a->b->c
    ax, ay = a.x - b.x, a.y - b.y
    cx, cy = c.x - b.x, c.y - b.y
    dot = ax*cx + ay*cy
    mag = (ax**2 + ay**2)**0.5 * (cx**2 + cy**2)**0.5
    if mag == 0:
        return 0
    return math.degrees(math.acos(max(-1, min(1, dot/mag))))

def is_finger_extended(landmarks, tip, pip, mcp):
    a = angle(landmarks[mcp], landmarks[pip], landmarks[tip])
    return a > 150  # close to straight

def detect_gesture(landmarks):
    index_up  = is_finger_extended(landmarks, 8, 7, 5)
    middle_up = is_finger_extended(landmarks, 12, 11, 9)
    ring_up   = is_finger_extended(landmarks, 16, 15, 13)
    pinky_up  = is_finger_extended(landmarks, 20, 19, 17)
    thumb_up  = is_finger_extended(landmarks, 4, 3, 2)

    if index_up and middle_up and not ring_up and not pinky_up:
        return "peace"
    elif thumb_up and index_up and not middle_up and not ring_up and not pinky_up:
        return "fingergun"
    elif not thumb_up and not index_up and not middle_up and not ring_up and not pinky_up:
        return "fist"
    elif thumb_up and not index_up and not middle_up and not ring_up and not pinky_up:
        return "thumbsup"
    elif thumb_up and index_up and middle_up and ring_up and pinky_up:
        return "flat"
    elif not index_up and middle_up and ring_up and pinky_up:
        return "ball"
    elif index_up and not middle_up and not ring_up and pinky_up:
        return "rock"
    
    return None

cap = cv2.VideoCapture(1)

# Create meme window
cv2.namedWindow("Meme", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Meme", 600, 800)

# Black placeholder for meme window
blank = np.zeros((400, 400, 3), dtype=np.uint8)
cv2.imshow("Meme", blank)

current_gesture = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect(mp_image)

    gesture = None

    if result.hand_landmarks:
        for hand_landmarks in result.hand_landmarks:
            points = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]

            for a, b in HAND_CONNECTIONS:
                cv2.line(frame, points[a], points[b], (0, 255, 0), 2)
            for px, py in points:
                cv2.circle(frame, (px, py), 4, (0, 0, 255), -1)

            gesture = detect_gesture(hand_landmarks)

            # Show gesture label on webcam feed
            if gesture:
                cv2.putText(frame, gesture, (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    # Only update meme window when gesture changes
    if gesture != current_gesture:
        current_gesture = gesture
        if gesture and gesture in memes:
            cv2.imshow("Meme", memes[gesture])
        else:
            cv2.imshow("Meme", blank)

    cv2.imshow("Webcam", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
