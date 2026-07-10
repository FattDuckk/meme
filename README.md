# Hand Gesture Meme Overlay

A real-time hand gesture recognition system that detects hand poses via webcam and displays corresponding meme images in a separate window.

Built with Google MediaPipe's Tasks API and OpenCV.

## Demo

Hold up a gesture in front of your webcam — the meme window updates instantly when your pose changes.

| Gesture | Meme |
|---|---|
| ✌️ Peace sign | Hamster peace |
| 👍 Thumbs up | Hamster thumbs up |
| (thumb+pointing finger) Finger gun | Hamster with gun |
| ✊ Fist | Screaming hamster |
| 🖐️ Flat hand | Flat hamster |

## How it works

Each frame from the webcam is passed to MediaPipe's Hand Landmarker model, which returns 21 landmarks per detected hand. Gesture detection is done by measuring the angle at each finger's middle joint (PIP) — if the angle is greater than 150°, the finger is considered extended. This angle-based approach is more robust than simple y-coordinate comparison, as it works regardless of hand orientation.
### ⚠️ Press `Q` to quit/exit

## Setup

### Requirements

```bash
pip install mediapipe opencv-python
```

### Model

Download the MediaPipe hand landmarker model and place it in a `model/` folder:

```
https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
```

### Images

Add your meme images to an `images/` folder:

```
images/
  hamster_peace.webp
  hamster_thumbs_up.jpg
  hamster_gun.webp
  hamster_scream.jpg
  hamster_flat.webp
```

### Project structure

```
project/
├── model/
│   └── hand_landmarker.task
├── images/
│   └── *.webp / *.jpg
├── src/
│   └── demo.py
└── README.md
```

## Usage

```bash
python src/hand_track_hamster.py
```
or
```bash
python3 src/hand_track_hamster.py
```

- Two windows will open — the webcam feed with skeleton overlay, and the meme display window
- Press `Q` to quit

## Adding new gestures

In `detect_gesture()`, each gesture is defined by which fingers are extended. For example:

```python
# All fingers up
if thumb_up and index_up and middle_up and ring_up and pinky_up:
    return "flat"
```

Add a new entry to the `memes` dict and a new condition in `detect_gesture()` to map it.

## Adding new memes

1. Drop your image into the `images/` folder
2. Add it to the `memes` dict in `hand_track_hamster.py`:
```python
   "gesture_name": cv2.imread("../images/your_image.jpg"),
```
3. Add a condition in `detect_gesture()` that returns `"gesture_name"`

## Tech stack

- [MediaPipe Tasks API](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker/python) — hand landmark detection
- [OpenCV](https://opencv.org/) — webcam capture, drawing, windowing

## Credits
Built and documented with assistance from [Claude](https://claude.ai) (Anthropic).
