
import cv2
import mediapipe as mp
import time

# MediaPipe configuration
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = PoseLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="pose_landmarker_lite.task"
    ),
    running_mode=RunningMode.VIDEO,
    num_poses=1
)

# Body connections: shoulder to elbow, etc.
CONNECTIONS = [
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
    (11, 23), (12, 24),
    (23, 24),
    (23, 25), (25, 27),
    (24, 26), (26, 28),
    (15, 17), (15, 19), (15, 21),
    (16, 18), (16, 20), (16, 22),
    (27, 29), (27, 31),
    (28, 30), (28, 32)
]

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Could not open webcam")

with PoseLandmarker.create_from_options(options) as landmarker:
    start_time = time.monotonic()

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # Mirror the webcam
        frame = cv2.flip(frame, 1)

        height, width = frame.shape[:2]

        # OpenCV uses BGR, MediaPipe uses RGB
        rgb_frame = cv2.cvtColor(
            frame, cv2.COLOR_BGR2RGB
        )

        # Convert the image to MediaPipe format
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        # Each video frame needs a timestamp
        timestamp_ms = int(
            (time.monotonic() - start_time) * 1000
        )

        # Detect pose landmarks
        result = landmarker.detect_for_video(
            mp_image, timestamp_ms
        )

        # Draw skeleton if a person was detected
        if result.pose_landmarks:
            for pose in result.pose_landmarks:
                points = []

                for landmark in pose:
                    x = int(landmark.x * width)
                    y = int(landmark.y * height)
                    points.append((x, y))

                # Draw skeleton connections
                for start, end in CONNECTIONS:
                    if (
                        pose[start].visibility > 0.5
                        and pose[end].visibility > 0.5
                    ):
                        cv2.line(
                            frame,
                            points[start],
                            points[end],
                            (0, 255, 0),
                            2
                        )

                # Draw joint markers
                for i, point in enumerate(points):
                    if pose[i].visibility > 0.5:
                        cv2.circle(
                            frame,
                            point,
                            4,
                            (0, 0, 255),
                            -1
                        )

        cv2.imshow("MediaPipe Pose", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
