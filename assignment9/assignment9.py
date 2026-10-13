import os
import cv2

# Automatically locate cctv.mp4 in the same directory as this script file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEO_PATH = os.path.join(BASE_DIR, "cctv.mp4")

# Open the video
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise SystemExit(
        f"Could not open video at: {VIDEO_PATH}\nCheck that cctv.mp4 is in the same folder."
    )

# Initialise HOG people detector
hog = cv2.HOGDescriptor()

hog.setSVMDetector(
    cv2.HOGDescriptor_getDefaultPeopleDetector()
)

while True:

    # Read the next video frame
    success, frame = cap.read()

    if not success:
        break

    # Resize frame
    frame = cv2.resize(frame, (640, 360))

    # Detect people
    boxes, weights = hog.detectMultiScale(
        frame,
        winStride=(8, 8),
        padding=(8, 8),
        scale=1.05
    )

    # Draw bounding boxes
    for (x, y, w, h) in boxes:

        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            "Person",
            (x, max(20, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    # Display processed video
    cv2.imshow(
        "CCTV Person Detection",
        frame
    )

    # Press q to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()