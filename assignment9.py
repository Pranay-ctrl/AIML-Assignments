import cv2

# Video filename
VIDEO_PATH = "cctv.mp4"

# Open the video
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise SystemExit(
        "Could not open video. Check the filename and path."
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