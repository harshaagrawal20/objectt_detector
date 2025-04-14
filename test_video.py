import cv2

video_path = "test_video/5446310-hd_1920_1080_30fps.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error: Could not open video file: {video_path}")
else:
    ret, frame = cap.read()
    if ret:
        print("Successfully read first frame")
        cv2.imshow('First Frame', frame)
        cv2.waitKey(0)
    else:
        print("Error: Could not read frame from video")

cap.release()
cv2.destroyAllWindows()