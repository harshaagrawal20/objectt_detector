import cv2

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()
    
ret, frame = cap.read()
if ret:
    print("Camera is working")
    cv2.imshow('test', frame)
    cv2.waitKey(0)
cap.release()
cv2.destroyAllWindows()