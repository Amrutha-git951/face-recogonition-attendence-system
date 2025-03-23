import cv2

# Check OpenCV version
print(cv2.__version__)

# Create LBPH face recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()
