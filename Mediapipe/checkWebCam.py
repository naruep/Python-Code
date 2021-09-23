import cv2

# Initialize the VideoCapture object to read from the webcam.
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    
    cv2.imshow('checkWebCam',frame)
    
# Wait for 1ms. If a a key is pressed, retreive the ASCII code of the key.
    k = cv2.waitKey(1) & 0xFF

    # Check if 'ESC' is pressed and break the loop.
    if (k == 27):
        break

# Release the VideoCapture Object and close the windows.
cv2.moveWindow('checkWebCam', 0, 0)
cap.release()
