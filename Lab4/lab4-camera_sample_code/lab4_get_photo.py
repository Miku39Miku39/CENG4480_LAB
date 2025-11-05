import cv2

# initialize the camera: video0 
cap = cv2.VideoCapture(0)

# check if the camera is opened correctly
if cap.isOpened():
    print("camera is opened")
    # read the frame
    ret, frame = cap.read()
    # show the frame
    # cv2.imshow("real_time",frame)
    # save the frame
    cv2.imwrite("test.jpg", frame)

# release the camera
cap.release()
