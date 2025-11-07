import cv2
import datetime
 
cap = cv2.VideoCapture(0)

desired_width = 640
desired_height = 480
cap.set(cv2.CAP_PROP_FRAME_WIDTH, desired_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, desired_height)

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
word_x = int(frame_width / 10)
word_y = int(frame_height / 10)
 
while (cap.isOpened()):
    ret,frame = cap.read()
    time_text = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    cv2.putText(frame, time_text, (word_x,word_y),
                cv2.FONT_HERSHEY_SIMPLEX,1,(55,255,155),2)
    print(frame.shape)
    resized_frame = cv2.resize(frame, (desired_width, desired_height))
    cv2.imshow("real_time",resized_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
 
cap.release()
cv2.destroyAllWindows()
