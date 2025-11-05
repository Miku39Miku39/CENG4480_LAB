import cv2
import datetime
 
cap = cv2.VideoCapture(0)
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
    cv2.imshow("real_time",frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
 
cap.release()
cv2.destroyAllWindows()
