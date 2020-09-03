#代码版权：黑马程序员
import cv2
import os
import sys
count=0
IMG_SAVE_PATH = './image_data'
IMG_CLASS_PATH = os.path.join(IMG_SAVE_PATH, 'circleTurn')
cap = cv2.VideoCapture(1)
while True:
    ret, frame = cap.read()
    if not ret:
        continue


    cv2.rectangle(frame, (100, 100), (500, 500), (255, 255, 255), 2)
    k = cv2.waitKey(10)
    if k == 32:
        roi = frame[100:500, 100:500]
        save_path = os.path.join(IMG_CLASS_PATH, '{}.jpg'.format(count + 1))
        cv2.imwrite(save_path, roi)
        count+=1
    if k == ord('q'):
        break

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, "Collecting {}".format(count),
            (5, 50), font, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow("Collecting images", frame)


print("\n{} image(s) saved to {}".format(count, IMG_CLASS_PATH))
cap.release()
cv2.destroyAllWindows()
