from tensorflow.keras.models import load_model
import cv2
import numpy as np
from random import choice
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont 



REV_CLASS_MAP = {
    0: "环岛转弯",
    1: "限速30",
    2: "禁止停车",
    3: "未识别"
}


def mapper(val):
    return REV_CLASS_MAP[val]




model = load_model("detect-sign-model.h5")

cap = cv2.VideoCapture(1)

prev_move = None

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    # rectangle for user to play
    cv2.rectangle(frame, (100, 100), (500, 500), (255, 255, 255), 2)
    # rectangle for computer to play

    # extract the region of image within the user rectangle
    roi = frame[100:500, 100:500]
    img = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (227, 227))
    

    # predict the move made
    pred = model.predict(np.array([img/1.0]))
    print(pred)
    move_code = np.argmax(pred[0])
    #print(pred)
    user_move_name = mapper(move_code)

    font = cv2.FONT_HERSHEY_SIMPLEX
    #cv2.putText(frame, "Your Move: " + user_move_name,
    #            (50, 50), font, 1.2, (255, 255, 255), 2, cv2.LINE_AA)

    fontpath = "./simsun.ttc"  # 宋体字体文件
    font_1 = ImageFont.truetype(fontpath, 45)  # 加载字体, 字体大小
    img_pil = Image.fromarray(frame)
    draw = ImageDraw.Draw(img_pil)
    draw.text((50,60), user_move_name, font=font_1, fill=(0, 255, 255))
    frame = np.array(img_pil)

    cv2.imshow("itheima", frame)

    k = cv2.waitKey(10)
    if k == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
