import cv2
import numpy as np
import os
import time
import pickle

from face_detection import RetinaFace
filename = 'obama.jpg'
model = 'resnet50'
# model = 'mobilenet0.25'
scale = '1'
raw_img = cv2.imread(os.path.join('../TestImg', filename))
name = 'retinaFace'
count = 0

if __name__ == "__main__":
    detector = RetinaFace()
    t0 = time.time()
    print('start')
    faces = detector(raw_img)
    t1 = time.time()
    print(f'took {round(t1 - t0, 3)} to get {len(faces)} faces')
    for box, landmarks, score in faces:
        box = box.astype(np.int)
        cropped = raw_img[box[1]:box[3], box[0]:box[2]]
        newimg = cv2.resize(cropped, (112, 112))
        count += 1
        cv2.imwrite("../cropped_face/face_" + str(count) + ".jpg", newimg)
        cv2.rectangle(raw_img, (box[0], box[1]), (box[2], box[3]), color=(255, 0, 0), thickness=1)
        print(box[0], box[1], box[2] - box[0], box[3] - box[1])
        # left = box[0]
        # top = box[1]
        # width = box[2] - box[0]
        # height = box[3] - box[1]
        # print(left, top, width, height)
        # print(box)

    font = cv2.FONT_HERSHEY_DUPLEX
    text = f'took {round(t1 - t0, 3)} to get {len(faces)} faces'
    cv2.putText(raw_img, text, (20, 20), font, 0.5, (255, 255, 255), 1)
    cv2.imwrite(os.path.join('../TestOutput', f'{name}_{model}_{scale}_{filename}'), raw_img)
    while True:
        cv2.imshow('IMG', raw_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

