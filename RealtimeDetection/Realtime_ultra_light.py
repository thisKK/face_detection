# -*- coding: utf-8 -*-
# @Author: fyr91
# @Date:   2019-10-22 15:05:15
# @Last Modified by:   fyr91
# @Last Modified time: 2019-10-31 12:38:02
import cv2
from box_utils import *

import onnx
import onnxruntime as ort
from onnx_tf.backend import prepare

video_capture = cv2.VideoCapture('../TestVideo/maskoff.mp4')

onnx_path = '../UltraLight/models/ultra_light_640.onnx'
onnx_model = onnx.load(onnx_path)
predictor = prepare(onnx_model)
ort_session = ort.InferenceSession(onnx_path)
input_name = ort_session.get_inputs()[0].name

while True:
    ret, frame = video_capture.read()
    if frame is not None:
        h, w, _ = frame.shape

        # preprocess img acquired
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # convert bgr to rgb
        img = cv2.resize(img, (640, 480)) # resize
        img_mean = np.array([127, 127, 127])
        img = (img - img_mean) / 128
        img = np.transpose(img, [2, 0, 1])
        img = np.expand_dims(img, axis=0)
        img = img.astype(np.float32)

        confidences, boxes = ort_session.run(None, {input_name: img})
        boxes, labels, probs = predict(w, h, confidences, boxes, 0.7)

        for i in range(boxes.shape[0]):
            box = boxes[i, :]
            x1, y1, x2, y2 = box
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (80,18,236), 2)
        cv2.imshow('Video', frame)

        # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
