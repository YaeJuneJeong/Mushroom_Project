# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'untitled.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.
import collections
import datetime
import os
import random

import tensorflow as tf
import cv2
from PIL import Image
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import pyqtSlot
from object_detection.builders import model_builder
from object_detection.utils import label_map_util, config_util
import numpy as np
from object_detection.utils import visualization_utils as viz_utils

PATH_TO_MODEL_DIR = 'C:/Users/jyj98/tensorflow/workspace/training_demo/exported-models/mushroom_model1'
PATH_TO_CFG = PATH_TO_MODEL_DIR + "/pipeline.config"
PATH_TO_CKPT = PATH_TO_MODEL_DIR + "/checkpoint"
PATH_TO_LABELS = 'C:/Users/jyj98/tensorflow/workspace/training_demo/annotations/label_map.pbtxt'
PATH_TO_SAVED_MODEL = PATH_TO_MODEL_DIR + "/saved_model"
PATH_TO_IMG = 'C:/Users/jyj98/tensorflow/workspace/training_demo/images/train/Mushroom.jpg'

# url_register = "http://184.73.45.24/api/myfarm/register/ip"
# url_info = "http://184.73.45.24/api/myfarm/info"

# Load pipeline config and build a detection model
configs = config_util.get_configs_from_pipeline_file(PATH_TO_CFG)
model_config = configs['model']
detection_model = model_builder.build(model_config=model_config, is_training=False)

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(PATH_TO_CKPT, 'ckpt-0')).expect_partial()
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)


@tf.function
def detect_fn(image):
    """Detect objects in image."""

    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)

    return detections


def detection(img):
    detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)
    input_tensor = tf.convert_to_tensor(img)
    input_tensor = input_tensor[tf.newaxis, ...]
    detections = detect_fn(input_tensor)
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}
    detections['num_detections'] = num_detections
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    return detections


def load_image_into_numpy_array(path):
    """Load an image from file into a numpy array.

    Puts image into numpy array to feed into tensorflow graph.
    Note that by convention we put it into a numpy array with shape
    (height, width, channels), where channels=3 for RGB.

    Args:
      path: the file path to the image

    Returns:
      uint8 numpy array with shape (img_height, img_width, 3)
    """
    return np.array(Image.open(path))


def get_shiitake_location(boxes, scores, min_score_thresh, max_boxes_to_draw=20):
    box_to_color_map = collections.defaultdict(str)
    if not max_boxes_to_draw:
        max_boxes_to_draw = boxes.shape[0]
    for i in range(boxes[0].shape[0]):
        if max_boxes_to_draw == len(box_to_color_map):
            break
        if scores[i] > min_score_thresh:
            box = tuple(boxes[i].tolist())
            box_to_color_map[box] = 'size'

    return box_to_color_map


def get_size(boxes):
    return random.randrange(1, 8)


class Ui_Dialog(object):
    def __init__(self):
        super(Ui_Dialog, self).__init__()
        self.currentPictures = ('de0.jpg', 'de90.jpg', 'de180.jpg', 'de270.jpg')
        self.current = 0
        self.collect = 0

    def setupUi(self, Dialog):

        Dialog.setObjectName("Dialog")
        Dialog.resize(1024, 600)
        Dialog.setMaximumSize(QtCore.QSize(1024, 600))
        self.pushButton = QtWidgets.QPushButton(Dialog)
        self.pushButton.setGeometry(QtCore.QRect(860, 530, 111, 41))
        self.pushButton.setObjectName("pushButton")
        self.total_num = QtWidgets.QLabel(Dialog)
        self.total_num.setGeometry(QtCore.QRect(580, 400, 111, 71))
        font = QtGui.QFont()
        font.setPointSize(20)
        self.total_num.setFont(font)
        self.total_num.setTextFormat(QtCore.Qt.PlainText)
        self.total_num.setAlignment(QtCore.Qt.AlignCenter)
        self.total_num.setObjectName("total_num")
        self.gather_num = QtWidgets.QLabel(Dialog)
        self.gather_num.setGeometry(QtCore.QRect(830, 400, 111, 71))
        font = QtGui.QFont()
        font.setPointSize(20)
        self.gather_num.setFont(font)
        self.gather_num.setTextFormat(QtCore.Qt.AutoText)
        self.gather_num.setAlignment(QtCore.Qt.AlignCenter)
        self.gather_num.setObjectName("gather_num")
        self.label_12 = QtWidgets.QLabel(Dialog)
        self.label_12.setGeometry(QtCore.QRect(830, 310, 111, 71))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.label_12.setFont(font)
        self.label_12.setTextFormat(QtCore.Qt.AutoText)
        self.label_12.setAlignment(QtCore.Qt.AlignCenter)
        self.label_12.setObjectName("label_12")
        self.label_13 = QtWidgets.QLabel(Dialog)
        self.label_13.setGeometry(QtCore.QRect(580, 310, 111, 71))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.label_13.setFont(font)
        self.label_13.setTextFormat(QtCore.Qt.AutoText)
        self.label_13.setAlignment(QtCore.Qt.AlignCenter)
        self.label_13.setObjectName("label_13")
        self.mushroom_picture = QtWidgets.QLabel(Dialog)
        self.mushroom_picture.setGeometry(QtCore.QRect(620, 20, 291, 141))
        self.mushroom_picture.setTextFormat(QtCore.Qt.AutoText)
        self.mushroom_picture.setAlignment(QtCore.Qt.AlignCenter)
        self.mushroom_picture.setObjectName("mushroom_picture")
        self.pushButton_2 = QtWidgets.QPushButton(Dialog)
        self.pushButton_2.setGeometry(QtCore.QRect(660, 530, 81, 41))
        font = QtGui.QFont()
        font.setFamily("Agency FB")
        font.setPointSize(22)
        self.pushButton_2.setFont(font)
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_3 = QtWidgets.QPushButton(Dialog)
        self.pushButton_3.setGeometry(QtCore.QRect(570, 530, 81, 41))
        font = QtGui.QFont()
        font.setFamily("Agency FB")
        font.setPointSize(22)
        self.pushButton_3.setFont(font)
        self.pushButton_3.setObjectName("pushButton_3")
        self.file_name = QtWidgets.QLabel(Dialog)
        self.file_name.setGeometry(QtCore.QRect(560, 200, 421, 71))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.file_name.setFont(font)
        self.file_name.setTextFormat(QtCore.Qt.AutoText)
        self.file_name.setAlignment(QtCore.Qt.AlignCenter)
        self.file_name.setObjectName("label_14")
        self.label_15 = QtWidgets.QLabel(Dialog)
        self.label_15.setGeometry(QtCore.QRect(600, 470, 111, 51))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.label_15.setFont(font)
        self.label_15.setTextFormat(QtCore.Qt.AutoText)
        self.label_15.setAlignment(QtCore.Qt.AlignCenter)
        self.label_15.setObjectName("label_15")
        self.label_16 = QtWidgets.QLabel(Dialog)
        self.label_16.setGeometry(QtCore.QRect(60, 80, 111, 71))
        font = QtGui.QFont()
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.label_16.setFont(font)
        self.label_16.setTextFormat(QtCore.Qt.AutoText)
        self.label_16.setAlignment(QtCore.Qt.AlignCenter)
        self.label_16.setObjectName("label_16")
        self.time = QtWidgets.QLabel(Dialog)
        self.time.setGeometry(QtCore.QRect(250, 80, 171, 71))
        font = QtGui.QFont()
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.time.setFont(font)
        self.time.setTextFormat(QtCore.Qt.AutoText)
        self.time.setAlignment(QtCore.Qt.AlignCenter)
        self.time.setObjectName("label_17")
        self.temp = QtWidgets.QLabel(Dialog)
        self.temp.setGeometry(QtCore.QRect(250, 230, 171, 71))
        font = QtGui.QFont()
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.temp.setFont(font)
        self.temp.setTextFormat(QtCore.Qt.AutoText)
        self.temp.setAlignment(QtCore.Qt.AlignCenter)
        self.temp.setObjectName("temp")
        self.label_19 = QtWidgets.QLabel(Dialog)
        self.label_19.setGeometry(QtCore.QRect(60, 230, 111, 71))
        font = QtGui.QFont()
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.label_19.setFont(font)
        self.label_19.setTextFormat(QtCore.Qt.AutoText)
        self.label_19.setAlignment(QtCore.Qt.AlignCenter)
        self.label_19.setObjectName("label_19")
        self.hum = QtWidgets.QLabel(Dialog)
        self.hum.setGeometry(QtCore.QRect(250, 430, 171, 71))
        font = QtGui.QFont()
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.hum.setFont(font)
        self.hum.setTextFormat(QtCore.Qt.AutoText)
        self.hum.setAlignment(QtCore.Qt.AlignCenter)
        self.hum.setObjectName("hum")
        self.label_21 = QtWidgets.QLabel(Dialog)
        self.label_21.setGeometry(QtCore.QRect(60, 430, 111, 71))
        font = QtGui.QFont()
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.label_21.setFont(font)
        self.label_21.setTextFormat(QtCore.Qt.AutoText)
        self.label_21.setAlignment(QtCore.Qt.AlignCenter)
        self.label_21.setObjectName("label_21")
        self.verticalLayoutWidget = QtWidgets.QWidget(Dialog)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(-10, 0, 521, 641))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.verticalLayoutWidget_2 = QtWidgets.QWidget(Dialog)
        self.verticalLayoutWidget_2.setGeometry(QtCore.QRect(510, 0, 521, 641))
        self.verticalLayoutWidget_2.setObjectName("verticalLayoutWidget_2")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.verticalLayoutWidget_2)
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_2.setObjectName("verticalLayout_2")

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

        self.mushroom_picture.resize(320, 240)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.pushButton.setText(_translate("Dialog", "갱신하기"))
        self.total_num.setText(_translate("Dialog", "N 개"))
        self.gather_num.setText(_translate("Dialog", "N 개"))
        self.label_12.setText(_translate("Dialog", "수확 대상"))
        self.label_13.setText(_translate("Dialog", "현재 버섯수"))
        self.pushButton_2.setText(_translate("Dialog", "→"))
        self.pushButton_3.setText(_translate("Dialog", "←"))
        self.file_name.setText(_translate("Dialog", "file_name"))
        self.label_15.setText(_translate("Dialog", "사진 이동"))
        self.label_16.setText(_translate("Dialog", "갱신 시간"))
        self.time.setText(_translate("Dialog", "갱신 시간"))
        self.temp.setText(_translate("Dialog", "-도"))
        self.label_19.setText(_translate("Dialog", "온도"))
        self.hum.setText(_translate("Dialog", "-도"))
        self.label_21.setText(_translate("Dialog", "갱신 시간"))
        self.change_picture()

    def change_picture(self, current=0):
        file_name = self.currentPictures[self.current]
        self.file_name.setText(file_name)
        current_img = load_image_into_numpy_array(PATH_TO_IMG)
        detections = detection(current_img)
        viz_utils.visualize_boxes_and_labels_on_image_array(current_img, detections['detection_boxes'],
                                                            detections['detection_classes'],
                                                            detections['detection_scores'], category_index,
                                                            use_normalized_coordinates=True,
                                                            max_boxes_to_draw=200,
                                                            min_score_thresh=0.5,
                                                            agnostic_mode=False)
        current_img = cv2.resize(current_img, dsize=(280, 210), interpolation=cv2.INTER_AREA)
        qImg = QtGui.QImage(current_img.data, 280, 210, QtGui.QImage.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(qImg)
        self.mushroom_picture.setPixmap(pixmap)

        boxes = get_shiitake_location(detections['detection_boxes'], detections['detection_classes'],
                                      0.5)

        self.total_num.setText(str(len(boxes)-1))
        self.collect = 0
        for box in boxes:
            size = get_size(box)
            print(size)
            if size > 6:
                self.collect += 1
        self.gather_num.setText(str(self.collect))

    def left_click(self):
        self.current += 1
        self.change_picture(self.current)

    @pyqtSlot(str, str)
    def renewal(self, arg1, arg2):
        self.temp.setText(arg1)
        self.hum.setText(arg2)
        self.time.setText(str(datetime.datetime.now()))
        # self.data1.repaint(arg)
        # self.data2.repaint()
        # self.layout.repaint()
        print(arg1, arg2)


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    Dialog = QtWidgets.QDialog()
    ui = Ui_Dialog()
    ui.setupUi(Dialog)
    Dialog.show()
    sys.exit(app.exec_())
