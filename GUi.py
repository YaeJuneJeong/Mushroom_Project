import base64
import collections
import datetime
import json
import random
import threading
import time
import os
import serial
import socketio
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QThread, pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import QMessageBox
import requests
import socket
import cv2
import pyrealsense2 as rs
import numpy as np
from PIL import Image
import tensorflow as tf
from object_detection.builders import model_builder
from object_detection.utils import label_map_util, config_util
from object_detection.utils import visualization_utils as viz_utils

PATH_TO_MODEL_DIR = 'C:/Users/LattePanda/tensorflow/workspace/training_demo/exported-models/mushroom_model1'
PATH_TO_CFG = PATH_TO_MODEL_DIR + "/pipeline.config"
PATH_TO_CKPT = PATH_TO_MODEL_DIR + "/checkpoint"
PATH_TO_LABELS = 'C:/Users/LattePanda/tensorflow/workspace/training_demo/annotations/label_map.pbtxt'
PATH_TO_SAVED_MODEL = PATH_TO_MODEL_DIR + "/saved_model"
PATH_TO_IMG = 'C:/Users/jyj98/tensorflow/workspace/training_demo/images/train'

url_register = "http://184.73.45.24/api/myfarm/register/ip"
url_info = "http://184.73.45.24/api/myfarm/info"

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


# input ymin,xmin,ymax,xmax
def make_data(prgID, range, boxes, size):
    location = {
        'rotation': str(range),
        'y': str(boxes[0]),
        'x': str(boxes[1]),
        'height': str(abs(boxes[2] - boxes[0])),
        'width': str(abs(boxes[3] - boxes[1]))
    }
    location_json = json.dumps(location)
    data = {'prgId': prgID, 'metaJSON': location_json, 'size': size}
    response = requests.post(url_register, data=data)
    return response.status_code


s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.connect(('8.8.8.8', 1))  # connect() for UDP doesn't send packets
local_ip_address = s.getsockname()[0]

pipeline_check = False
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# socket 서버 연결
sio = socketio.Client()
sio.connect('http://localhost:3001')

stream_end_value = False


@sio.on("req_cosdata")
def socket_data(temp=0, hum=0):
    sio.emit("res_cosdata", {"temperature": temp, "humidity": hum})


@sio.on("req_video")
def socket_stream(req):
    global stream_end_value
    print("web socket req_video connect")
    if req == 'disconnect':
        print("web socket req_video disconnect")
        stream_end_value = True
        return
    t = threading.Thread(target=stream_thread)
    t.start();


def stream_thread():
    global pipeline_check
    global pipeline
    global config
    global stream_end_value

    if pipeline_check == False:
        pipeline_check = True
        profile = pipeline.start(config)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
        print("Web socket on")

        while True:
            time.sleep(0.2)
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()

            # Convert images to numpy arrays
            color_image = np.asanyarray(color_frame.get_data())

            result, frame = cv2.imencode('.jpg', color_image, encode_param)
            data = base64.b64encode(frame).decode('utf-8')
            sio.emit('res_video', data)
            if stream_end_value:
                pipeline.stop()
                print("Web socket disconnet")
                stream_end_value = False
                break
        pipeline_check = False
        return
    else:
        return


SERVER_URL = 'http://184.73.45.24/api'  # 서버 url
PIN = '107512'  # 기기 고유 핀번호

R = "R"  # 환경 데이터 프로토콜
C = "C"  # 온도 프로토콜
S = "S"  # 습도 프로토콜
WATER_ORDER = 'W'  # 물주기 프로토콜
MOTER_ORDER = 'M'  # 3D 촬영 프로토콜

DAY = 86400
HOUR = 3600  # 온도 데이터 전송 시간 기준값
WATER_TIME = 0  # 물주는 시간 기준값
MOTER_TIME = HOUR * 3  # 모터 시간 기준값
D2_TIME = 1000

water_num = 0

data = None  # 환경 데이터 전역
title = 'title'
file_path = 'D:/example1.jpg'

prg_id = 0

RUNNING = False


def drawline(img, pt1, pt2, color, thickness=3, style='dotted', gap=30):
    dist = ((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2) ** .5
    pts = []
    for i in np.arange(0, dist, gap):
        r = i / dist
        x = int((pt1[0] * (1 - r) + pt2[0] * r) + .5)
        y = int((pt1[1] * (1 - r) + pt2[1] * r) + .5)
        p = (x, y)
        pts.append(p)

    if style == 'dotted':
        for p in pts:
            cv2.circle(img, p, thickness, color, -1)
    else:
        s = pts[0]
        e = pts[0]
        i = 0
        for p in pts:
            s = e
            e = p
            if i % 2 == 1:
                cv2.line(img, s, e, color, thickness)
            i += 1


def drawpoly(img, pts, color, thickness=1, style='dotted', ):
    s = pts[0]
    e = pts[0]
    pts.append(pts.pop(0))
    for p in pts:
        s = e
        e = p
        drawline(img, s, e, color, thickness, style)


def drawrect(img, pt1, pt2, color, thickness=1, style='dotted'):
    pts = [pt1, (pt2[0], pt1[1]), pt2, (pt1[0], pt2[1])]
    drawpoly(img, pts, color, thickness, style)


# 시작전 polling 코드
class Start_before(QThread):
    errors = pyqtSignal(int)
    signal = pyqtSignal(str, str)
    finished = pyqtSignal()
    global data
    global water_num
    global prg_id
    global pipeline

    def __init__(self, parent=None):
        super().__init__()
        self.main = parent
        self.isRun = False

    def run(self):
        try:
            machien = {'id': 1, 'ip': 'http://192.168.0.10'}
            response_machien = requests.put(SERVER_URL + "/myfarm/register/ip", data=machien)

            print(f" machien ip set server : {response_machien.status_code}")

            # GUI 코드
            while True:
                response_user = requests.get(SERVER_URL + "/user/info/3")

                print(f" user data : {response_user.status_code}")

                if response_user.status_code == 200:
                    break
                time.sleep(5)

            # while True :
            # time.sleep(1)
            # print("exist check ...")
            # pipeline.start()
            # frames = pipeline.wait_for_frames()
            # color_frame = frames.get_color_frame()
            # color_image = np.asarray(color_frame.get_data())

            ## take the only location of mushroom pot -> 1/3 * width,1/2*height
            # recent_image = color_image[100:350, 290:550]
            # check_image = cv2.imread('./check.jpg')[100:350, 290:550]
            # cv2.imwrite('./rec.jpg',check_image)
            # cv2.imwrite('./recent.jpg',recent_image)
            # hist_recent = cv2.calcHist(recent_image, [0,1], None, [180,256], [0,180,0,256])
            # hist_check = cv2.calcHist(check_image, [0,1], None, [180,256], [0,180,0,256])
            # number = cv2.compareHist(hist_recent, hist_check, cv2.HISTCMP_CORREL)

            # print(number)
            # pipeline.stop()
            # if number > 0.4:
            # print('Not exist')

            # else:
            #            배지입력 확인
            # print("Exist !!")
            # break

            while self.isRun:
                params = {'pin': PIN}
                response = requests.get(
                    SERVER_URL + '/farm/exist',
                    params=params
                )

                if response.status_code == 200:
                    prg_id = response.text
                    break
                else:
                    print("Not prg")
                time.sleep(5)  # 429 에러 방지

            params = {'id': prg_id, 'type': 'custom'}
            response = requests.get(SERVER_URL + '/farm/data', params=params)
            result = json.loads(response.text)
            water_num = result['water']
            temp_array = result['temperature']
            hum_array = result['humidity']

            params_status = {'id': "1", "status": "true"}
            response_status = requests.put(SERVER_URL + '/myfarm/status', params=params_status)

            print(f"machien status : {response_status.status_code}")

            data_len = len(temp_array)

            data = [None] * data_len

            for i in range(0, data_len):
                temp_array[i] = str(temp_array[i]['setting_value'])
                hum_array[i] = str(hum_array[i]['setting_value'])
                data[i] = R + C + temp_array[i] + S + hum_array[i]

            print(f"water_num : {water_num}")
            print(f"temp_array : {temp_array}")
            print(f"hum_array : {hum_array}")
            print(f"total_data : {data}")

            # return True if data else False

            # HOUR * 24 / water_num
            WATER_TIME = HOUR * 24 / water_num

            serial_send_len = 0
            hour = 0  # 시간 초로 변환
            # WATER_TIME
            water_time = 10000  # 값 받아 오면 연산할 물주기 시간
            # MOTER_TIME
            moter_time = MOTER_TIME
            picTime = 100000

            now = datetime.datetime.now()
            Arduino = serial.Serial(port='COM5', baudrate=9600)
            print(f"data success : {data}")

            seconds = 0
            dt2 = None
            loadTime = 0
            hum = 0
            temp = 0

            while self.isRun:
                dt1 = datetime.datetime.now()
                result = dt1 - now
                seconds = int(result.total_seconds()) - loadTime

                print(f"Python hour : {hour}, water_time : {water_time} moter_time : {moter_time} image : {picTime}")
                print(f"Python seconds : {seconds}")

                if Arduino.readable():
                    LINE = Arduino.readline()
                    code = str(LINE.decode().replace('\n', ''))
                    print(code)
                    hum = code[18: 20]
                    temp = code[38: 40]
                    socket_data(temp, hum)
                    self.signal.emit(str(temp), str(hum))
                if seconds - 2 <= hour <= seconds + 2:

                    if len(data) - 1 < serial_send_len:
                        response_status_prg = requests.put(SERVER_URL + f'/farm/end?id={prg_id}')
                        print(f"prg status : {response_status_prg.status_code}")

                        params_status = {'id': "1", "status": "false"}
                        response_status = requests.put(SERVER_URL + '/myfarm/status', params=params_status)
                        print(f"machien status : {response_status.status_code}")
                        break

                    Arduino.write(data[serial_send_len].encode('utf-8'))
                    req_data_humi_temp = {'prgId': prg_id, 'tempValue': temp, 'humiValue': hum}
                    humi_temp_res = requests.post(SERVER_URL + "/data/add", data=req_data_humi_temp);
                    print(f"Python res_temp : {humi_temp_res.status_code}")
                    serial_send_len += 1
                    hour += DAY

                if seconds - 2 <= water_time <= seconds + 2:
                    Arduino.write(WATER_ORDER.encode('utf-8'))
                    dt2 = datetime.datetime.now()

                    while Arduino.readable():
                        LINE = Arduino.readline()
                        code = str(LINE.decode().replace('\n', ''))
                        print(code)
                        if code[0: 3] == 'end':
                            loadTime += int((datetime.datetime.now() - dt2).total_seconds())
                            break

                    water_time += WATER_TIME

                if seconds - 2 <= moter_time - HOUR / 3 <= seconds + 2:

                    if pipeline_check == False:
                        pipeline_check = True
                        Arduino.write(MOTER_ORDER.encode('utf-8'))
                        dt2 = datetime.datetime.now()

                        # 3d config
                        device = rs.context().query_devices()[0]
                        advnc_mode = rs.rs400_advanced_mode(device)
                        depth_table_control_group = advnc_mode.get_depth_table()
                        depth_table_control_group.disparityShift = 60
                        depth_table_control_group.depthClampMax = 4000
                        depth_table_control_group.depthClampMin = 0
                        advnc_mode.set_depth_table(depth_table_control_group)

                        pipeline1 = rs.pipeline()
                        config1 = rs.config()

                        config1.enable_stream(rs.stream.depth, rs.format.z16, 30)
                        config1.enable_stream(rs.stream.color, rs.format.bgr8, 30)

                        # Start streaming
                        profile1 = pipeline1.start(config)
                        # Get stream profile and camera intrinsics
                        profile = pipeline1.get_active_profile()
                        depth_profile = rs.video_stream_profile(profile1.get_stream(rs.stream.depth))
                        pc = rs.pointcloud()
                        decimate = rs.decimation_filter()
                        decimate.set_option(rs.option.filter_magnitude, 1)

                        file_name_3d = [0, 90, 180, 270]
                        file_names_3d = ['./3dScan{}.ply'.format(i) for i in file_name_3d]

                        i = 0

                        while Arduino.readable():
                            LINE = Arduino.readline()
                            code = str(LINE.decode().replace('\n', ''))
                            print(code)
                            print(f"i : {i}")
                            if code[0: 3] == 'end':
                                loadTime += int((datetime.datetime.now() - dt2).total_seconds())
                                break
                            take_3Dpicture(file_names_3d[i], pipeline1, decimate, pc)
                            i += 0 if i >= 3 else 1

                        files = {'ply': open(file_names_3d[0], 'rb')}
                        req_data_3d = [("machineid", 1)]
                        res_3d = requests.post(SERVER_URL + "/upload/ply", files=files, data=req_data_3d);
                        print(f"Python res_3d : {res_3d.status_code}")

                        pipeline1.stop()
                        moter_time += MOTER_TIME
                        pipeline_check = False
                    else:
                        moter_time += 50

                if seconds - 2 <= picTime + 30 <= seconds + 2:

                    if pipeline_check == False:
                        pipeline_check = True
                        profile = pipeline.start(config)

                        depth_sensor = profile.get_device().first_depth_sensor()

                        frames = pipeline.wait_for_frames()
                        color_frame = frames.get_color_frame()

                        # Convert images to numpy arrays
                        color_image = np.asanyarray(color_frame.get_data())
                        cv2.imwrite('./color_sensor.jpg', color_image)

                        files = {'compost': ('compost.jpg', open('./color_sensor.jpg', 'rb'))}
                        data1 = [('machineid', 1)]
                        response_image = requests.post(SERVER_URL + "/upload/compost", files=files, data=data1)
                        print(f"Python res_image : {response_image.status_code}")
                        pipeline.stop()
                        picTime += D2_TIME

                        # Turning moter

                        #  MUSHROOM DETECTION , requeset
                        frames = pipeline.wait_for_frames()
                        color_frame = frames.get_color_frame()
                        color_image = np.asanyarray(color_frame.get_data())
                        detections = detection(color_image)
                        boxes = get_shiitake_location(detections['detection_boxes'], detections['detection_classes'],
                                                      0.5)
                        print(boxes)
                        pipeline_check = False
                        pipeline.stop()

                        for box in boxes:
                            size = get_size(box)
                            res = make_data(52, box, size)
                            print(res)
                    else:
                        picTime += 50


        except Exception:
            self.errors.emit(100)
            self.isRun = False


def refresh_data():
    refresh_url = SERVER_URL + '/api/myfarm/data/hour'
    data = {'id': 2, 'ip': local_ip_address}
    response = requests.put(refresh_url, data=data, timeout=10)


def encode_serial_data(str):
    return str.encode('utf-8')


def take_3Dpicture(name, pipeline, decimate, pc):
    try:

        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        depth_frame = decimate.process(depth_frame)
        color_image = np.asanyarray(color_frame.get_data())

        mapped_frame, color_source = color_frame, color_image
        points = pc.calculate(depth_frame)
        pc.map_to(mapped_frame)

        points.export_to_ply(name, mapped_frame)

    except Exception as e:
        print(e)


# 기기 가동 코드
# class Start(QThread):
#     error = pyqtSignal(int)
#     singal = pyqtSignal(str, str)
#     finished = pyqtSignal()
#
#     global water_num
#     global data
#     global D2_TIME
#     global pipeline
#     global config
#     global pipeline_check
#     global prg_id
#     global DAY
#
#     def __init__(self, parent=None):
#         super().__init__()
#         self.main = parent
#         self.isRun = False
#
#     def run(self):
#         try:
#
#
#
#                 # 끝내기 데이터 오면 break 후 리턴
#                 self.finished.emit()
#         except Exception:
#             self.error.emit(100)
#             self.isRun = False
#

class Send(QThread):
    error = pyqtSignal(int)
    info = pyqtSignal(dict)
    finished = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__()
        self.main = parent
        self.isRun = False

    def run(self):
        try:
            while self.isRun:
                # change ip address
                data = {'id': 2, 'ip': local_ip_address}
                response = requests.put(url_register, data=data, timeout=10)
                print(response.status_code)

                if response.status_code != 200:
                    raise Exception

                # response = requests.get(url_info, params={'id': 2})
                # print(response.json())
                # if (response.json()['machine_userid']) is not 2:
                #     self.info.emit(response.json())
                #     break
                # time.sleep(10)

                # user_id
                # response.json()['machine_userid']
                pipeline_check = True
                profile = pipeline.start(config)
                frames = pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                color_image = np.asarray(color_frame.get_data())
                cv2.imwrite('./recent.jpg', color_image)
                self.isRun = False
                self.finished.emit()
                break
        except Exception:
            self.error.emit(100)
            self.isRun = False

        finally:
            pipeline.stop()

class play(QThread):
    def __init__(self, parent=None):
        super().__init__()
        self.main = parent
        self.isRun = False

    def run(self):
        try:
            profile = pipeline.start(config)
            while self.isRun:

                frames = pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                color_image = np.asarray(color_frame.get_data())

        except Exception:
            self.error.emit(100)
            self.isRun = False

        finally:
            pipeline.stop()

# class RotateMe(QtWidgets.QLabel, QThread):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self._pixmap = QtGui.QPixmap()
#         self.state = False
#         self._animation = QtCore.QVariantAnimation(
#             self,
#             startValue=0.0,
#             endValue=360.0,
#             duration=1000,
#             valueChanged=self.on_valueChanged
#         )
#         self._animation.setLoopCount(-1)
#
#     def set_pixmap(self, pixmap):
#         self._pixmap = pixmap
#         self.setPixmap(self._pixmap)
#
#     def start_animation(self):
#         if self.state is False:
#             self._animation.start()
#             self.state = True
#
#     def stop_animations(self):
#         if self.state is True:
#             self._animation.stop()
#             self.state = False
#
#     def on_valueChanged(self, value):
#         t = QtGui.QTransform()
#         t.rotate(value)
#         self.setPixmap(self._pixmap.transformed(t))


class Window1(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(Window1, self).__init__(parent)

        self.setFixedHeight(600)
        self.setFixedWidth(1024)
        # self.secondWindow = Window2()
        self.label = QtWidgets.QLabel(alignment=QtCore.Qt.AlignCenter)
        self.movie = QtGui.QMovie('./giphy.gif')
        self.label.setMovie(self.movie)
        self.movie.start()
        self.movie.stop()
        self.send = Send(self)
        self.send.finished.connect(self.stop)
        self.send.finished.connect(self.change_stack)
        # self.send.info.connect(self.get_info)
        self.send.error.connect(self.error)

        button = QtWidgets.QPushButton('ip 저장')
        # button.clicked.connect(self.label.start_animation)
        button.clicked.connect(self.start)

        lay = QtWidgets.QVBoxLayout(self)
        lay.addWidget(self.label)
        lay.addWidget(button)

    def start(self):
        if not self.send.isRun:
            self.send.isRun = True
            self.send.start()
            self.movie.start()

    def stop(self):
        if self.send.isRun:
            self.send.isRun = False
            self.movie.stop()

    def error(self):
        self.send.isRun = False
        self.movie.stop()
        QMessageBox.information(self, 'Error', 'Please check Network')

    def change_stack(self):
        self.parent().stack.setCurrentIndex(1)

    def get_info(self):
        self.parent().change(data)


class Window2(QtWidgets.QWidget):

    def __init__(self, parent=None):
        super(Window2, self).__init__(parent)

        self.layout = QtWidgets.QVBoxLayout(self)
        self.label = QtWidgets.QLabel()
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.resize(320, 240)
        self.layout.addWidget(self.label)

        self.button1 = QtWidgets.QPushButton('영상 시작')
        # self.button1.clicked.connect(self.start)
        self.layout.addWidget(self.button1)

        self.button2 = QtWidgets.QPushButton('배지 확인')
        self.button2.clicked.connect(self.check)
        self.layout.addWidget(self.button2)

        self.win = QtWidgets.QWidget()
        self.take_picture = False

        self.color_frame = None

    def play(self):
        try:
            profile = pipeline.start(config)
            while self.take_picture:
                frames = pipeline.wait_for_frames()
                self.color_frame = frames.get_color_frame()
                color_image = np.asarray(self.color_frame.get_data())
                # color_resize = cv2.resize(color_image, dsize=(400, 300), interpolation=cv2.INTER_AREA)
                h, w, c = color_image.shape
                drawrect(color_image, (234, 222), (400, 300), (0, 255, 255), 4, 'dotted')
                qImg = QtGui.QImage(color_image.data, w, h, w * c, QtGui.QImage.Format_RGB888)
                pixmap = QtGui.QPixmap.fromImage(qImg)
                self.label.setPixmap(pixmap)

        except Exception:
            QMessageBox.information(self, 'Error', 'Cannot read frame')
            print("cannot read frame.")
        finally:
            self.take_picture = False


    def start(self):
        self.take_picture = True
        th = threading.Thread(target=self.play)
        th.start()
        print("started..")

    def stop(self):
        self.take_picture = False
        print("stoped..")

    def check(self):
        # profile = pipeline.start(config)
        profile = pipeline.start(config)
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        color_image = np.asarray(color_frame.get_data())

        # take the only location of mushroom pot -> 1/3 * width,1/2*height
        recent_image = color_image[240:, 214:428]
        check_image = cv2.imread('./recent.jpg')[240:, 214:428]

        hist_recent = cv2.calcHist(recent_image, [1], None, [255], [0, 255])
        hist_check = cv2.calcHist(check_image, [1], None, [255], [0, 255])
        number = cv2.compareHist(hist_recent, hist_check, cv2.HISTCMP_CORREL)

        print(number)
        if number > 0.4:
            QMessageBox.information(self, 'Error', '배지를 넣어 주세요')
        else:
            #            배지입력 확인
            #
            print('Success')
            response = requests.get(SERVER_URL + 'myfarm/status', params={'id': 2, 'status': 'true'})

            self.stop()
            self.change_stack()
        pipeline.stop()

    def change_stack(self):
        self.parent().stack.setCurrentIndex(2)


class Window3(QtWidgets.QWidget, object):

    def __init__(self, parent=None):
        super(Window3, self).__init__()
        self.currentPictures = ('de0.jpg', 'de90.jpg', 'de180.jpg', 'de270.jpg')
        self.current = 0
        self.collect = 0

        self.start_before = Start_before(self)
        # self.start = Start(self)

        # self.start_before.finished.connect(self.go)
        self.start_before.signal.connect(self.renewal)

    # def setupUi(self, Dialog):
        Dialog = QtWidgets.QDialog()
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

    # def retranslateUi(self, Dialog):
    #     _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle("Dialog")
        self.pushButton.setText("갱신하기")
        self.total_num.setText("N 개")
        self.gather_num.setText( "N 개")
        self.label_12.setText("수확 대상")
        self.label_13.setText("현재 버섯수")
        self.pushButton_2.setText("→")
        self.pushButton_3.setText("←")
        self.file_name.setText("file_name")
        self.label_15.setText("사진 이동")
        self.label_16.setText("갱신 시간")
        self.time.setText("-----")
        self.temp.setText("-도")
        self.label_19.setText("온도")
        self.hum.setText("Dialog", "-도")
        self.label_21.setText("Dialog", "갱신 시간")

        self.pushButton_2.clicked.connect(self.right_click)
        self.pushButton_3.clicked.connect(self.left_click)
        self.pushButton.clicked.connect(self.before_run)
        self.change_picture()

    def change_picture(self, current=0):
        file_name = self.currentPictures[self.current]
        self.file_name.setText(file_name)
        current_img = load_image_into_numpy_array(os.path.join(PATH_TO_IMG, file_name))
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

        self.total_num.setText(str(len(boxes) - 1))
        self.collect = 0
        for box in boxes:
            size = get_size(box)
            print(size)
            if size > 6:
                self.collect += 1
        self.gather_num.setText(str(self.collect))

    def right_click(self):
        self.current = 0 if self.current + 1 > 3 else self.current + 1
        self.change_picture(self.current)

    def left_click(self):
        self.current = 3 if self.current - 1 < 0 else self.current - 1
        self.change_picture(self.current)

    def before_run(self):
        if not self.start_before.isRun:
            self.start_before.isRun = True
            self.start_before.start()

    def go(self):
        if not self.start.isRun:
            self.start.start()

    @pyqtSlot(str, str)
    def renewal(self, arg1, arg2):
        self.temp.setText(arg1 + '도')
        self.hum.setText(arg2 + '도')
        self.time.setText(str(datetime.datetime.now()))
        # self.data1.repaint(arg)
        # self.data2.repaint()
        # self.layout.repaint()
        print(arg1, arg2)

    @pyqtSlot(str, str)
    def renewal(self, arg1, arg2):
        self.data1.setText(arg1)
        self.data2.setText(arg2)
        # self.data1.repaint(arg)
        # self.data2.repaint()
        # self.layout.repaint()
        print(arg1, arg2)


class MainWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("버섯 GUI")
        self.setWindowIcon(QtGui.QIcon('mushroom2.jpg'))
        self.stack = QtWidgets.QStackedLayout(self)
        self.stack1 = Window1(self)
        self.stack2 = Window2(self)
        self.stack3 = Window3(self)

        self.stack.addWidget(self.stack1)
        self.stack.addWidget(self.stack2)
        self.stack.addWidget(self.stack3)
        self.show()

    # def change(self, data):
    #     self.stack2.set(data)


if __name__ == '__main__':
    app = QtWidgets.QApplication([])
    main = MainWindow()
    # main = Window1()
    app.exit(app.exec_())
