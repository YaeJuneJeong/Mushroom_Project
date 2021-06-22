import os
import time
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
import warnings
import pyrealsense2 as rs
import numpy as np
import cv2
# warnings.filterwarnings('ignore')   # Suppress Matplotlib warnings
from object_detection.builders import model_builder
from object_detection.utils import label_map_util, config_util
from object_detection.utils import visualization_utils as viz_utils

# from PicutreCapture.extract import label_id_offset

PATH_TO_MODEL_DIR = 'C:/Users/jyj98/tensorflow/workspace/training_demo/exported-models/mushroom_model1'
PATH_TO_CFG = PATH_TO_MODEL_DIR + "/pipeline.config"
PATH_TO_CKPT = PATH_TO_MODEL_DIR + "/checkpoint"

print('Loading model... ', end='')
start_time = time.time()
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)
frames = pipeline.wait_for_frames()
color_frame = frames.get_color_frame()
color_image = np.asanyarray(color_frame.get_data())
pipeline.stop()

# Load pipeline config and build a detection model
configs = config_util.get_configs_from_pipeline_file(PATH_TO_CFG)
model_config = configs['model']
detection_model = model_builder.build(model_config=model_config, is_training=False)

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(PATH_TO_CKPT, 'ckpt-0')).expect_partial()


@tf.function
def detect_fn(image):
    """Detect objects in image."""

    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)

    return detections


end_time = time.time()
elapsed_time = end_time - start_time

PATH_TO_LABELS = 'C:/Users/jyj98/tensorflow/workspace/training_demo/annotations/label_map.pbtxt'

PATH_TO_SAVED_MODEL = PATH_TO_MODEL_DIR + "/saved_model"
detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)


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


# for image_path in IMAGE_PATHS:
# image_path = 'C:/Users/jyj98/tensorflow/workspace/training_demo/images/train/Mushroom.jpg'
# print('Running inference for {}... '.format(image_path), end='')

# image_path = color_image
# image_np = load_image_into_numpy_array(image_path)
image_np = color_image
# Things to try:
# Flip horizontally
# image_np = np.fliplr(image_np).copy()

# Convert image to grayscale
# image_np = np.tile(np.mean(image_np, 2, keepdims=True), (1, 1, 3)).astype(np.uint8)

# The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
input_tensor = tf.convert_to_tensor(image_np)
# The model expects a batch of images, so add an axis with `tf.newaxis`.
input_tensor = input_tensor[tf.newaxis, ...]

# input_tensor = np.expand_dims(image_np, 0)
detections = detect_fn(input_tensor)

# All outputs are batches tensors.
# Convert to numpy arrays, and take index [0] to remove the batch dimension.
# We're only interested in the first num_detections.
num_detections = int(detections.pop('num_detections'))
detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}
detections['num_detections'] = num_detections
print(num_detections)
# detection_classes should be ints.
detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

image_np_with_detections = image_np.copy()

label_id_offset = 1
viz_utils.visualize_boxes_and_labels_on_image_array(image_np_with_detections, detections['detection_boxes'],
                                                    detections['detection_classes'],
                                                    detections['detection_scores'], category_index,
                                                    use_normalized_coordinates=True,
                                                    max_boxes_to_draw=200,
                                                    min_score_thresh=0.5,
                                                    agnostic_mode=False)

# plt.figure()
# plt.imshow(image_np_with_detections)
cv2.imwrite('./mushroom4.jpg',color_image)
cv2.imwrite('./results.jpg', image_np_with_detections)
print('Done')
# print(detections['detection_boxes'])
# print(detections['detection_scores'])
# plt.show()

# sphinx_gallery_thumbnail_number = 2
