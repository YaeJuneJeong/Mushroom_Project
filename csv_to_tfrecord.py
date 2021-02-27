from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import io
import pandas as pd
import tensorflow as tf
import glob
import xml.etree.ElementTree as ET

from PIL import Image
import base64
from collections import namedtuple, OrderedDict

from object_detection.dataset_tools import tf_record_creation_util
from object_detection.utils import dataset_util
from object_detection.utils import label_map_util

flags = tf.app.flags
# flags.DEFINE_string('data_dir', '', 'Root directory to raw pet dataset.')
# flags.DEFINE_string('output_dir', '', 'Path to directory to output TFRecords.')
# flags.DEFINE_string('label_map_path', 'data/pet_label_map.pbtxt',
#                     'Path to label map proto')
# flags.DEFINE_boolean('faces_only', True, 'If True, generates bounding boxes '
#                      'for pet faces.  Otherwise generates bounding boxes (as '
#                      'well as segmentations for full pet bodies).  Note that '
#                      'in the latter case, the resulting files are much larger.')
# flags.DEFINE_string('mask_type', 'png', 'How to represent instance '
#                     'segmentation masks. Options are "png" or "numerical".')
# flags.DEFINE_integer('num_shards', 10, 'Number of TFRecord shards')
#
FLAGS = flags.FLAGS


class picture_info():

    def __init__(self,name, minx, maxx, miny, maxy):
        self.name = name
        self.minx = minx
        self.maxx = maxx
        self.miny = miny
        self.maxy = maxy


# mushroom
def class_text_to_int():
    return 1


def create_tf_example(array,img_path):
    height = None
    width = None
    name = None
    encoded_image_data = []

    xmin = []
    xmax = []
    ymin = []
    ymax = []
    class_text = []
    classes = []

    name = array[0].name
    img_path = os.path.join(img_path,array[0].name)
    img = Image.open(img_path)
    width,height = img.size()
    encoded_image_data = base64.b64encode(img)

    for obj in array:
        xmin.append(obj.minx)
        xmax.append(obj.maxx)
        ymin.append(obj.miny)
        ymax.append(obj.maxy)
        class_text.append('Mushroom')
        classes.append(1)

    tf_example = tf.train.Example(features=tf.train.Feature(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(name.encode('utf8')),
        'image/source_id': dataset_util.bytes_feature(name.encode('utf8')),
        'image/encoded': dataset_util.bytes_feature(encoded_image_data),
        'image/format': dataset_util.bytes_feature('jpg'.encode('utf8')),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
        'image/object/class/text': dataset_util.bytes_list_feature(class_text.encode('utf8')),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))

    return tf_example


# coco type csv to
def tfrecord(csv_path, img_path):
    writer = tf.python_io.TFRecordWriter('.Dataset/train.record')
    data = pd.read_csv(csv_path)

    stack_list = []
    for index in data.index:
        name = data.loc[index,'name']
        minx = data.loc[index,'min_x']
        maxx = data.loc[index,'max_x']
        miny = data.loc[index,'min_y']
        maxy = data.loc[index,'max_y']

        row = picture_info(name,minx,maxx,miny,maxy)
        if len(stack_list) is 0:
            stack_list.append(row)
        elif stack_list[0].name is row.name:
            stack_list.append(row)
        else:
            tf_example = create_tf_example(stack_list,img_path)
            writer.write(tf_example.SerializeToString())
            stack_list.clear()
            stack_list.append(row)
    writer.close()

tfrecord('csv_path','img_path')
