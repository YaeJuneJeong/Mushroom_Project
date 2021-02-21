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
from object_detection.utils import dataset_util
from collections import namedtuple, OrderedDict

# mushroom
def class_text_to_int():
    return 1

def create_tf_example(xml_path,img_path):
    width = []
    height = []
    mushroom_name = []
    mushroom_minx = []
    mushroom_miny = []
    mushroom_maxx = []
    mushroom_maxy = []

    dir_path = xml_path + '/*.xml'

    for filename in glob.glob(dir_path):

        tree = ET.parse(filename)
        tree = tree.getroot()
        size = tree.find('size')
        whole_width = int(size.find('width').text)
        whole_height = int(size.find('height').text)
        filename = filename.split('.', 1)
        filename = filename[0].split('\\')[1]
        full_path = os.path.join(img_path,filename+'.jpg')
        with tf.gfile.GFile(full_path,'rb') as fid:
            encoded_jpg = fid.read()
        for line in tree.findall('object'):
            line = line.find('bndbox')
            min_width = int(line.find('xmin').text)
            min_height = int(line.find('ymin').text)
            max_width = int(line.find('xmax').text)
            max_height = int(line.find('ymax').text)

            #
            mushroom_name.append(filename)
            mushroom_minx.append(min_width/whole_width)
            mushroom_miny.append(min_height/whole_height)
            mushroom_maxx.append(max_width/whole_width)
            mushroom_maxy.append(max_height/whole_height)
            width.append(whole_width)
            height.append(whole_height)
    tf_example = tf.train.Example(features = tf.train.Feature(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename.encode('utf8')),
        'image/source_id': dataset_util.bytes_feature(filename.encode('utf8')),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature('jpg'.encode('utf8')),
        'image/object/bbox/xmin': dataset_util.float_list_feature(mushroom_minx),
        'image/object/bbox/xmax': dataset_util.float_list_feature(mushroom_maxx),
        'image/object/bbox/ymin': dataset_util.float_list_feature(mushroom_miny),
        'image/object/bbox/ymax': dataset_util.float_list_feature(mushroom_maxy),
        'image/object/class/text': dataset_util.bytes_list_feature(0),
        'image/object/class/label': dataset_util.int64_list_feature('mushroom'),
    }))

    return tf_example

