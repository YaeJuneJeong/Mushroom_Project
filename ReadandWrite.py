import shutil
import xml.etree.ElementTree as ET
import cv2
import pandas as pd
import glob
import numpy as np
import os
from PIL import Image


# class picture():
#     # location = >array yolo format
#     def __init__(self,name,x,y,width,height):
#         self.name = name
#         self.x = x
#         self.y = y
#         self.width = width
#         self.height = height

def merge(dir_path, move_path):

    path,dirs,files = next(os.walk(dir_path))
    file_count = 0
    for i in dirs:
        directory = os.path.join(dir_path,i)
        path, dirs, files = next(os.walk(directory))
        file_count += len(files)
    print(file_count)
    fname = ['mushroom{}.jpg'.format(i) for i in range(file_count)]
    f =0
    for files_path in os.listdir(dir_path):
        files_path = os.path.join(dir_path,files_path)
        files_path = files_path + '/*.jpg'
        for file_path in glob.glob(files_path):
            dst = os.path.join(move_path,fname[f])
            f = f + 1
            print(dst)
            shutil.copyfile(file_path,dst)


def get_point(dir_path):

    mushroom_name = []
    mushroom_x = []
    mushroom_y = []
    mushroom_width = []
    mushroom_height = []

    dir_path = dir_path + '/*.txt'
    for filename in glob.glob(dir_path):
        fn = open(filename)

        filename = filename.split('.',1)
        filename = filename[0].split('\\')[1]
        lines = fn.readlines()
        for line in lines:
            line = line.replace('\n', '')
            location = line.split(' ')
            mushroom_name.append(filename)
            mushroom_x.append(location[1])
            mushroom_y.append(location[2])
            mushroom_width.append(location[3])
            mushroom_height.append(location[4])
    return pd.DataFrame({'name':mushroom_name,'x':mushroom_x,'y':mushroom_y,'width':mushroom_width,'height':mushroom_height})

def get_xml(dir_path):

    mushroom_name = []
    mushroom_x = []
    mushroom_y = []
    mushroom_width = []
    mushroom_height = []

    dir_path = dir_path + '/*.xml'

    for filename in glob.glob(dir_path):


        tree = ET.parse(filename)
        tree = tree.getroot()
        size = tree.find('size')
        whole_width = int(size.find('width').text)
        whole_height = int(size.find('height').text)
        filename = filename.split('.', 1)
        filename = filename[0].split('\\')[1]
        for line in tree.findall('object'):
            line = line.find('bndbox')
            min_width = int(line.find('xmin').text)
            min_height = int(line.find('ymin').text)
            max_width = int(line.find('xmax').text)
            max_height = int(line.find('ymax').text)

            #
            mushroom_name.append(filename)
            mushroom_x.append(min_width)
            mushroom_y.append(min_height)
            mushroom_width.append(max_width)
            mushroom_height.append(max_height)
    # mushrooms = np.array(mushroom_name,mushroom_x,mushroom_y,mushroom_width,mushroom_height)
    return pd.DataFrame({'name':mushroom_name,'min_x':mushroom_x,'min_y':mushroom_y,'max_x':mushroom_width,'max_y':mushroom_height})

def get_pictures(names,file_path):

    for value in names:
        print(value)
def convert_coco(file_path,picture_path):
    data =pd.read_csv(file_path)
    names = []
    minx = []
    miny = []
    maxx = []
    maxy = []
    for index , row in data.iterrows():
        name = row['name']+'.jpg'
        path = os.path.join(picture_path,name)
        image = Image.open(path)
        width,height = image.size
        min_x = round(width*(row['x']-row['width']/2))
        max_x = round(width*(row['x']+row['width']/2))
        min_y = round(height*(row['y']-row['height']/2))
        max_y = round(height*(row['y']-row['height']/2))
        minx.append(min_x)
        miny.append(min_y)
        maxx.append(max_x)
        maxy.append(max_y)
        names.append(row['name'])
    return pd.DataFrame({'file':names,'min_x':minx,'min_y':miny,'max_x':maxx,'max_y':maxy})



# name_array = get_point('D:/third')
# name_array.to_csv('D:/third.csv')
# get_pictures(name_array)
# merge('D:\Mushrooms','D:\merge')
# name =get_xml('D:/third')
# name.to_csv('D:/second_xml1.csv')
# coco = convert_coco('D:/third.csv','D:/Mushrooms/Boletus')
# coco.to_csv('D:/third_xml.csv')