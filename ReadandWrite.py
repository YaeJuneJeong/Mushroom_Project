import shutil
import xml.etree.ElementTree as ET
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

#  전체 파일 통합
def merge(dir_path, move_path):
    path, dirs, files = next(os.walk(dir_path))
    file_count = 0
    for i in dirs:
        directory = os.path.join(dir_path, i)
        path, dirs, files = next(os.walk(directory))
        file_count += len(files)
    print(file_count)
    fname = ['mushroom{}.jpg'.format(i) for i in range(file_count)]
    f = 0
    for files_path in os.listdir(dir_path):
        files_path = os.path.join(dir_path, files_path)
        files_path = files_path + '/*.jpg'
        for file_path in glob.glob(files_path):
            dst = os.path.join(move_path, fname[f])
            f = f + 1
            print(dst)
            shutil.copyfile(file_path, dst)


# txt -> yolo csv
def get_point(dir_path):

    picture_name = []
    picture_width = []
    picture_height = []
    mushroom_xmins = []
    mushroom_xmaxs = []
    mushroom_ymins = []
    mushroom_ymaxs = []

    width = 0
    height =0
    dirtxt_path = dir_path + '/*.txt'

    for filename in glob.glob(dirtxt_path):
        fn = open(filename)
        image_name = filename.split('.')[0]
        image_name = image_name.split('\\')[-1]
        image_name = image_name+'.jpg'
        image = Image.open(os.path.join(dir_path,image_name))
        width, height = image.size
        lines = fn.readlines()
        for line in lines:
            line = line.replace('\n', '')
            location = line.split(' ')
            picture_name.append(image_name)
            picture_width.append(width)
            picture_height.append(height)

            centerx= int(float(location[1])*width)
            centery= int(float(location[2])*height)
            mushroom_width = int(float(location[3])*width/2)
            mushroom_height = int(float(location[4])*height/2)

            mushroom_xmin = centerx - mushroom_width
            mushroom_xmax = centerx + mushroom_width
            mushroom_ymin = centery - mushroom_height
            mushroom_ymax = centery + mushroom_height

            mushroom_xmins.append(mushroom_xmin)
            mushroom_xmaxs.append(mushroom_xmax)
            mushroom_ymins.append(mushroom_ymin)
            mushroom_ymaxs.append(mushroom_ymax)
    result = pd.DataFrame(
        {'filename': picture_name, 'width': picture_width, 'height': picture_height,'class':'mushroom',
         'xmin': mushroom_xmins, 'ymin': mushroom_ymins,'xmax':mushroom_xmaxs,'ymax':mushroom_ymaxs})
    result.to_csv('./third_csv_1.csv')

# xml -> coco csv
def get_xml(dir_path, dir_num):
    num = [0, 353, 750, 1073]
    num_sum = np.sum(num[0:dir_num], dtype=np.int32)
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
        filename = int(filename.split('_')[0]) + num_sum
        filename = 'mushroom' + str(filename)
        for line in tree.findall('object'):
            line = line.find('bndbox')
            min_width = int(line.find('xmin').text)
            min_height = int(line.find('ymin').text)
            max_width = int(line.find('xmax').text)
            max_height = int(line.find('ymax').text)

            #
            mushroom_name.append(filename)
            mushroom_x.append(min_width/whole_width)
            mushroom_y.append(min_height/whole_height)
            mushroom_width.append(max_width/whole_width)
            mushroom_height.append(max_height/whole_height)
    # mushrooms = np.array(mushroom_name,mushroom_x,mushroom_y,mushroom_width,mushroom_height)
    return pd.DataFrame({'name': mushroom_name, 'min_x': mushroom_x, 'min_y': mushroom_y, 'max_x': mushroom_width,
                         'max_y': mushroom_height})


def get_pictures(names, file_path):
    for value in names:
        print(value)


# yolo csv -> coco csv
def convert_coco(file_path, picture_path):
    data = pd.read_csv(file_path)
    names = []
    minx = []
    miny = []
    maxx = []
    maxy = []
    for index, row in data.iterrows():
        name = row['name'] + '.jpg'
        path = os.path.join(picture_path, name)
        image = Image.open(path)
        width, height = image.size
        # check int values
        min_x =  (row['x'] - row['width'] / 2)
        max_x =  (row['x'] + row['width'] / 2)
        min_y =  (row['y'] - row['height'] / 2)
        max_y =  (row['y'] + row['height'] / 2)
        minx.append(min_x)
        miny.append(min_y)
        maxx.append(max_x)
        maxy.append(max_y)
        names.append(row['name'])
    return pd.DataFrame({'file': names, 'min_x': minx, 'min_y': miny, 'max_x': maxx, 'max_y': maxy})

# txt file to coco csv file
def merge_sample(dir_path,picture_path):

    mushroom_name = []
    mushroom_minwidth = []
    mushroom_maxwidth = []
    mushroom_minheight = []
    mushroom_maxheight = []
    dir_path = dir_path + '/*.txt'
    sequence = sorted(glob.glob(dir_path))
    file_count = len(sequence)
    fname = ['mushroom{}.jpg'.format(i) for i in range(file_count)]
    for file,picture_name in zip(sequence,fname):
        fn = open(file)

        filename = file.split('.', 1)
        filename = filename[0].split('\\')[1]
        filename = os.path.join(picture_path,filename+'.jpg')
        dst = os.path.join('D:/merge1',picture_name)
        shutil.copy(filename, dst)
        lines = fn.readlines()
        for line in lines:
            line = line.replace('\n', '')
            location = line.split(' ')
            mushroom_name.append(picture_name)
            mushroom_minwidth.append(float(location[1])-float(location[3])/2)
            mushroom_maxwidth.append(float(location[1])+float(location[3])/2)
            mushroom_minheight.append(float(location[2])-float(location[4])/2)
            mushroom_maxheight.append(float(location[2])+float(location[4])/2)
    record = pd.DataFrame({'name': mushroom_name,'min_x':mushroom_minwidth,'max_x':mushroom_maxwidth,'min_y':mushroom_minheight,'max_y':mushroom_maxheight})
    record.to_csv('D:/example.csv')

def xml_to_csv(path):
    """Iterates through all .xml files (generated by labelImg) in a given directory and combines
    them in a single Pandas dataframe.

    Parameters:
    ----------
    path : str
        The path containing the .xml files
    Returns
    -------
    Pandas DataFrame
        The produced dataframe
    """

    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            value = (root.find('filename').text,
                     int(root.find('size')[0].text),
                     int(root.find('size')[1].text),
                     member[0].text,
                     int(member[4][0].text),
                     int(member[4][1].text),
                     int(member[4][2].text),
                     int(member[4][3].text)
                     )
            xml_list.append(value)
    column_name = ['filename', 'width', 'height',
                   'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    xml_df.to_csv('./third_csv_2.csv')


# get_point('D:third')
xml_to_csv('D:third')


# merge_sample('D:/third','D:/Mushrooms/Boletus')
# name_array = get_point('D:/third', 3)
# name_array.to_csv('D:/third.csv')
# get_pictures(name_array)
# merge('D:\Mushrooms','D:\merge')
# name = get_xml('D:/third',3)
# name.to_csv('D:/third_xml1.csv')
# coco = convert_coco('D:/third.csv', 'D:/merge')
# coco.to_csv('D:/third_xml.csv')
