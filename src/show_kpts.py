import os
import cv2
import xml.etree.ElementTree as ET
import time
from scipy.spatial.transform import Rotation as R
import numpy as np

base_path = '/root/share/tf/dataset/final_localization/corners_1.0/'
images = os.listdir(os.path.join(base_path, 'annotations'))
images.sort(reverse=True)
images = iter(images)
frame = None
alpha = 0.5
img_fname = None
ann_fname = None
fname = None
previous = None
while True:
    time.sleep(0.2)
    try:
        previous, fname = fname, next(images)
        img_fname = base_path + 'images/' + fname[:-4]+'.png'
        ann_fname = base_path + 'annotations/' + fname
        print(img_fname)

        tree = ET.parse(ann_fname)
        root = tree.getroot()
        xmin = 0
        xmax = 0
        ymin = 0
        ymax = 0
        frame = cv2.imread(img_fname)
        size = root.find('size')
        w = int(size.find('width').text)
        h = int(size.find('height').text)
        for object in root.findall('object'):
            bbox = object.find('bndbox')
            xmin = float(bbox.find('xmin').text)*w
            xmax = float(bbox.find('xmax').text)*w
            ymin = float(bbox.find('ymin').text)*h
            ymax = float(bbox.find('ymax').text)*h
            kps = object.find('keypoints')
            keypoints = []
            for i in range(4):
                kp = kps.find('keypoint' + str(i))
                keypoints.append((float(kp.find('x').text), float(kp.find('y').text)))
            for i, kp in enumerate(keypoints):
                print(kp)
                if i==0:
                    color = (0,0,255)
                elif i==1:
                    color = (255,0,0)
                elif i==2:
                    color = (0,255,0)
                elif i==3:
                    color = (255,255,255)
                else:
                    color = (255, 0, 255)
                center = (int(kp[0]*w), int(kp[1]*h))
                cv2.circle(frame, center, 10, color, -1)


        cv2.imwrite('/root/catkin_ws/src/charger_kpts_train/src/test/out.jpg', cv2.resize(frame, (720,720)))

    except StopIteration:
        break