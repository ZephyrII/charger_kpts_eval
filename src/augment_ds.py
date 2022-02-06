import os
import numpy as np
import xml.etree.ElementTree as ET
import cv2
import random
import time
import copy

dataset_dir = os.path.join("/root/share/tf/dataset", '4_point')
output_dir = os.path.join("/root/share/tf/dataset", '4_point_final')
if not os.path.exists(os.path.join(output_dir)):
    os.makedirs(os.path.join(output_dir))
if not os.path.exists(os.path.join(output_dir, 'images')):
    os.makedirs(os.path.join(output_dir, 'images'))
if not os.path.exists(os.path.join(output_dir, 'annotations')):
    os.makedirs(os.path.join(output_dir, 'annotations'))
annotations = os.listdir(os.path.join(dataset_dir, 'annotations'))
scale_ratio = 0.5
full_size = np.array([5472, 3648])
# output_size = (full_size * scale_ratio).astype(np.int16)
output_size = np.array([960, 960]) #(full_size * scale_ratio).astype(np.int16)
# output_size = np.array([768, 768]) #(full_size * scale_ratio).astype(np.int16)
# Add images

annotations.sort(reverse=True)
for a in annotations:
    image_path = os.path.join(dataset_dir, 'full_img', a[:-4] + '.png')
    # if not os.path.exists(image_path):
    #     continue
    # full_img = cv2.imread(image_path)
    # secret_delta = full_img.shape[0]/full_size[0]
    print(image_path)
    no_augments = 15
    orig_tree = ET.parse(os.path.join(dataset_dir, "annotations_full", a))
    root = orig_tree .getroot()
    size = root.find('size')
    width_val = int(size.find('width').text)
    height_val = int(size.find('height').text)
    obj = root.find('object')

    bbox = obj.find('bndbox')

    xminv = float(bbox.find('xmin').text) * width_val
    yminv = float(bbox.find('ymin').text) * height_val
    xmaxv = float(bbox.find('xmax').text) * width_val
    ymaxv = float(bbox.find('ymax').text) * height_val
    hor_span = (xmaxv - xminv)
    ver_span = (ymaxv - yminv)
    # print("sizes", hor_span*ver_span, width_val*height_val*0.15)
    if hor_span*ver_span>width_val*height_val*0.15:
        no_augments = 15
        print("bigg")
    else:
        no_augments=10
    # no_augments=1
    for i in range(no_augments):
        starttime = time.time()
        # image_path = os.path.join(dataset_dir, 'full_img', a[:-4] + '.png')
        # if not os.path.exists(image_path):
        #     continue
        # full_img = cv2.imread(image_path)
        # secret_delta = full_img.shape[0]/full_size[0]
        # print(image_path)
        # no_augments = 15
        tree = copy.deepcopy(orig_tree)
        root = tree.getroot()
        size = root.find('size')
        width_val = int(size.find('width').text)
        height_val = int(size.find('height').text)
        obj = root.find('object')
        # scale = float(obj.find('scale').text)
        # offset_xv = int(float(root.find('offset_x').text))
        # offset_yv = int(float(root.find('offset_y').text))
        # print(offset_xv, offset_yv, "lol")
        # obj.find('scale').text = str(1.0)
        bbox = obj.find('bndbox')
        xmin = bbox.find('xmin')
        ymin = bbox.find('ymin')
        xmax = bbox.find('xmax')
        ymax = bbox.find('ymax')
        # xminv = float(bbox.find('xmin').text) * width_val
        # yminv = float(bbox.find('ymin').text) * height_val
        # xmaxv = float(bbox.find('xmax').text) * width_val
        # ymaxv = float(bbox.find('ymax').text) * height_val
        hor_span = (xmaxv - xminv)
        ver_span = (ymaxv - yminv)
        # height, width = image.shape[:2]
        max_zoom = min(0.85 *output_size[0]/hor_span, 0.85*output_size[1]/ver_span)
        # zoom = random.uniform(scale * 1.1, scale*1.6)
        # zoom = max(0.265, random.uniform(max_zoom*0.8, max_zoom))
        zoom = max(0.263157895, random.uniform(max_zoom*0.8, max_zoom))
        # print("zoom", zoom)
        # print("mzoom", max_zoom)
        # zoom=1
        obj.find('scale').text = str(zoom)
        center = np.array((xminv + (xmaxv - xminv) / 2,
                           yminv + (ymaxv - yminv) / 2))
        center = center*zoom
        # print("center", center)
        crop_offset = center - tuple(  # change to scaled_kp[5] in 8-point
            x / 2 + random.uniform(-0.05, 0.05) * x for x in output_size)

        # crop_offset = np.array(crop_offset*zoom).astype(np.int16)
        crop_offset = [int(max(min(crop_offset[0], full_size[0] * zoom - output_size[0]), 0)),
                       int(max(min(crop_offset[1], full_size[1] * zoom - output_size[1]), 0))]
        # print("crop_offset", crop_offset)
        root.find('offset_x').text = str(crop_offset[0])
        root.find('offset_y').text = str(crop_offset[1])
        xmin.text = str(max([(xminv * zoom - crop_offset[0]) / output_size[0], 0]))
        xmax.text = str(min([(xmaxv * zoom - crop_offset[0]) / output_size[0], output_size[0]]))
        ymin.text = str(max([(yminv * zoom - crop_offset[1]) / output_size[1], 0]))
        ymax.text = str(min([(ymaxv * zoom - crop_offset[1]) / output_size[1], output_size[1]]))
        # xmin.text = str((xminv * zoom - crop_offset[0]) / output_size[0])
        # xmax.text = str((xmaxv * zoom - crop_offset[0]) / output_size[0])
        # ymin.text = str((yminv * zoom - crop_offset[1]) / output_size[1])
        # ymax.text = str((ymaxv * zoom - crop_offset[1]) / output_size[1])
        if float(xmax.text)*output_size[0]>output_size[0] or float(xmin.text)*output_size[0]<0 \
                or float(ymax.text)*output_size[1]>output_size[1] or float(ymin.text)*output_size[1]<0:
            print("x")
        kps = obj.find('keypoints')
        for idx in range(4):
            kp = kps.find('keypoint' + str(idx))
            x = kp.find('x')
            y = kp.find('y')
            x_val = float(kp.find('x').text) * width_val
            y_val = float(kp.find('y').text) * height_val
            out_x = (x_val * zoom - crop_offset[0]) / output_size[0]
            out_y = (y_val * zoom - crop_offset[1]) / output_size[1]
            x.text = str(out_x)
            y.text = str(out_y)
        out_ann_file = os.path.join(output_dir, "annotations", a[:-4] + str(i) + ".txt")
        # tree.write(out_ann_file)
        image_path = os.path.join(dataset_dir, 'full_img', a[:-4] + '.png')
        out_image_path = os.path.join(output_dir, 'images', a[:-4] + str(i) + '.png')
        fulll_img = cv2.imread(image_path)

        #Brightness:
        alpha = np.random.uniform(1.5, 0.4)
        beta = np.random.uniform(0, 40)
        # alpha = np.random.normal(1.0, 0.3)
        # beta = np.random.normal(0.0, 5)
        full_img = np.clip(alpha*fulll_img + beta, 0, 255)
        # full_img = fulll_img

        # crop_offset = np.array(crop_offset/zoom).astype(np.int16)
        full_img = cv2.resize(full_img, None, fx=zoom, fy=zoom)
        out_image = full_img[crop_offset[1]:crop_offset[1] + output_size[1],
                    crop_offset[0]:crop_offset[0] + output_size[0]]

        size.find('width').text = str(out_image.shape[1])
        size.find('height').text = str(out_image.shape[0])  # TODO: fix

        if crop_offset[0] < 0 or crop_offset[1] < 0:
            print("xd")
        # print(out_image.shape)
        # print(output_size)
        if out_image.shape[1] < output_size[0] or out_image.shape[0] < output_size[1]:
            print("xdd")
        tree.write(out_ann_file)
        cv2.imwrite(out_image_path, out_image)
        # cv2.imwrite(out_label_path, out_label)
        # print("time", time.time()-starttime)
