"""
Mask R-CNN
Train on the toy charger dataset and implement color splash effect.

Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------
"""

import os
import sys
import numpy as np
import skimage.draw
import cv2
import xml.etree.ElementTree as ET

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")
from mrcnn.config import Config
from mrcnn import model as modellib, utils


# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")


# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = ""
np.set_printoptions(threshold=sys.maxsize)
############################################################
#  Configurations
############################################################


class chargerConfig(Config):

    NAME = "charger"
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 1  # Background + charger
    STEPS_PER_EPOCH = 10000
    DETECTION_MIN_CONFIDENCE = 0.9
    LEARNING_RATE = 0.0001
    NUM_POINTS = 4

############################################################
#  Dataset
############################################################

class ChargerDataset(utils.Dataset):

    def __init__(self, class_map=None):
        super().__init__(class_map=class_map)
        self.increase_bbox_percent = 0.00

    def load_charger(self, dataset_dir, subset):
        """Load a subset of the charger dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have only one class to add.
        self.add_class("charger", 1, "charger")

        # Train or validation dataset?
        assert subset in ["train", "val"]
        if subset == "val":
            # pass
            dataset_dir = os.path.join(dataset_dir, 'val')
        annotations = os.listdir(os.path.join(dataset_dir, 'annotations'))

        # Add images
        for a in annotations:

            # info = self.image_info[image_id]
            # ann_fname = info['annotation']
            tree = ET.parse(os.path.join(dataset_dir, 'annotations', a))
            root = tree.getroot()
            size = root.find('size')
            w = int(size.find('width').text)
            h = int(size.find('height').text)
            for obj in root.findall('object'):
                bndboxxml = obj.find('bndbox')
                if bndboxxml is not None:
                    xmin = int(float(bndboxxml.find('xmin').text) * w - self.increase_bbox_percent * w)
                    ymin = int(float(bndboxxml.find('ymin').text) * h - self.increase_bbox_percent * h)
                    xmax = int(float(bndboxxml.find('xmax').text) * w + self.increase_bbox_percent * w)
                    ymax = int(float(bndboxxml.find('ymax').text) * h + self.increase_bbox_percent * h)

                    # return

            image_path = os.path.join(dataset_dir, 'images', a[:-4]+'.png')
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            self.add_image(
                "charger",
                image_id=a,  # use file name as a unique image id
                path=image_path,
                width=width, height=height,
                mask=os.path.join(dataset_dir, 'labels', a[:-4]+'_label.png'),
                annotation=os.path.join(dataset_dir, 'annotations', a),
                bbox=np.array([xmin, ymin, xmax, ymax]))

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a charger dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "charger":
            return super(self.__class__, self).load_mask(image_id)

        info = self.image_info[image_id]
        mask = cv2.imread(info['mask'])
        return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)

    def load_kp(self, image_id, num_points):

        xmin, ymin, xmax, ymax = self.image_info[image_id]['bbox']
        info = self.image_info[image_id]
        # print(image_id)
        ann_fname = info['annotation']
        tree = ET.parse(ann_fname)
        root = tree.getroot()
        keypoints = []
        size = root.find('size')
        w = int(size.find('width').text)
        h = int(size.find('height').text)
        kp_maps = np.zeros((num_points, w, h), dtype=np.int32)
        for object in root.findall('object'):
            kps = object.find('keypoints')
            bbox = object.find('bndbox')
            # xmin = float(bbox.find('xmin').text)
            # ymin = float(bbox.find('ymin').text)
            # xmax = float(bbox.find('xmax').text)
            # ymax = float(bbox.find('ymax').text)
            xmin = float(bbox.find('xmin').text) * w - self.increase_bbox_percent
            ymin = float(bbox.find('ymin').text) * h - self.increase_bbox_percent
            xmax = float(bbox.find('xmax').text) * w + self.increase_bbox_percent
            ymax = float(bbox.find('ymax').text) * h + self.increase_bbox_percent
            bw = (xmax - xmin) * w
            bh = (ymax - ymin) * h
            for i in range(num_points):
                kp = kps.find('keypoint' + str(i))
                point_size = 5
                point_center = (
                int((float(kp.find('y').text) * h - ymin) * self.image_info[image_id]['height'] / (ymax - ymin)),
                int((float(kp.find('x').text) * w - xmin) * self.image_info[image_id]['width'] / (xmax - xmin)))
                keypoints.append((point_center[1], point_center[0]))
                # kp_maps[i, int(float(kp.find('y').text)*h), int(float(kp.find('x').text)*w)] = 1.0
                kp_maps[i, point_center[0] - point_size:point_center[0] + point_size,
                point_center[1] - point_size:point_center[1] + point_size] = 255
                # kp_maps[i] = cv2.GaussianBlur(kp_maps[i], (5,5), sigmaX=2)
                # kp_maps[i] = cv2.GaussianBlur(kp_maps[i], (3,3), sigmaX=0)
                # cv2.imshow('xddlol', kp_maps[i])
                # cv2.waitKey(0)

        return kp_maps, keypoints

    def load_yaw(self, image_id):

        info = self.image_info[image_id]
        ann_fname = info['annotation']
        tree = ET.parse(ann_fname)
        root = tree.getroot()
        theta = 0
        for obj in root.findall('object'):
            thetaxml = obj.find('theta')
            if thetaxml is not None:
                theta = float(thetaxml.text)
        return (theta + 1) / 2

    def load_bbox(self, image_id):
        info = self.image_info[image_id]
        ann_fname = info['annotation']
        tree = ET.parse(ann_fname)
        root = tree.getroot()
        size = root.find('size')
        w = int(size.find('width').text)
        h = int(size.find('height').text)
        for obj in root.findall('object'):
            bndboxxml = obj.find('bndbox')
            if bndboxxml is not None:
                xmin = int(float(bndboxxml.find('xmin').text) * w - self.increase_bbox_percent * w)
                ymin = int(float(bndboxxml.find('ymin').text) * h - self.increase_bbox_percent * h)
                xmax = int(float(bndboxxml.find('xmax').text) * w + self.increase_bbox_percent * w)
                ymax = int(float(bndboxxml.find('ymax').text) * h + self.increase_bbox_percent * h)

                return np.array([xmin, ymin, xmax, ymax])

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "charger":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


def train(model):
    """Train the model."""
    # Training dataset.
    dataset_train = ChargerDataset()
    dataset_train.load_charger(args.dataset, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = ChargerDataset()
    dataset_val.load_charger(args.dataset, "val")
    dataset_val.prepare()

    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=300,
                layers='5+')


############################################################
#  Training
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect chargers.')
    parser.add_argument('--dataset', required=False,
                        metavar="/root/share/tf/dataset/Inea/7-point",
                        help='Directory of the charger dataset')
    parser.add_argument('--weights', required=True,
                        metavar="./mask_rcnn_coco.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    args = parser.parse_args()

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    config = chargerConfig()
    config.display()

    # Create model
    model = modellib.MaskRCNN(mode="training", config=config, model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
        # weights_path = model.get_imagenet_inc_res_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    train(model)
