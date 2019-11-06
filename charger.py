"""
Mask R-CNN
Train on the toy charger dataset and implement color splash effect.

Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 charger.py train --dataset=/path/to/charger/dataset --weights=coco

    # Resume training a model that you had trained earlier
    python3 charger.py train --dataset=/path/to/charger/dataset --weights=last

    # Train a new model starting from ImageNet weights
    python3 charger.py train --dataset=/path/to/charger/dataset --weights=imagenet

    # Apply color splash to an image
    python3 charger.py splash --weights=/path/to/weights/file.h5 --image=<URL or path to file>

    # Apply color splash to video using the last weights you trained
    python3 charger.py splash --weights=last --video=<URL or path to file>
"""

import os
import sys
import threading
import datetime
import numpy as np
import skimage.draw
import cv2
import xml.etree.ElementTree as ET
import tensorflow as tf
import keras

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

############################################################
#  Configurations
############################################################


class chargerConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "charger"
    NUM_POINTS = 8

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + charger

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 50

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9
    LEARNING_RATE = 0.001


############################################################
#  Dataset
############################################################

class ChargerDataset(utils.Dataset):

    def __init__(self, directory, mode):
        super().__init__()
        self.mask = None
        self.overlay = None
        self.kp = []
        self.poly = []
        self.frame = None
        self.slice_size = (720, 960)
        self.directory = directory
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)
        if not os.path.exists(os.path.join(self.directory, 'images')):
            os.makedirs(os.path.join(self.directory, 'images'))
        if not os.path.exists(os.path.join(self.directory, 'labels')):
            os.makedirs(os.path.join(self.directory, 'labels'))
        if not os.path.exists(os.path.join(self.directory, 'annotations')):
            os.makedirs(os.path.join(self.directory, 'annotations'))

        labeled_path = os.path.join(self.directory, 'images')
        self.labeled = os.listdir(labeled_path)
        self.add_class("charger", 1, "charger")
        if mode=='train':
            raw_path = os.path.join(self.directory, 'raw')
            raw = os.listdir(raw_path)
            self.unlabeled = [x for x in raw if x not in self.labeled]
            # np.random.shuffle(self.imgs)

            cv2.namedWindow("Mask labeler", 0)
        elif mode=='val':
            self.add_labeled_images()

    def get_unlabeled(self):
        image_id = self.unlabeled.pop(0)
        image_path = os.path.join(self.directory, "raw", image_id)
        image = cv2.imread(image_path)
        return image, image_id

    # def load_charger(self, dataset_dir, subset):
    #     """Load a subset of the charger dataset.
    #     dataset_dir: Root directory of the dataset.
    #     subset: Subset to load: train or val
    #     """
    #     # Add classes. We have only one class to add.
    #     self.add_class("charger", 1, "charger")
    #
    #     # Train or validation dataset?
    #     assert subset in ["train", "val"]
    #
    #     if subset=='val':
    #         annotations = os.listdir(os.path.join(dataset_dir, 'annotations'))
    #         for a in annotations:
    #             image_path = os.path.join(dataset_dir, 'images',  a[:-4]+'.png')
    #             image = cv2.imread(image_path) # skimage.io.imread(image_path)
    #             height, width = image.shape[:2]
    #
    #             self.add_image(
    #                 "charger",
    #                 labeled=True,
    #                 image_id=a,  # use file name as a unique image id
    #                 path=image_path,
    #                 width=width, height=height,
    #                 mask=os.path.join(dataset_dir, 'labels',  a[:-4]+'_label.png'),
    #                 annotation=os.path.join(dataset_dir, 'annotations', a))
    #     else:
    #         imgs = os.listdir(dataset_dir)
    #         np.random.shuffle(imgs)
    #         # Add images
    #         for a in imgs[:2]:
    #             image_path = os.path.join(dataset_dir, a)
    #             image = cv2.imread(image_path) # skimage.io.imread(image_path)
    #             height, width = image.shape[:2]
    #
    #             self.add_image(
    #                 "charger",
    #                 labeled=False,
    #                 image_id=a,  # use file name as a unique image id
    #                 path=image_path,
    #                 width=width, height=height)

    def self_train(self, pred, image, image_id):
        print('xd', pred["masks"].shape)
        self.mask = pred["masks"][:, :, 0].astype(np.uint8) #(np.sum(pred["masks"], -1, keepdims=True) >= 1).astype(np.uint8)  # TODO get best prediction
        self.kp = np.reshape(pred["kp"][0, 0], (4, 2))
        print(self.mask.shape)
        print(np.max(self.mask))
        self.frame = image
        print(self.frame.shape)
        height, width = self.frame.shape[:2]

        self.show_mask()

        k = cv2.waitKey(0)
        if k == ord(' '):
            self.save(image_id)
            self.add_image(
                "charger",
                labeled=True,
                image_id=image_id[:-4],  # use file name as a unique image id
                path=os.path.join(self.directory, 'images', image_id),
                width=width, height=height,
                mask=os.path.join(self.directory, 'labels', image_id[:-4] + '_label.png'),
                annotation=os.path.join(self.directory, 'annotations', image_id[:-4] + '.txt'))
        if k == ord('r'):
            self.mask = None
            self.kp = []
            self.query_gt(image_id)
        if k == ord('q'):
            exit(0)

    def add_labeled_images(self):
        for a in self.labeled:
            image_path = os.path.join(self.directory, 'images', a)
            image = cv2.imread(image_path) # skimage.io.imread(image_path)
            height, width = image.shape[:2]

            self.add_image(
                "charger",
                labeled=True,
                image_id=a,  # use file name as a unique image id
                path=image_path,
                width=width, height=height,
                mask=os.path.join(self.directory, 'labels', a[:-4] + '_label.png'),
                annotation=os.path.join(self.directory, 'annotations', a[:-4] + '.txt'))

    def save(self, image_id):
        print(image_id)
        # image_id = str(image_id)
        if self.mask is not None:
            for res in [1]:
                label_mask = np.copy(self.mask)
                image = np.copy(self.frame)
                resized_label = cv2.resize(label_mask, None, fx=res, fy=res)
                resized_image = cv2.resize(image, None, fx=res, fy=res)
                print(np.array(self.kp).shape)

                scaled_kp = (np.array(self.kp) / np.array(self.frame.shape[:2])) * np.array(
                    resized_image.shape[:2])
                crop_offset = scaled_kp[0] - tuple(x / 2 for x in self.slice_size)
                crop_offset = [int(max(min(crop_offset[0], resized_image.shape[1] - self.slice_size[0]), 0)),
                               int(max(min(crop_offset[1], resized_image.shape[0] - self.slice_size[1]), 0))]
                final_kp = scaled_kp - crop_offset
                final_label = resized_label[crop_offset[1]:crop_offset[1] + self.slice_size[0],
                              crop_offset[0]:crop_offset[0] + self.slice_size[1]]
                final_image = resized_image[crop_offset[1]:crop_offset[1] + self.slice_size[0],
                              crop_offset[0]:crop_offset[0] + self.slice_size[1]]

                mask_coords = np.argwhere(final_label == 1)
                label_fname = os.path.join(self.directory, "labels", image_id[:-4] + "_label.png")
                cv2.imwrite(label_fname, final_label)

                img_fname = os.path.join(self.directory, "images", image_id)
                cv2.imwrite(img_fname, final_image)

                # img_yuv = cv2.cvtColor(final_image, cv2.COLOR_BGR2HSV)
                # clahe = cv2.createCLAHE(2.0, (8, 8))
                # img_yuv[:, :, 2] = clahe.apply(img_yuv[:, :, 2])
                # final_image = cv2.cvtColor(img_yuv, cv2.COLOR_HSV2BGR)
                # img_fname = os.path.join(self.output_directory, "images_bright",
                #                          str(res) + '_' + self.data_reader.fname[:-4] + ".png")
                # cv2.imwrite(img_fname, final_image)

                ann_fname = os.path.join(self.directory, "annotations", image_id[:-4] + ".txt")

                with open(ann_fname, 'w') as f:
                    f.write(self.makeXml(mask_coords, final_kp, "charger", final_image.shape[1], final_image.shape[0],
                                         ann_fname, 0, 100, (0, 0)))

            # self.add_image(
            #     "charger",
            #     labeled=True,
            #     image_id=image_id+"_labeled",  # use file name as a unique image id
            #     path=img_fname,
            #     width=final_image.shape[1], height=final_image.shape[0],
            #     mask=os.path.join(self.output_directory, 'labels', str(res) + '_' + image_id + ".png"),
            #     annotation=os.path.join(self.output_directory, 'annotations', str(res) + '_' + image_id + ".txt"))
            self.labeled.append(image_id)
            self.kp = []
            self.poly = []
            print("Saved", label_fname)

    def makeXml(self, mask_coords, keypoints_list,  className, imgWidth, imgHeigth, filename, distance, score, offset):
        rel_xmin = np.min(mask_coords[:, 1])
        rel_ymin = np.min(mask_coords[:, 0])
        rel_xmax = np.max(mask_coords[:, 1])
        rel_ymax = np.max(mask_coords[:, 0])
        xmin = rel_xmin / imgWidth
        ymin = rel_ymin / imgHeigth
        xmax = rel_xmax / imgWidth
        ymax = rel_ymax / imgHeigth
        ann = ET.Element('annotation')
        ET.SubElement(ann, 'folder').text = 'images'
        ET.SubElement(ann, 'filename').text = filename + ".png"
        ET.SubElement(ann, 'path')
        source = ET.SubElement(ann, 'source')
        ET.SubElement(source, 'database').text = "Unknown"
        size = ET.SubElement(ann, 'size')
        ET.SubElement(size, 'width').text = str(imgWidth)
        ET.SubElement(size, 'height').text = str(imgHeigth)
        ET.SubElement(size, 'depth').text = "3"
        ET.SubElement(ann, 'segmented').text = "0"
        ET.SubElement(ann, 'offset_x').text = str(offset[1])
        ET.SubElement(ann, 'offset_y').text = str(offset[0])
        object = ET.SubElement(ann, 'object')
        ET.SubElement(object, 'name').text = className
        ET.SubElement(object, 'pose').text = "Unspecified"
        ET.SubElement(object, 'truncated').text = "0"
        ET.SubElement(object, 'difficult').text = "0"
        bndbox = ET.SubElement(object, 'bndbox')
        ET.SubElement(bndbox, 'xmin').text = str(xmin)
        ET.SubElement(bndbox, 'ymin').text = str(ymin)
        ET.SubElement(bndbox, 'xmax').text = str(xmax)
        ET.SubElement(bndbox, 'ymax').text = str(ymax)
        ET.SubElement(object, 'distance').text = str(distance)
        ET.SubElement(object, 'weight').text = str(score)
        keypoints = ET.SubElement(object, 'keypoints')
        for i, kp in enumerate(keypoints_list):
            xml_kp = ET.SubElement(keypoints, 'keypoint'+str(i))
            ET.SubElement(xml_kp, 'x').text = str(int(kp[0]))
            ET.SubElement(xml_kp, 'y').text = str(int(kp[1]))
        return ET.tostring(ann, encoding='unicode', method='xml')

    def show_mask(self):
        alpha = 0.3
        overlay = self.mask
        imm = cv2.addWeighted(cv2.cvtColor(overlay * 255, cv2.COLOR_GRAY2BGR), alpha, self.frame,
                              1 - alpha, 0)
        cv2.imshow("Mask labeler", imm)
        cv2.waitKey(10)

    def video_click(self, e, x, y, flags, param):
        if e == cv2.EVENT_LBUTTONDOWN:

            self.poly.append([x, y])
            if len(self.poly) > 2:
                self.mask = np.full(self.frame.shape[:2], 0, np.uint8)
                cv2.fillPoly(self.mask, np.array(self.poly, dtype=np.int32)[np.newaxis, :, :], 1)
                self.show_mask()
            if len(self.poly) != 5 and len(self.poly)<8:
                self.kp.append((x, y))

    def query_gt(self, image_id):
        cv2.setMouseCallback("Mask labeler", self.video_click)
        cv2.imshow("Mask labeler", self.frame)
        k = cv2.waitKey(0)
        if k == ord(' '):
            if len(self.poly) != 8 or len(self.kp) != 6:
                print("SELECT KEYPOINTS!", len(self.kp), len(self.poly))
                cv2.waitKey(0)
        mask = np.expand_dims(self.mask, -1) #cv2.imread(info['mask'])
        kp = np.expand_dims(self.kp, axis=0)
        self.save(image_id)

        return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32), kp

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

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        # mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
        #                 dtype=np.uint8)
        mask = cv2.imread(info['mask'])
        # for i, p in enumerate(info["polygons"]):
        #     Get indexes of pixels inside the polygon and set them to 1
            # rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            # mask[rr, cc, i] = 1

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)

    def load_kp(self, image_id):

        info = self.image_info[image_id]
        ann_fname = info['annotation']
        # label_fname = base_path + 'labels/' + fname[:-4] + '_label.png'
        # print(img_fname)

        tree = ET.parse(ann_fname)
        root = tree.getroot()
        keypoints = []
        size = root.find('size')
        w = int(size.find('width').text)
        h = int(size.find('height').text)
        for object in root.findall('object'):
            kps = object.find('keypoints')
            bbox = object.find('bndbox')
            xmin = float(bbox.find('xmin').text)
            ymin = float(bbox.find('ymin').text)
            xmax = float(bbox.find('xmax').text)
            ymax = float(bbox.find('ymax').text)
            bw = xmax-xmin
            bh = ymax-ymin
            for i in range(4):
                kp = kps.find('keypoint' + str(i))
                keypoints.append(((float(kp.find('x').text)-(xmin*w))/bw/w, (float(kp.find('y').text)-(ymin*h))/bh/h))
            # print("keypoints", keypoints)
        return keypoints

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "charger":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    """
    Freezes the state of a session into a pruned computation graph.

    Creates a new computation graph where variable nodes are replaced by
    constants taking their current value in the session. The new graph will be
    pruned so subgraphs that are not necessary to compute the requested
    outputs are removed.
    @param session The TensorFlow session to be frozen.
    @param keep_var_names A list of variable names that should not be frozen,
                          or None to freeze all the variables in the graph.
    @param output_names Names of the relevant graph outputs.
    @param clear_devices Remove the device directives from the graph for better portability.
    @return The frozen graph definition.
    """
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = tf.graph_util.convert_variables_to_constants(
            session, input_graph_def, output_names, freeze_var_names)
        return frozen_graph


def train(model, inf_model, dataset_directory):
    """Train the model."""

    epochs = 200
    for ep in range(1, epochs+1): # TODO allow background training
        # Training dataset.
        dataset_train = ChargerDataset(dataset_directory, 'train')

        dataset_train.add_labeled_images()
        dataset_train.prepare()

        # Validation dataset
        dataset_val = ChargerDataset("/root/share/tf/dataset/online_val", 'val')
        # dataset_val.load_charger("/root/share/tf/dataset/online_val", "val")
        dataset_val.prepare()

        # *** This training schedule is an example. Update to your needs ***
        # Since we're using a very small dataset, and starting from
        # COCO trained weights, we don't need to train too long. Also,
        # no need to train all layers, just the heads should do it.
        tr_steps = len(dataset_train.image_info)
        print("Training network heads")

        # st_thread = threading.Thread(target=model.train, args=(dataset_train, dataset_val),
        #                              kwargs={'learning_rate':config.LEARNING_RATE,
        #                                      'epochs':1,
        #                                      'steps':tr_steps,
        #                                      'layers':'heads'})
        # st_thread.start()
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    epochs=1,
                    steps=tr_steps,
                    layers='heads')
        image, image_id = dataset_train.get_unlabeled()
        r = inf_model.detect([image], verbose=1)[0]
        # st_thread = threading.Thread(target=dataset_train.self_train, args=(r, image, image_id))
        # st_thread.start()
        dataset_train.self_train(r, image, image_id)
        # st_thread.join()
        inf_model.load_weights(train_model.find_last(), by_name=True)


def color_splash(image, mask):
    """Apply color splash effect.
    image: RGB image [height, width, 3]
    mask: instance segmentation mask [height, width, instance count]

    Returns result image.
    """
    # Make a grayscale copy of the image. The grayscale copy still
    # has 3 RGB channels, though.
    gray = np.zeros_like(image) #skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
    # Copy color pixels from the original color image where mask is set
    if mask.shape[-1] > 0:
        # We're treating all instances as one, so collapse the mask into one layer
        mask = (np.sum(mask, -1, keepdims=True) >= 1)
        splash = np.where(mask, gray, image).astype(np.uint8)
    else:
        splash = gray.astype(np.uint8)
    return splash


def detect_and_color_splash(model, image_path=None, video_path=None):
    assert image_path or video_path
    save = False
    if save:
        frozen_graph = freeze_session(keras.backend.get_session(),
                                      output_names=[out.op.name for out in model.keras_model.outputs])
        tf.train.write_graph(frozen_graph, "/root/share/tf/Keras", "frozen_inference_graph.pb", as_text=True)

    # Image or video?
    if image_path:
        images = os.listdir(image_path)
        for im in images:
            # Run model detection and generate the color splash effect
            # print("Running on {}".format(args.image))
            # Read image
            image = skimage.io.imread(os.path.join(image_path, im))
            # Detect objects
            r = model.detect([image], verbose=1)[0]
            # Color splash
            splash = color_splash(image, r['masks'])
            kps = r['kp'][0][0]
            if len(r['rois'])==0:
                continue
            roi = r['rois'][0]
            bw = roi[3]-roi[1]
            bh = roi[2]-roi[0]
            for i in range(8):
                cv2.circle(splash, (int(kps[i*2]*bw)+roi[1], int(kps[i*2+1]*bh+roi[0])), 20, (0, 0, 255), -1)
            cv2.imshow('lol', cv2.resize(splash, (1280, 960)))
            # attention = r['attention']
            # attention = (attention+abs(np.min(attention)))/(abs(np.min(attention))+abs(np.max(attention)))
            # attention = np.squeeze(attention.astype(np.uint8))
            # print(np.max(attention))
            # print(np.min(attention))
            # print(attention.shape)
            # attention = np.transpose(np.squeeze(attention), [2, 0, 1])
            # for i in range(5):
            #     cv2.imshow('att', attention[i])
            #     k = cv2.waitKey(0)

            k = cv2.waitKey(0)
            if k==ord('q'):
                exit(0)
            # Save output
            # file_name = "splash_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
            # skimage.io.imsave(file_name, splash)
    elif video_path:
        # Video capture
        vcapture = cv2.VideoCapture(video_path)
        width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = vcapture.get(cv2.CAP_PROP_FPS)

        # Define codec and create video writer
        file_name = "splash_{:%Y%m%dT%H%M%S}.avi".format(datetime.datetime.now())
        # vwriter = cv2.VideoWriter(file_name,
        #                           cv2.VideoWriter_fourcc(*'MJPG'),
        #                           fps, (width, height))

        count = 0
        success = True
        while success:
            print("frame: ", count)
            # Read next image
            success, image = vcapture.read()
            if success:
                # OpenCV returns images as BGR, convert to RGB
                image = image[..., ::-1]
                # Detect objects
                r = model.detect([image], verbose=0)[0]
                print(r)
                # Color splash
                splash = color_splash(image, r['masks'])
                # RGB -> BGR to save image to video
                splash = splash[..., ::-1]
                cv2.imshow('lol', cv2.resize(splash, (1280, 960)))
                cv2.waitKey(0)
                # Add image to video writer
                # vwriter.write(splash)
                count += 1
        # vwriter.release()
    print("Saved to ", file_name)


############################################################
#  Training
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect chargers.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'splash'")
    parser.add_argument('--dataset', required=False,
                        metavar="/root/share/tf/dataset/mask_front_kp/GP_A8",
                        help='Directory of the charger dataset')
    parser.add_argument('--weights', required=True,
                        metavar="./mask_rcnn_coco.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image to apply the color splash effect on')
    parser.add_argument('--video', required=False,
                        metavar="path or URL to video",
                        help='Video to apply the color splash effect on')
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "splash":
        assert args.image or args.video,\
               "Provide --image or --video to apply color splash"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = chargerConfig()
    else:
        class InferenceConfig(chargerConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = InferenceConfig()
    config.display()

    # Create model
    # if args.command == "train":
    #     model = modellib.MaskRCNN(mode="training", config=config,
    #                               model_dir=args.logs, num_points=config.NUM_POINTS)
    # else:
    #     model = modellib.MaskRCNN(mode="inference", config=config,
    #                               model_dir=args.logs, num_points=config.NUM_POINTS)

    train_model = modellib.MaskRCNN(mode="training", config=config,
                                    model_dir=args.logs, num_points=config.NUM_POINTS)
    inf_model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs, num_points=config.NUM_POINTS)


    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = train_model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = train_model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        train_model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        train_model.load_weights(weights_path, by_name=True)
        inf_model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(train_model, inf_model, args.dataset)
    elif args.command == "splash":
        detect_and_color_splash(train_model, image_path=args.image,
                                video_path=args.video)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'splash'".format(args.command))
