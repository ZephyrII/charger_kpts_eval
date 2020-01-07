
import skimage.draw
import numpy as np
import sys
import os
import cv2
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


class InferenceConfig(Config):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    # Give the configuration a recognizable name
    NAME = "att"

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + charger

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 5

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9


config = InferenceConfig()
config.display()

model = modellib.MaskRCNN(mode="inference", config=config, model_dir=DEFAULT_LOGS_DIR)

weights_path = COCO_WEIGHTS_PATH
if not os.path.exists(weights_path):
    utils.download_trained_weights(weights_path)

model.load_weights(weights_path, by_name=True, exclude=[
        "mrcnn_class_logits", "mrcnn_bbox_fc",
        "mrcnn_bbox", "mrcnn_mask"])

path = "/root/share/imagenet/train2014/"
dst = "/root/share/imagenet/target2014/"
images = os.listdir(path)
for im in images:
    print(im)

    image = cv2.imread(os.path.join(path, im), cv2.IMREAD_COLOR)

    r = model.detect([image], verbose=1)[0]
    attention = r['attention']
    attention = (attention + abs(np.min(attention))) / (abs(np.min(attention)) + abs(np.max(attention)))
    # attention = np.squeeze(attention.astype(np.uint8))
    print(np.max(attention))
    print(np.min(attention))
    print(attention.shape)
    attention = np.transpose(np.squeeze(attention), [2, 0, 1])
    # for i in range(5):
    np.save(os.path.join(dst, im[:-4]), attention)
        # cv2.imwrite(os.path.join(dst, im[:-4]+"_i.jpg"), attention[i])
        # cv2.imshow('att', attention[i])
        # k = cv2.waitKey(0)
        # if k==ord('q'):
        #     exit(0)
