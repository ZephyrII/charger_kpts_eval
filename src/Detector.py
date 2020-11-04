import tensorflow as tf
import numpy as np
import cv2
import time
from mrcnn.config import Config
from mrcnn import model as modellib
from YOLO.yolo import YOLO
from PIL import Image
from sklearn.cluster import DBSCAN

try:
    from cv2 import cv2
except ImportError:
    pass


class ChargerConfig(Config):

    NAME = "charger"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 1  # Background + charger
    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9
    IMAGE_MAX_DIM = 960


class Detector:

    def __init__(self, path_to_model, path_to_pole_model, num_points_front, path_to_model_bottom=None):
        np.set_printoptions(suppress=True)
        np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
        self.config = ChargerConfig()
        self.config.NUM_POINTS = num_points_front
        self.config.display()
        # Size of image passed to network defined above as IMAGE_MAX_DIM. Should be the asme as used during training
        self.slice_size = (self.config.IMAGE_MAX_DIM, self.config.IMAGE_MAX_DIM)
        # Top left corner of ROI with charging pole
        self.offset = (0, 0)
        # scaling factors in x and y axis used to fit ROI to IMAGE_MAX_DIMxIMAGE_MAX_DIM size
        self.scale = (1.0, 1.0)
        # self.charger_to_slice_ratio = 0.5

        self.detections = []
        self.best_detection = None
        self.bbox = None
        self.frame_shape = None
        self.moving_avg_image = None
        self.bottom = False

        # Init YOLO model using path to weights. Should be trained using keras-yolo3 repo
        self.yolo = YOLO(model_path=path_to_pole_model)

        # Get weights and init keypoints network
        if path_to_model.endswith(".h5"):
            weights_path = path_to_model
            path_to_model = "/".join(path_to_model.split("/")[:-1])
            model = modellib.MaskRCNN(mode="inference", config=self.config, model_dir=path_to_model)
        else:
            model = modellib.MaskRCNN(mode="inference", config=self.config, model_dir=path_to_model)
            weights_path = model.find_last()

        # Load weights
        model.load_weights(weights_path, by_name=True)
        self.det_model = model

        # Init model used by back camera if path to weights is provided
        if path_to_model_bottom is not None:
            self.config.NUM_POINTS = 4
            model_bottom = modellib.MaskRCNN(mode="inference", config=self.config, model_dir=path_to_model_bottom)
            weights_path_bottom = model_bottom.find_last()

            # Load weights
            print("Loading weights ", weights_path_bottom)
            model_bottom.load_weights(weights_path_bottom, by_name=True)
            self.det_model_bottom = model_bottom


    def init_size(self, shape):
        self.frame_shape = shape
        self.moving_avg_image = np.full((self.config.NUM_POINTS, shape[0], shape[1]), 0.0, dtype=np.float64)

    def get_CNN_output(self, image_np):
        if self.bottom:
            r = self.det_model_bottom.detect([image_np], verbose=0)
        else:
            r = self.det_model.detect([image_np], verbose=0)
        uncertainty = r['uncertainty'][0][0]
        kps = r['kp'][0][0]
        # Change dimensions order from [x, y, num_kp] to [num_kp, x, y]
        kps = np.transpose(kps, (2, 0, 1))
        # Buffers for kp coordinates in full, not scaled image and uncertainty from heatmap
        absolute_kp = []
        heatmap_uncertainty = []

        for i, kp in enumerate(kps):
            raw_kp = kp
            # subtract all other heatmaps from current heatmap. Use when more than one detection is on the same point on image
            # background = kps
            # background = np.sum(np.delete(background, i, 0), axis=0)
            # kp = kp - background

            # add to current heatmap moving average from last detections. Use to avoid false detections
            # far from from last detection. Uncomment line 124 to update moving avg
            # xmin, ymin, xmax, ymax = self.bbox
            # alpha = 0.8
            # print("dtypes", kp.astype(np.float64).shape, self.moving_avg_image[i, ymin:ymax, xmin:xmax].shape)
            # kp = cv2.addWeighted(kp.astype(np.float64), alpha,
            #                      cv2.resize(self.moving_avg_image[i, ymin:ymax, xmin:xmax], self.slice_size), 1 - alpha,
            #                      0.0)

            # Scale to 0-1 values to do clustering
            kp = (kp - np.min(kp)) / (np.max(kp) - np.min(kp))
            # cv2.imshow("kp", kp)
            # cv2.waitKey(0)
            h, w = kp.shape
            # Binarize kp heatmaps
            ret, kp = cv2.threshold(kp, 0.2, 1.0, cv2.THRESH_BINARY)
            # Get coordinates of white pixels
            X = np.argwhere(kp == 1)
            if X.shape[0] == 0:
                absolute_kp.append((0, 0))
                heatmap_uncertainty = np.array([[np.inf, np.inf], [np.inf, np.inf]])
                continue
            # Init algorithm for clustering
            clustering = DBSCAN(eps=3, min_samples=2)
            clustering.fit(X)
            cluster_scores = []
            # Get labels of all clusters
            unique_labels = np.unique(clustering.labels_)

            # for each cluster calculate their "score" by summing values of all pixels in cluster
            for id in np.unique(clustering.labels_):
                cluster = X[np.where(clustering.labels_ == id)]
                cluster_scores.append(np.sum(raw_kp[cluster[:, 0], cluster[:, 1]]))
            # Get pixels of cluster with max score
            cluster = X[np.where(clustering.labels_ == unique_labels[np.argmax(cluster_scores)])]
            mask = np.zeros_like(kp)
            mask[cluster[:, 0], cluster[:, 1]] = raw_kp[cluster[:, 0], cluster[:, 1]]

            # self.update_moving_avg(i, mask)
            # Get uncertainty of kp prediction based on shape of cluster
            if mask[mask != 0].reshape(-1).shape == 0:
                heatmap_uncertainty = np.full((2, 2), np.inf)
            else:
                heatmap_uncertainty.append(np.cov(cluster.T, aweights=mask[mask != 0].reshape(-1)))
            # Get weighted center of mass of cluster which is prediceted keypoint
            if np.sum(mask) == 0:
                center = (0, 0)
            else:
                center = np.average(np.sum(mask, axis=1) * np.arange(w)) / np.sum(mask) * w, \
                         np.average(np.sum(mask, axis=0) * np.arange(h)) / np.sum(mask) * h
            absolute_kp.append(
                ((center[1] * self.scale[0] + self.bbox[0]),
                 (center[0] * self.scale[1] + self.bbox[1])))

        self.best_detection = dict(keypoints=absolute_kp, uncertainty=uncertainty,
                                   heatmap_uncertainty=np.array(heatmap_uncertainty))

    def update_moving_avg(self, i, mask):
        """Updates map which stores historical detection. It should reduce false positives using historical data
        The map has size of full image,at each step, the area around detection is highlighted with higher probability
        of detection. Old detections decay over time (variable decay can control speed of decay)"""
        decay = 0.5
        xmin, ymin, xmax, ymax = self.bbox
        overlay = np.zeros(self.frame_shape, dtype=np.float64)
        mk = cv2.resize(mask, (xmax - xmin, ymax - ymin))
        # increasing size of area to highlight. Optical flow should work better.
        ret, mk = cv2.threshold(mk, 0.5, 1.0, cv2.THRESH_BINARY)
        mk = cv2.dilate(mk, np.ones((10, 10), np.uint8), iterations=15)
        mk = cv2.blur(mk, (50, 50))
        try:
            overlay[ymin:ymax, xmin:xmax] = cv2.resize(mk, (xmax - xmin, ymax - ymin))
        except ValueError:
            print("error", xmin, ymin, xmax, ymax, overlay.shape)
        cv2.accumulateWeighted(overlay, self.moving_avg_image[i], decay)
        # cv2.imshow("ma", cv2.resize(self.moving_avg_image[i], (960, 960)))

    def detect(self, frame):
        """Actual detection is divided at two parts. Full image is resized to IMAGE_MAX_DIM size.
        This image is sent to YOLO detector to get coordinates of charging pole. Then from image at full
        resolution ROI from YOLO output is cropped and sent to keypoint detector. Croppeed
        image is resized to match IMAGE_MAX_DIMxIMAGE_MAX_DIM size"""
        self.detections = []
        self.best_detection = None

        # Testing ARTags
        # self.bbox = [0, 0, 512, 512]
        # self.scale = (1, 1)
        # self.get_CNN_output(frame)

        # self.init_detection(frame)
        small_frame = cv2.resize(frame, (self.slice_size[1], self.slice_size[0]))

        ymin, xmin, ymax, xmax = self.yolo.detect_image(Image.fromarray(small_frame))
        ymin = int(ymin * frame.shape[0] / self.slice_size[0])
        ymax = int(ymax * frame.shape[0] / self.slice_size[0])
        xmin = int(xmin * frame.shape[1] / self.slice_size[1])
        xmax = int(xmax * frame.shape[1] / self.slice_size[1])
        margin = (xmax - xmin) * 0.1, (ymax - ymin) * 0.1

        xmin -= int(margin[0])
        ymin -= int(margin[1])
        xmax += int(margin[0])
        ymax += int(margin[1])
        self.scale = ((xmax - xmin) / self.slice_size[0], (ymax - ymin) / self.slice_size[1])
        self.bbox = [xmin, ymin, xmax, ymax]

        try:
            frame = cv2.resize(frame[ymin:ymax, xmin:xmax], self.slice_size)
            cv2.imshow("yolo", frame)
        except cv2.error as e:
            # print(e.msg)
            return
        self.get_CNN_output(frame)
        if self.best_detection is None:
            print("No detections")
        return
