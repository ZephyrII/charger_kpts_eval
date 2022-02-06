from matplotlib.transforms import Bbox
import numpy as np
import cv2
from tensorflow.python.eager.context import device
from torch.utils import data
from pl_charger.charger_kpts import ChargerKpts
from YOLO.yolo import YOLO
from PIL import Image
from sklearn.cluster import DBSCAN
import torch
import time
try:
    from cv2 import cv2
except ImportError:
    pass
from timebudget import timebudget

timebudget.report_at_exit()
import warnings

from mmcv import Config
import torch
from mmpose.apis import init_pose_model

from mmpose.apis.inference import _inference_single_pose_model
from mmpose.datasets import DatasetInfo

class DetectorMmposeDB_HM1:

    def __init__(self, path_to_model, path_to_pole_model=None, num_points=4):
        np.set_printoptions(suppress=True)
        np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
        self.slice_size = (512, 512)
        self.scale = (1.0, 1.0)
        self.num_points = num_points

        # cfg = Config.fromfile("/root/mmpose/configs/charger/hrnet_w48_charger_tape_slice_960x960.py")
        # cfg = Config.fromfile("/root/mmpose/configs/charger/scnet101_charger_corners_960x960_clean.py")
        cfg = Config.fromfile("/root/mmpose/configs/charger/hrnet_udp_bl.py")

        self.det_model = init_pose_model(
            cfg, path_to_model, device='cuda')
        self.cfg = cfg
        self.yolo = YOLO(model_path=path_to_pole_model)


    def init_size(self, shape):
        self.frame_shape = shape
        self.moving_avg_image = np.full((self.num_points, shape[0], shape[1]), 0.0, dtype=np.float64)

    def detect(self, frame):

        self.detections = []
        self.best_detection = None

        small_frame = cv2.resize(frame, (self.slice_size[1], self.slice_size[0]))
        ymin, xmin, ymax, xmax = self.yolo.detect_image(Image.fromarray(small_frame))
        ymin = int(ymin * frame.shape[0] / self.slice_size[0])
        ymax = int(ymax * frame.shape[0] / self.slice_size[0])
        xmin = int(xmin * frame.shape[1] / self.slice_size[1])
        xmax = int(xmax * frame.shape[1] / self.slice_size[1])
        margin = (xmax - xmin) * 0.1, (ymax - ymin) * 0.5

        xmin -= int(margin[0])
        ymin -= int(margin[1])
        xmax += int(margin[0])
        ymax += int(margin[1])

        xmin = np.max([xmin, 0])
        ymin = np.max([ymin, 0])
        xmax = np.min([xmax, self.frame_shape[1]])
        ymax = np.min([ymax, self.frame_shape[0]])
        self.scale = ((xmax - xmin) / self.slice_size[0], (ymax - ymin) / self.slice_size[1])
        self.bbox = [xmin, ymin, xmax, ymax]

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        try:
            frame = cv2.resize(frame[ymin:ymax, xmin:xmax], self.slice_size)
            # cv2.imshow("yolo", frame)
        except cv2.error as e:
            # print(e.msg)
            return

        cfg = Config.fromfile('/root/mmpose/configs/_base_/datasets/charger_tape.py')["dataset_info"]
        di = DatasetInfo(cfg)
        pred, heatmaps = _inference_single_pose_model(self.det_model, frame, [[0,0, self.slice_size[0], self.slice_size[1]]], dataset_info=di, return_heatmap=True)
        hm = np.sum(heatmaps[0], axis=0)
        hm = cv2.resize(hm, (0,0), fx=4, fy=4)
        absolute_kp = []
        raw_kp = hm
        kp = (hm - np.min(hm)) / (np.max(hm) - np.min(hm))
        h, w = kp.shape
        # Binarize kp heatmaps
        ret, kp = cv2.threshold(kp, 0.2, 1.0, cv2.THRESH_BINARY)
        # Get coordinates of white pixels
        X = np.argwhere(kp == 1)
        if X.shape[0] == 0:
            preds = [(0, 0), (0, 0), (0, 0), (0, 0)]
        # Init algorithm for clustering
        clustering = DBSCAN(eps=3, min_samples=2, n_jobs=8)
        # clustering = KMeans(n_clusters=4)
        clustering.fit(X)
        cluster_scores = []
        # Get labels of all clusters

        unique_labels = np.unique(clustering.labels_)

        # for each cluster calculate their "score" by summing values of all pixels in cluster
        for id in np.unique(clustering.labels_):
            cluster = X[np.where(clustering.labels_ == id)]
            cluster_scores.append(np.sum(raw_kp[cluster[:, 0], cluster[:, 1]]))

            # Get pixels of cluster with max score
        num_clusters = min(4, len(cluster_scores))
        preds = np.zeros((4, 2))
        for id, cl_id in enumerate(np.argpartition(cluster_scores, -num_clusters)[-num_clusters:]):
            cluster = X[np.where(clustering.labels_ == unique_labels[cl_id])]
            mask = np.zeros_like(kp)
            mask[cluster[:, 0], cluster[:, 1]] = raw_kp[cluster[:, 0], cluster[:, 1]]

            if np.sum(mask) == 0:
                center = (0, 0)
            else:
                center = np.average(np.sum(mask, axis=1) * np.arange(w)) / np.sum(mask) * w, \
                        np.average(np.sum(mask, axis=0) * np.arange(h)) / np.sum(mask) * h
            preds[id] = ((center[1] * self.scale[0] + self.bbox[0]),
                        (center[0] * self.scale[1] + self.bbox[1]))
        preds = np.array(preds)
        id_3, id_0 = np.argpartition(-preds[:, 0], -2)[-2:]
        id_1 = np.argmax(preds[:, 1])
        id_2 = list(set([0,1,2,3])-set([id_0, id_1, id_3]))[0]
        preds = preds[[id_0, id_1, id_2, id_3]]

        self.best_detection = dict(keypoints=np.array(preds)[:, :2], uncertainty=[],

                                   heatmap_uncertainty=np.array([]), bbox=self.bbox)
