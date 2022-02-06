import numpy as np
import cv2
from pl_charger.charger_kpts_r50_1l_HM1 import ChargerKptsR50L1HM1
from YOLO.yolo import YOLO
from PIL import Image
from sklearn.cluster import DBSCAN
import torch
try:
    from cv2 import cv2
except ImportError:
    pass
from timebudget import timebudget

timebudget.report_at_exit()


class ChargerConfig():

    DETECTION_MIN_CONFIDENCE = 0.9
    NUM_POINTS = 4
    IMAGE_MAX_DIM = 512


class Detector:

    def __init__(self, path_to_model, path_to_pole_model, num_points):
        np.set_printoptions(suppress=True)
        np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
        self.config = ChargerConfig()
        self.config.NUM_POINTS = num_points

        # Size of image passed to network defined above as IMAGE_MAX_DIM. Should be the asme as used during training
        self.slice_size = (self.config.IMAGE_MAX_DIM, self.config.IMAGE_MAX_DIM)
        # Top left corner of ROI with charging pole
        self.offset = (0, 0)
        # scaling factors in x and y axis used to fit ROI to IMAGE_MAX_DIMxIMAGE_MAX_DIM size
        self.scale = (1.0, 1.0)
        self.detections = []
        self.best_detection = None
        self.bbox = None
        self.frame_shape = None
        self.moving_avg_image = None
        self.bottom = False

        # Init YOLO model using path to weights. Should be trained using keras-yolo3 repo
        self.yolo = YOLO(model_path=path_to_pole_model)

        # Init pl model
        model = ChargerKptsR50L1HM1.load_from_checkpoint(path_to_model, gpus=1, precision=16, device='cuda')
        model.eval()     
        self.det_model = model.to("cuda")


    def init_size(self, shape):
        self.frame_shape = shape
        self.moving_avg_image = np.full((self.config.NUM_POINTS, shape[0], shape[1]), 0.0, dtype=np.float64)

    
    def get_CNN_output(self, image_np):
        image_np = np.expand_dims(image_np, 0)
        image_np = np.transpose(image_np, [0,3,1,2])
        image_tensor = torch.from_numpy(image_np).float()/255

        with timebudget("Detection"):  
            r =  self.det_model(image_tensor.to("cuda"))
        
        uncertainty = None 
        hm = r.detach().cpu().numpy().squeeze()
        heatmap_uncertainty = []
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
        num_clusters = min(self.config.NUM_POINTS, len(cluster_scores))
        preds = np.zeros((self.config.NUM_POINTS, 2))
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
        # print("absolute_kp", absolute_kp)
        self.best_detection = dict(keypoints=preds, uncertainty=uncertainty,
                                   heatmap_uncertainty=np.array(heatmap_uncertainty), bbox=np.array(self.bbox))

    @timebudget
    def detect(self, frame):
        """Actual detection is divided at two parts. Full image is resized to IMAGE_MAX_DIM size.
        This image is sent to YOLO detector to get coordinates of charging pole. Then from image at full
        resolution ROI from YOLO output is cropped and sent to keypoint detector. Croppeed
        image is resized to match IMAGE_MAX_DIMxIMAGE_MAX_DIM size"""
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

        try:
            frame = cv2.resize(frame[ymin:ymax, xmin:xmax], self.slice_size)
            # cv2.imshow("yolo", frame)
        except cv2.error as e:
            # print(e.msg)
            return
        self.get_CNN_output(frame)
        if self.best_detection is None:
            print("No detections")
        return
