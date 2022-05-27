import numpy as np
import cv2
from YOLO.yolo import YOLO
from PIL import Image
try:
    from cv2 import cv2
except ImportError:
    pass
from timebudget import timebudget

timebudget.report_at_exit()

from mmcv import Config
from mmpose.apis import init_pose_model

from mmpose.apis.inference import _inference_single_pose_model
from mmpose.datasets import  DatasetInfo
import time

class DetectorMmposeUnc:

    def __init__(self, path_to_model, path_to_pole_model=None, num_points=4):
        np.set_printoptions(suppress=True)
        np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
        self.slice_size = (512, 512)
        self.scale = (1.0, 1.0)
        self.num_points = num_points

        # cfg = Config.fromfile("/root/mmpose/configs/charger/c_hrnet32_udp_512_hm128.py")
        cfg = Config.fromfile("/root/mmpose/configs/charger/c_hrnet32_udp_512_hm128_unc.py")
        # cfg = Config.fromfile("/root/mmpose/configs/charger/c_hrnet32_udp_512_hm512_repr_v2.py")

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

        self.scale = ( frame.shape[0]/ self.slice_size[0], frame.shape[1] / self.slice_size[1])
        frame = cv2.resize(frame, self.slice_size)

        cfg = Config.fromfile('/root/mmpose/configs/_base_/datasets/charger_tape.py')["dataset_info"]
        di = DatasetInfo(cfg)
        tic = time.perf_counter()
        result = _inference_single_pose_model(self.det_model, frame, [[0,0, self.slice_size[0], self.slice_size[1]]], dataset_info=di, return_heatmap=False)
        toc = time.perf_counter()
        print(f"MMP time: : {toc - tic:0.4f} seconds")
        pred = result["preds"]
        heatmaps = result["output_heatmap"]
        #uncertainty
        uncertainty = result["output_uncertainty"][0]
        # uncertainty = np.zeros(64)
        bbox = [0,0,0,0]

        absolute_kp = []
        relative_kp = []
        for i, kp in enumerate(pred[0, :, :2]):
            absolute_kp.append(
                ((kp[0] * self.scale[0] + bbox[0]),
                (kp[1] * self.scale[1] + bbox[1])))
            relative_kp.append(
                ((kp[0] * self.scale[0]),
                (kp[1] * self.scale[1])))

        self.best_detection = dict(keypoints=np.array(absolute_kp)[:, :2], rel_keypoints=np.array(relative_kp), uncertainty=uncertainty,
                                   heatmap_uncertainty=np.array([]), bbox=bbox, scale=self.scale)
