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

class DetectorMmpose:

    def __init__(self, path_to_model, path_to_pole_model=None, num_points=4):
        np.set_printoptions(suppress=True)
        np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
        self.slice_size = (512, 512)
        self.scale = (1.0, 1.0)
        self.num_points = num_points

        # cfg = Config.fromfile("/root/mmpose/configs/charger/hrnet_w48_charger_tape_slice_960x960.py")
        cfg = Config.fromfile("/root/mmpose/configs/charger/c_hrnet48_udp_512_hm512.py")
        # cfg = Config.fromfile("/root/mmpose/configs/charger/c_hrnet48_udp_512_hm128_c2.py")

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
        bbox = [xmin, ymin, xmax, ymax]

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        try:
            frame = cv2.resize(frame[ymin:ymax, xmin:xmax], self.slice_size)
            # cv2.imshow("yolo", frame)
        except cv2.error as e:
            # print(e.msg)
            return

        cfg = Config.fromfile('/root/mmpose/configs/_base_/datasets/charger_tape.py')["dataset_info"]
        di = DatasetInfo(cfg)
        pred, heatmaps = _inference_single_pose_model(self.det_model, frame, [[0,0, self.slice_size[0], self.slice_size[1]]], dataset_info=di)

        absolute_kp = []
        for i, kp in enumerate(pred[0, :, :2]):
            absolute_kp.append(
                ((kp[0] * self.scale[0] + bbox[0]),
                (kp[1] * self.scale[1] + bbox[1])))

        self.best_detection = dict(keypoints=np.array(absolute_kp)[:, :2], uncertainty=[],
                                   heatmap_uncertainty=np.array([]), bbox=bbox)
