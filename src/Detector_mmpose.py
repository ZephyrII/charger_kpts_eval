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
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint
from mmpose.apis import init_pose_model

from mmpose.apis import single_gpu_test
from mmpose.apis.inference import _inference_single_pose_model
from mmpose.datasets import build_dataloader, build_dataset, DatasetInfo
from mmpose.models import build_posenet

from torch.utils.data import TensorDataset
try:
    from mmcv.runner import wrap_fp16_model
except ImportError:
    warnings.warn('auto_fp16 from mmpose will be deprecated from v0.15.0'
                  'Please install mmcv>=1.1.4')
    from mmpose.core import wrap_fp16_model

class DetectorMmpose:

    def __init__(self, path_to_model, path_to_pole_model=None, num_points=4):
        np.set_printoptions(suppress=True)
        np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
        self.slice_size = (960, 960)
        self.scale = (1.0, 1.0)

        cfg = Config.fromfile("/root/mmpose/configs/charger/hrnet_w48_charger_tape_384x288.py")

        # # if args.cfg_options is not None:
        # #     cfg.merge_from_dict(args.cfg_options)

        # # # set cudnn_benchmark
        # # if cfg.get('cudnn_benchmark', False):
        # #     torch.backends.cudnn.benchmark = True
        # cfg.model.pretrained = None
        # cfg.data.test.test_mode = True

        # model = build_posenet(cfg.model)
        # fp16_cfg = cfg.get('fp16', None)
        # if fp16_cfg is not None:
        #     wrap_fp16_model(model)
        # load_checkpoint(model, path_to_model, map_location='cpu')
        # self.det_model = MMDataParallel(model, device_ids=[0])

        self.det_model = init_pose_model(
            cfg, path_to_model, device='cuda')
        self.cfg = cfg
        self.yolo = YOLO(model_path=path_to_pole_model)


    def detect(self, frame):

        self.detections = []
        self.best_detection = None

        small_frame = cv2.resize(frame, (self.slice_size[1], self.slice_size[0]))
        ymin, xmin, ymax, xmax = self.yolo.detect_image(Image.fromarray(small_frame))
        ymin = int(ymin * frame.shape[0] / self.slice_size[0])
        ymax = int(ymax * frame.shape[0] / self.slice_size[0])
        xmin = int(xmin * frame.shape[1] / self.slice_size[1])
        xmax = int(xmax * frame.shape[1] / self.slice_size[1])
        # margin = (xmax - xmin) * 0.1, (ymax - ymin) * 0.1

        # xmin -= int(margin[0])
        # ymin -= int(margin[1])
        # xmax += int(margin[0])
        # ymax += int(margin[1])
        self.scale = ((xmax - xmin) / self.slice_size[0], (ymax - ymin) / self.slice_size[1])
        bbox = [xmin, ymin, xmax-xmin, ymax-ymin]


        image_np = np.expand_dims(frame, 0)
        # image_np = np.transpose(image_np, [0,3,1,2])
        # image_tensor = torch.from_numpy(image_np).float()#/255
        cfg = Config.fromfile('/root/mmpose/configs/_base_/datasets/charger_tape.py')["dataset_info"]
        di = DatasetInfo(cfg)
        pred, heatmaps = _inference_single_pose_model(self.det_model, frame, [bbox], dataset_info=di)

        print("pred", pred[:,:, :2])
        


        # print("absolute_kp", absolute_kp)
        self.best_detection = dict(keypoints=pred, uncertainty=[],
                                   heatmap_uncertainty=np.array([]))
