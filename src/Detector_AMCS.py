import imp
import numpy as np
import cv2
import torch
from torchvision import transforms
from YOLO.yolo import YOLO
from PIL import Image
try:
    from cv2 import cv2
except ImportError:
    pass
from timebudget import timebudget

timebudget.report_at_exit()

from keypoints.src.model import Keypoints
from keypoints.src.prediction import Prediction
from keypoints.config import NUM_CLASSES, IMG_SMALL_HEIGHT, IMG_SMALL_WIDTH

class DetectorAMCS:

    def __init__(self, path_to_model, path_to_pole_model=None, num_points=4, cuda=True):
        np.set_printoptions(suppress=True)
        np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
        # self.slice_size = (257, 257)
        self.slice_size = (353, 353)
        self.scale = (1.0, 1.0)
        self.num_points = num_points
        self.cuda = cuda


        kpts = Keypoints(NUM_CLASSES, img_height=self.slice_size[0], img_width=self.slice_size[1])
        kpts.load_state_dict(torch.load(path_to_model))
        if self.cuda:
            torch.cuda.set_device(0)
            kpts = kpts.cuda()

        model = Prediction(kpts, NUM_CLASSES, self.slice_size[0], self.slice_size[1], IMG_SMALL_HEIGHT, IMG_SMALL_WIDTH, self.cuda)
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Resize([self.slice_size[0], self.slice_size[1]])])

        self.det_model = model
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


        img_t = self.transform(frame)
        if self.cuda:
            img_t = img_t.cuda()
        result, keypoints = self.det_model.predict(img_t)

        keypoints = keypoints.cpu().numpy()
        keypoints = keypoints * [self.slice_size[0]/IMG_SMALL_HEIGHT, self.slice_size[1]/IMG_SMALL_WIDTH]
        # print(keypoints.shape, keypoints)
        absolute_kp = []
        relative_kp = []
        for i, kp in enumerate(keypoints[0]):
            absolute_kp.append(
                ((kp[1] * self.scale[0] + bbox[0]),
                (kp[0] * self.scale[1] + bbox[1])))
            relative_kp.append(
                ((kp[0] * self.scale[0]),
                (kp[1] * self.scale[1])))

        self.best_detection = dict(keypoints=np.array(absolute_kp)[:, :2], rel_keypoints=np.array(relative_kp), uncertainty=[],
                                   heatmap_uncertainty=np.array([]), bbox=bbox)
