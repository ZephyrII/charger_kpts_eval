# from utils.app_utils import draw_boxes_and_labels
import tensorflow as tf
import numpy as np
import cv2
import six
import collections
from mrcnn.config import Config
from mrcnn import model as modellib, utils

try:
    from cv2 import cv2
except ImportError:
    pass

def draw_boxes_and_labels(
        boxes,
        classes,
        scores,
        instance_masks=None,
        keypoints=None,
        max_boxes_to_draw=None,
        min_score_thresh=.5,
        agnostic_mode=False):
    """Returns boxes coordinates, class names and colors

    Args:
        boxes: a numpy array of shape [N, 4]
        classes: a numpy array of shape [N]
        scores: a numpy array of shape [N] or None.  If scores=None, then
        this function assumes that the boxes to be plotted are groundtruth
        boxes and plot all boxes as black with no classes or scores.
        category_index: a dict containing category dictionaries (each holding
        category index `id` and category name `name`) keyed by category indices.
        instance_masks: a numpy array of shape [N, image_height, image_width], can
        be None
        keypoints: a numpy array of shape [N, num_keypoints, 2], can
        be None
        max_boxes_to_draw: maximum number of boxes to visualize.  If None, draw
        all boxes.
        min_score_thresh: minimum score threshold for a box to be visualized
        agnostic_mode: boolean (default: False) controlling whether to evaluate in
        class-agnostic mode or not.  This mode will display scores but ignore
        classes.
    """
    # Create a display string (and color) for every box location, group any boxes
    # that correspond to the same location.
    box_to_display_str_map = collections.defaultdict(list)
    box_to_color_map = collections.defaultdict(str)
    box_to_instance_masks_map = {}
    box_to_keypoints_map = collections.defaultdict(list)
    if not max_boxes_to_draw:
        max_boxes_to_draw = boxes.shape[0]
    for i in range(min(max_boxes_to_draw, boxes.shape[0])):
        if scores is None or scores[i] > min_score_thresh:
            box = tuple(boxes[i].tolist())
            if instance_masks is not None:
                box_to_instance_masks_map[box] = instance_masks[i]
            if keypoints is not None:
                box_to_keypoints_map[box].extend(keypoints[i])
            if scores is None:
                box_to_color_map[box] = 'black'
            else:
                display_str = int(100 * scores[i])
                box_to_display_str_map[box].append(display_str)

    rect_points = []
    class_scores = []
    for box, color in six.iteritems(box_to_color_map):
        ymin, xmin, ymax, xmax = box
        rect_points.append(dict(ymin=ymin, xmin=xmin, ymax=ymax, xmax=xmax))
        class_scores.append(box_to_display_str_map[box])
    return rect_points, class_scores

class ChargerConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "charger"

    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + charger

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9
    # LEARNING_RATE = 0.001
    NUM_POINTS = 5

class Detector:

    def __init__(self, path_to_model, path_to_pole_model, path_to_model_bottom=None):
        np.set_printoptions(suppress=True)
        np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
        self.slice_size = (720, 960)
        # self.slice_size = (1280, 1280)
        self.offset = (0, 0)
        self.scale = 1.0
        # self.scale = 0.7
        self.charger_to_slice_ratio = 0.5
        self.detections = []
        self.best_detection = None
        self.mask_reshaped = None
        self.contour = None
        self.frame_shape = None
        self.moving_avg_image = None
        self.init_det = True
        self.bottom = False
        self.gt_kp = None

        # path_to_pole_model = path_to_pole_model
        self.pole_detection_graph = tf.Graph()
        with self.pole_detection_graph.as_default():
            od_graph_def = tf.compat.v1.GraphDef()
            with tf.io.gfile.GFile(path_to_pole_model, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

            self.pole_sess = tf.compat.v1.Session(graph=self.pole_detection_graph)

        config = ChargerConfig()
        config.display()

        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=path_to_model)

        weights_path = model.find_last()
        # weights_path = "/root/share/tf/Keras/16_06_PP_5_wo_mask_bigger_head/charger20200616T1458/mask_rcnn_charger_0021.h5"

        # Load weights
        # print("Loading weights ", weights_path)
        model.load_weights(weights_path, by_name=True)
        self.det_model = model

        if path_to_model_bottom is not None:
            config.NUM_POINTS = 4
            model_bottom = modellib.MaskRCNN(mode="inference", config=config,
                                             model_dir=path_to_model_bottom)

            weights_path_bottom = model_bottom.find_last()

            # Load weights
            print("Loading weights ", weights_path_bottom)
            model_bottom.load_weights(weights_path_bottom, by_name=True)
            self.det_model_bottom = model_bottom

        self.rpn_box_predictor_features = None
        self.class_predictor_weights = None

        self.gt = None

    def init_size(self, shape):
        self.frame_shape = shape
        self.moving_avg_image = np.full(shape[:2], 100, dtype=np.uint8)

    def color_splash(self, image, mask):
        """Apply color splash effect.
        image: RGB image [height, width, 3]
        mask: instance segmentation mask [height, width, instance count]

        Returns result image.
        """
        return image
        # Make a grayscale copy of the image. The grayscale copy still
        # has 3 RGB channels, though.
        gray = np.zeros_like(image)  # skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
        # Copy color pixels from the original color image where mask is set
        if mask.shape[-1] > 0:
            # We're treating all instances as one, so collapse the mask into one layer
            mask = (np.sum(mask, -1, keepdims=True) >= 1)
            splash = np.where(mask, gray, image).astype(np.uint8)
        else:
            splash = gray.astype(np.uint8)
        return splash

    def get_CNN_output(self, image_np):  # TODO: scale keypoints
        # print("scale", self.scale, "\n\n")
        if self.bottom:
            r = self.det_model_bottom.detect([image_np], verbose=0)[0]
        else:
            r = self.det_model.detect([image_np], verbose=0)[0]
        # Color splash
        splash = self. color_splash(image_np, r['masks'])
        if len(r['rois']) == 0:
            return
        kps = r['kp'][0][0]
        uncertainty_raw = r["uncertainty"][0][0]
        # print(uncertainty_raw.shape)
        roi = r['rois'][0]
        # roi[2] = roi[2]+roi[2]-roi[0]
        bw = roi[3] - roi[1]
        bh = roi[2] - roi[0]
        absolute_kp = []
        uncertainty = []
        abs_xmin = int(roi[1] + self.offset[1])
        abs_ymin = int(roi[0] + self.offset[0])
        abs_xmax = np.min(
            (int(roi[3] + self.offset[1]), self.frame_shape[1]))
        abs_ymax = np.min(
            (int(roi[2] + self.offset[0]), self.frame_shape[0]))
        for i in range(int(len(kps) / 2)):
            absolute_kp.append(
                (int(kps[i * 2] * bw + self.offset[1] + roi[1]), int(kps[i * 2 + 1] * bh + self.offset[0] + roi[0])))
            cv2.circle(splash, (int(kps[i * 2] * bw + roi[1]), int(kps[i * 2 + 1] * bh + roi[0])), 5, (0, 0, 255), -1)
            cv2.circle(splash, (
            int(self.gt_kp[i][0] * self.scale - self.offset[1]), int(self.gt_kp[i][1] * self.scale - self.offset[0])),
                       3, (255, 255, 255), -1)
            print("gt_kp", int(self.gt_kp[i][0]), int(self.gt_kp[i][1]))
            L = np.array([[uncertainty_raw[i * 2], 0], [uncertainty_raw[i * 2 + 1], uncertainty_raw[i * 2 + 2]]])
            sigma = np.matmul(L, L.transpose())
            # uncertainty.append((uncertainty_raw[i * 2] * bw, uncertainty_raw[i * 2 + 1] * bh))
            uncertainty.append(sigma)
            cv2.ellipse(splash, (int(kps[i * 2] * bw + roi[1]), int(kps[i * 2 + 1] * bh + roi[0])),
                        (int(np.sqrt(sigma[0, 0] * bw)), int(np.sqrt(sigma[1, 1] * bh))), angle=0,
                        startAngle=0, endAngle=360, color=(0, 255, 255))
        cv2.rectangle(splash, (int(roi[1]), int(roi[0])), (int(roi[3]), int(roi[2])), (0, 255, 255), 2)
        cv2.imshow('Detection', cv2.resize(splash, (960, 960)))
        # print("uncert", kps[int(len(kps) / 2):])

        absolute_kp_scaled = np.multiply(absolute_kp, 1 / self.scale)
        detection = dict(score=r['scores'][0], abs_rect=(abs_xmin, abs_ymin, abs_xmax, abs_ymax),
                         mask=np.sum(r['masks'],
                                     -1, keepdims=False), keypoints=absolute_kp, uncertainty=np.array(uncertainty))
        self.detections.append(detection)

    def detect(self, frame, gt, gt_kp):
        self.gt_kp = gt_kp
        self.frame_shape = frame.shape[:2]
        y_off = int(np.max((0, np.min((self.offset[0], self.frame_shape[0] - self.slice_size[0])))))
        x_off = int(np.max((0, np.min((self.offset[1], self.frame_shape[1] - self.slice_size[1])))))
        self.offset = (y_off, x_off)
        self.gt = gt
        # if self.frame_shape is None:
        #     return
        self.detections = []
        self.best_detection = None
        if self.init_det and not self.bottom:  # comment in 1280x960 resolution
            print("init_det")
            self.init_detection(frame)
        else:
        # self.offset = (1988, 2851)            #uncomment in 1280x960 resolution
        #     print("frame.shape", frame.shape)
        #     cv2.imshow("slice", self.get_slice(frame))
        #     cv2.waitKey(1)
            self.get_CNN_output(self.get_slice(frame))
        if len(self.detections) == 0:
            print("No detections")
            self.init_det = True
        else:
            # self.init_det = False
            self.get_best_detections()
            # if width < self.slice_size[1]/2 and height < self.slice_size[0]/2:
            #     self.scale = 1.0
            # else:
            #     self.scale = min(self.slice_size[1]/width/2, self.slice_size[0]/height/2)
            # self.draw_detection(frame)
        if self.best_detection is not None:
            abs_xmin, abs_ymin, abs_xmax, abs_ymax = self.best_detection['abs_rect']
            width = abs_xmax - abs_xmin
            height = abs_ymax - abs_ymin
            # print("width", width)
            if width > self.slice_size[1] * self.charger_to_slice_ratio:
                self.scale = self.scale * self.slice_size[
                    1] / width * self.charger_to_slice_ratio  # min(self.slice_size[1]/width*self.charger_to_slice_ratio, self.slice_size[0]/height*self.charger_to_slice_ratio)
            # if self.scale < 0.2:
            #     self.bottom = True
            #     self.scale = 0.7
            #     self.scale = 1.0
        return

    def get_slice(self, frame, offset=None):
        # return cv2.resize(frame, self.slice_size)
        if offset is not None:
            return frame[offset[0]:offset[0] + self.slice_size[0],
                   offset[1]:offset[1] + self.slice_size[1]]
        return frame[self.offset[0]:self.offset[0] + self.slice_size[0],
                     self.offset[1]:self.offset[1] + self.slice_size[1]]

    def init_detection(self, frame):
        small_frame = cv2.resize(frame, (self.slice_size[1], self.slice_size[0]))
        # cv2.waitKey(10)
        width, height = self.detect_pole(small_frame)
        if width > self.slice_size[
            1] * self.charger_to_slice_ratio:  # or height > self.slice_size[0]*self.charger_to_slice_ratio:
            self.scale = self.slice_size[
                             1] / width * self.charger_to_slice_ratio  # min(self.slice_size[1]/width*self.charger_to_slice_ratio, self.slice_size[0]/height*self.charger_to_slice_ratio)
            frame = cv2.resize(frame, (0, 0), fx=self.scale, fy=self.scale)
            self.frame_shape = frame.shape[:2]
            self.offset = (int(self.offset[0] * self.scale), int(self.offset[1] * self.scale))
            y_off = int(np.max((0, np.min((self.offset[0], self.frame_shape[0] - self.slice_size[0])))))
            x_off = int(np.max((0, np.min((self.offset[1], self.frame_shape[1] - self.slice_size[1])))))
            self.offset = (y_off, x_off)
        # print(self.offset)
        if width == 0:
            return
        try:
            cv2.imshow("pole", self.get_slice(frame))
            cv2.waitKey(10)
        except cv2.error:
            print("self.get_slice(frame).shape", self.get_slice(frame).shape)
        # print("self.get_slice(frame).shape", self.get_slice(frame).shape)
        self.get_CNN_output(self.get_slice(frame))
        # else:
        #     self.scale = 1.0
        # frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
        # offset = self.offset*scale

        # cols = np.max((2, np.ceil(width / self.slice_size[1])))
        # rows = np.max((2, np.ceil(height / self.slice_size[0])))
        # # print("colsrows %f, %f", cols, rows)
        # y_start = self.offset[0]
        # x_start = self.offset[1]
        # y_step = self.slice_size[0] - int((rows * self.slice_size[0] - (self.offset[0] + height)) / (rows - 1))
        # x_step = self.slice_size[1] - int((cols * self.slice_size[1] - (self.offset[1] + width)) / (cols - 1))
        # for y_off in range(y_start, self.offset[0] + height - y_step, y_step):
        #     for x_off in range(x_start, self.offset[1] + width - x_step, x_step):
        #         self.offset = (y_off, x_off)
        #         self.get_CNN_output(self.get_slice(frame))

    def detect_pole(self, small_frame):
        image_np_expanded = np.expand_dims(small_frame, axis=0)
        image_tensor = self.pole_detection_graph.get_tensor_by_name('image_tensor:0')

        boxes = self.pole_detection_graph.get_tensor_by_name('detection_boxes:0')
        scores = self.pole_detection_graph.get_tensor_by_name('detection_scores:0')
        classes = self.pole_detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = self.pole_detection_graph.get_tensor_by_name('num_detections:0')
        (boxes, scores, classes, num_detections) = self.pole_sess.run([boxes, scores, classes, num_detections],
                                                                      feed_dict={image_tensor: image_np_expanded})

        # Visualization of the results of a detection.
        # rect_points, class_scores = draw_boxes_and_labels(
        #     boxes=np.squeeze(boxes),
        #     classes=np.squeeze(classes).astype(np.int32),
        #     scores=np.squeeze(scores),
        #     min_score_thresh=.10
        # )
        boxes = np.squeeze(boxes)
        scores = np.squeeze(scores)
        boxes_dict = []
        for box, score in zip(boxes, scores):
            # print("box", box)
            ymin, xmin, ymax, xmax = box
            boxes_dict.append(dict(ymin=ymin, xmin=xmin, ymax=ymax, xmax=xmax, score=score))
        # print("boxes", boxes)
        # print("scores", scores)

        detections = []
        for idx in range(len(boxes_dict)):
            abs_xmin = int(boxes_dict[idx]['xmin'] * self.frame_shape[1])
            abs_ymin = int(boxes_dict[idx]['ymin'] * self.frame_shape[0])
            abs_xmax = np.min(
                (int(boxes_dict[idx]['xmax'] * self.frame_shape[1]), self.frame_shape[1]))
            abs_ymax = np.min(
                (int(boxes_dict[idx]['ymax'] * self.frame_shape[0]), self.frame_shape[0]))
            if abs_xmax <= abs_xmin or abs_ymax <= abs_ymin:
                continue
            detection = dict(rel_rect=boxes_dict[idx], score=boxes_dict[idx]['score'],
                             abs_rect=(abs_xmin, abs_ymin, abs_xmax, abs_ymax))
            detections.append(detection)

        if len(detections) > 0:
            best_detection = sorted(detections, key=lambda k: k['score'], reverse=True)[0]
            # print("best_detection", best_detection)
            # print("self.frame_shape", self.frame_shape)
            y_off = int(np.max((0, np.min(
                (best_detection['abs_rect'][1] - self.slice_size[0] / 5, self.frame_shape[0] - self.slice_size[0])))))
            x_off = int(np.max((0, np.min(
                (best_detection['abs_rect'][0] - self.slice_size[1] / 5, self.frame_shape[1] - self.slice_size[1])))))
            self.offset = (y_off, x_off)
            # self.init_det = False
            return (best_detection['abs_rect'][2] - best_detection['abs_rect'][0],
                    best_detection['abs_rect'][3] - best_detection['abs_rect'][1])
        else:
            self.offset = (0, 0)  # (self.frame_shape[0]-1, self.frame_shape[1]-1)
            return 0, 0

    def get_best_detections(self):
        for idx, det in enumerate(self.detections):
            x1, y1, x2, y2 = det['abs_rect']
            ma_score = np.mean(self.moving_avg_image[y1:y2, x1:x2])
            self.detections[idx]['refined_score'] = ma_score + self.detections[idx]['score']
        self.best_detection = sorted(self.detections, key=lambda k: k['refined_score'], reverse=True)[0]
        x1, y1, x2, y2 = self.best_detection['abs_rect']
        # y_off = int(np.min((np.max((0, y1 - self.slice_size[0] / 2)), self.frame_shape[0])))
        # x_off = int(np.min((np.max((0, x1 - self.slice_size[1] / 2)), self.frame_shape[1])))
        # mask = self.best_detection['mask'].astype(np.uint8)
        # print(self.best_detection['mask'].shape, 'lool')
        # mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        # self.mask_reshaped = mask # cv2.resize(mask, dsize=(x2 - x1, y2 - y1))
        y_off = int(np.max((0, np.min((y1 - self.slice_size[0] / 5, self.frame_shape[0] - self.slice_size[0])))))
        x_off = int(np.max((0, np.min((x1 - self.slice_size[1] / 5, self.frame_shape[1] - self.slice_size[1])))))
        self.offset = (y_off, x_off)
        return (x2 - x1, y2 - y1)

    def draw_detection(self, frame):
        x1, y1, x2, y2 = self.best_detection['abs_rect']
        mask = self.best_detection['mask'] * 255
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        self.mask_reshaped = cv2.resize(mask, dsize=(x2 - x1, y2 - y1))

        if 'keypoints' in self.best_detection:
            for idx, pt in enumerate(self.best_detection['keypoints']):
                cv2.circle(frame, (int(pt[0]), int(pt[1])), 3, (0, 255, 0), 1)

        # frame[y1:y2, x1:x2] = self.mask_reshaped
        imgray = cv2.cvtColor(self.mask_reshaped.astype(np.uint8), cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(imgray, 127, 255, 0)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        if len(contours) > 0:
            self.contour = np.add(contours[0], [x1, y1])
            cv2.drawContours(frame, [self.contour], 0, (255, 50, 50), 1)

        # frame[y1:y2, x1:x2] = np.where(self.mask_reshaped>0.5, (255, 255, 255), frame[y1:y2, x1:x2])
        # cv2.rectangle(frame, (x1, y1), (x1+80, y1-30), (0, 255, 0), -1)
        # cv2.putText(frame, "charger", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 2)

        # self.class_predictor_weights = np.reshape(self.class_predictor_weights, (512, -1, 2))
        # heatmap = featureVisualizer.create_fm_weighted(0.7, np.squeeze(self.rpn_box_predictor_features), self.get_slice(frame), self.class_predictor_weights[:, :, 1])
        # cv2.rectangle(heatmap, (x1-self.offset[1], y1-self.offset[0]), (x2-self.offset[1], y2-self.offset[0]), (0, 255, 0), 3)
        # cv2.namedWindow("heatmap", 0)
        # cv2.imshow("heatmap", heatmap)
