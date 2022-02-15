#!/usr/bin/env python3

import os
import numpy as np
# try:
#     from cv2 import cv2
# except ImportError:
#     pass
import cv2
from DetectorHM1 import Detector
from Detector import Detector
from Detector_mmpose import DetectorMmpose
from Detector_mmposeDB import DetectorMmposeDB
from Detector_mmposeDB_HM1 import DetectorMmposeDB_HM1
from PoseEstimator import PoseEstimator
import rospy
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import PoseStamped
from custom_solvepnp_msg.msg import KeypointsWithCovarianceStamped

import xml.etree.ElementTree as ET
import time
from scipy.optimize import minimize


# Uncomment to run on CPU
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

class DetectorNode:
    """Decection can be made using feed from front (blackfly) or back (pointgrey) camera.
       Pipeline for back camera is not maintained for a long time. path_to_model_front can be
       path to directory - last weights are used or direct path to .h5 file. By default, images
       are read from ROS topic, but it is possible to read images from directory by passing path
       to directory containing "images" dir to self.read_from_dir_path and images size to
       self.blackfly_frame_shape. When "annotations" dir is available, gt can be loaded"""
    def __init__(self):
        # Prearing publishers
        self.posePublisher_front = rospy.Publisher('/pose_estimator/charger_pose/detection_front', PoseStamped,
                                                   queue_size=1)
        self.keypointsPublisher = rospy.Publisher('/pose_estimator/keypoints', KeypointsWithCovarianceStamped,
                                                  queue_size=1)
        # Topic names of camera input image
        self.blackfly_topic = '/blackfly/camera/image_color/compressed'
        rospy.init_node('deep_pose_estimator', log_level=rospy.DEBUG)
        # Paths to trained models: front, bask and YOLO pole detector
        path_to_model_front = "/root/share/tf/pl_checkpoints/c_r50_1l_1px/epoch=13-step=8749.ckpt"
        path_to_pole_model = os.path.join('/root/share/tf/YOLO/', '4_12trained_weights_final.h5')
        # path_to_pole_model = os.path.join('/root/share/tf/YOLO/', '13_10trained_weights_final.h5')
        self.scale = 1.0
        self.num_points_front = 4
        # Set True to equalize histogram of input image
        self.equalize_histogram = False
        # Camera intrinsics for local SolvePnP. Not used in default pipeline. See: custom_solvepnp repository
        # self.blackfly_camera_matrix = np.array([[4885.3110509, 0, 2685.5111516],
        #                                         [0, 4894.72687634, 2024.08742622],
        #                                         [0, 0, 1]]).astype(np.float64)
        self.blackfly_camera_matrix = np.array([[4947.48755782, 0, 2620.30371328],
                                                [0, 4961.12502108, 1888.56860792],
                                                [0, 0, 1]]).astype(np.float64)

        # self.blackfly_camera_distortion = (-0.10912282,  0.10204657,  0.00090473, -0.00106435)
        self.blackfly_camera_distortion = (-0.12819854,  0.14240317, -0.00049096, -0.00664523)

        # CALIBRATION AFTER CHANGE 17.02.2021
        # self.blackfly_camera_matrix = np.array([[5030.637829822024, 0, 2817.644564009482],
        #                                         [0, 5036.48155498098, 1829.1542473494844],
        #                                         [0, 0, 1]]).astype(np.float64)
        # self.blackfly_camera_distortion = (-0.09423173844130292, 0.07823597092902158,
        #                                    -0.0005218753461887675, 0.0027740948397291416)
        # self.blackfly_camera_matrix = np.array([[4947.48755782, 0, 2620.30371328],
        #                                         [0, 4961.12502108, 1888.56860792],
        #                                         [0, 0, 1]]).astype(np.float64)

        # # CALIBRATION 08.03.2021
        # self.blackfly_camera_matrix = np.array([[5030.48183786, 0, 2778.33655615],
        #                                         [0, 5040.5801764, 1898.59619234],
        #                                         [0, 0, 1]]).astype(np.float64)
        # self.blackfly_camera_distortion = (-0.10645633, 0.10450576, 0.00254424, -0.00042966)

        # 3D coordinates of points depending of used variant: 5 points-corners, 4-point black rectangles, 9-both
        if self.num_points_front == 4:
            # self.object_points = [[-0.31329, -0.02334, -0.62336], [-0.04812, -0.26695, -0.619169],
            #                         [0.09679, -0.26416, -0.61086], [0.34823, -0.01992, -0.605189]]       #roof
            self.object_points = [[-0.35792, -0.02384, -0.63703], [0.3991, -0.01473, -0.60985],
                                  [2.82224, -0.90896, -0.05048], [-0.10018, -0.74608, -0.05833]]       #corners
            # self.object_points = [[[0.01755, -0.30737, -0.624479],[2.71971, 0.65149, -0.05814],
            #                        [2.3428, -0.66646, -0.03803],[0.24037, -0.67778, -0.05468]]]       #tape
        elif self.num_points_front == 8:
            self.object_points = [[0.01755, -0.30737, -0.624479], [2.71971, 0.65149, -0.05814],
                                  [2.3428, -0.66646, -0.03803], [0.24037, -0.67778, -0.05468],
                                  [-0.35792, -0.02384, -0.63703], [0.3991, -0.01473, -0.60985],
                                  [2.82224, -0.90896, -0.05048], [-0.10018, -0.74608, -0.05833]]       #all
        self.keypoints = None
        self.gt_keypoints = []
        self.blackfly_image_timestamp = None
        self.blackfly_image = None
        self.pointgrey_image_timestamp = None
        self.pointgrey_image = None
        # Uncertainty estimation
        self.uncertainty = None
        self.last_image_timestamp = 0.0

        # To read images from directory insert here path to directory with images and annotations folder inside
        self.read_from_dir_path = None  # '/root/share/tf/dataset/artag/out_close'\
        self.blackfly_frame_shape = self.get_image_shape(self.blackfly_topic)
        print('SHAPE:', self.blackfly_frame_shape)
        # Initialize detector
        # self.detector = Detector(path_to_model_front,
        #                          path_to_pole_model,
        #                          self.num_points_front)  # , path_to_model_bottom=path_to_model_bottom)
        # self.detector = DetectorMmpose("/root/share/tf/mmpose_checkpoints/c_hrnet48_udp_512_hm512/epoch_7.pth", path_to_pole_model)
        # self.detector = DetectorMmpose("/root/share/tf/mmpose_checkpoints/c_hrnet48_udp_512_hm256/epoch_3.pth", path_to_pole_model)
        # self.detector = DetectorMmpose("/root/share/tf/mmpose_checkpoints/c_hrnet48_udp_512_hm128_c2/epoch_8.pth", path_to_pole_model)
        # self.detector = DetectorMmpose("/root/share/tf/mmpose_checkpoints/c_hrnet48_udp_512_hm1024/epoch_10.pth", path_to_pole_model)
        # self.detector = DetectorMmpose("/root/share/tf/mmpose_checkpoints/c_hrnet32_udp_512_hm128/epoch_5.pth", path_to_pole_model)
        # self.detector = DetectorMmpose("/root/share/tf/mmpose_checkpoints/c_hrnet32_udp_512_hm256/epoch_8.pth", path_to_pole_model)
        # self.detector = DetectorMmpose("/root/share/tf/mmpose_checkpoints/c_hrnet32_udp_512_hm512/epoch_8.pth", path_to_pole_model)
        # self.detector = DetectorMmpose("/root/share/tf/mmpose_checkpoints/c_litehrnet30_512_hm256/epoch_7.pth", path_to_pole_model)
        # self.detector = DetectorMmpose("/root/share/tf/mmpose_checkpoints/c_litehrnet30_512_hm128/epoch_10.pth", path_to_pole_model)
        # self.detector = DetectorMmpose("/root/share/tf/mmpose_checkpoints/c_litehrnet30_512_hm512/epoch_8.pth", path_to_pole_model)
        # self.detector = DetectorMmpose("/root/share/tf/mmpose_checkpoints/c_hrnet48_udp_512_hm256_c9/epoch_7.pth", path_to_pole_model)
        self.detector = DetectorMmpose("/root/share/tf/mmpose_checkpoints/c_hrnet48_udp_512_hm128/epoch_7.pth", path_to_pole_model)
        # self.detector = DetectorMmpose("/root/share/tf/mmpose_checkpoints/c_litehrnet30_512_hm512_b4/epoch_9.pth", path_to_pole_model)
        # self.detector = DetectorMmpose("/root/share/tf/mmpose_checkpoints/c_litehrnet30_512_hm512_d0.5/epoch_8.pth", path_to_pole_model)
        
        
        self.detector.init_size(self.blackfly_frame_shape)
        self.pose_estimator = PoseEstimator(self.blackfly_camera_matrix)

    def get_image_shape(self, camera_topic):
        """Returns size of images from given topic"""
        im = rospy.wait_for_message(camera_topic, CompressedImage)
        np_arr = np.fromstring(im.data, np.uint8)
        image_shape = cv2.imdecode(np_arr, -1).shape[:2]
        return list(image_shape)

    def start(self):
        """Detecting in loop"""
        if self.read_from_dir_path is not None:
            images = os.listdir(os.path.join(self.read_from_dir_path, 'annotations'))
            images = iter(images)
        else:
            rospy.Subscriber(self.blackfly_topic, CompressedImage, self.update_blackfly_image, queue_size=1, buff_size=2**24)
        while not rospy.is_shutdown():  # and self.pointgrey_image is not None:
            if self.read_from_dir_path is not None:
                self.read_from_dir(images, self.read_from_dir_path)
            if self.blackfly_image is not None:
                self.detect(self.blackfly_image, self.blackfly_image_timestamp)
            k = cv2.waitKey(10)
            if k == ord('q') or k == 27:
                exit(0)

        self.detector.yolo.close_session()

    def read_from_dir(self, images, base_path, read_gt=False):
        """Reads image from directory. Ready to read gt from annotations"""
        self.gt_keypoints = []
        fname = next(images)
        img_fname = os.path.join(base_path, 'images/', fname[:-4] + '.png')
        self.blackfly_image = cv2.imread(img_fname)
        if read_gt:
            ann_fname = os.path.join(base_path, 'annotations/', fname)
            print(img_fname)
            tree = ET.parse(ann_fname)
            root = tree.getroot()

            size = root.find('size')
            w = int(size.find('width').text)
            h = int(size.find('height').text)
            for object in root.findall('object'):
                bbox = object.find('bndbox')
                xmin = int(float(bbox.find('xmin').text) * w)
                ymin = int(float(bbox.find('ymin').text) * h)
                xmax = int(float(bbox.find('xmax').text) * w)
                ymax = int(float(bbox.find('ymax').text) * h)
                bbox = [xmin, ymin, xmax, ymax]

                kps = object.find('keypoints')
                for i in range(self.num_points_front):
                    kp = kps.find('keypoint' + str(i))
                    self.gt_keypoints.append(
                        (int(float(kp.find('x').text) * w), int((float(kp.find('y').text) * h))))

    def update_blackfly_image(self, image_msg):
        """Read image and timestamp. Remove distortion and equalize histogram"""
        self.blackfly_image_timestamp = image_msg.header.stamp
        np_arr = np.frombuffer(image_msg.data, np.uint8)
        image = cv2.imdecode(np_arr, -1)
        image = cv2.undistort(image, self.blackfly_camera_matrix, self.blackfly_camera_distortion)
        image = cv2.resize(image, (0,0), fx=self.scale, fy=self.scale)
        if self.equalize_histogram:
            img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            img_yuv[:, :, 2] = cv2.equalizeHist(img_yuv[:, :, 2])
            image = cv2.cvtColor(img_yuv, cv2.COLOR_HSV2BGR)
        self.blackfly_image = image
        print("Image update time", time.time()-self.last_image_timestamp)
        self.last_image_timestamp = time.time()



    def detect(self, frame, timestamp):
        """Run detector, draw and publish results"""

        disp = np.copy(frame)
        working_copy = np.copy(frame)
        # print("sssh", working_copy.shape)
        tic = time.perf_counter()
        self.detector.detect(working_copy)
        toc = time.perf_counter()
        print(f"Detection time: {toc - tic:0.4f} seconds")
        if self.detector.best_detection is not None:
            self.keypoints = self.detector.best_detection['keypoints'] 
            self.rel_keypoints = self.detector.best_detection['rel_keypoints'] 
            self.bbox = self.detector.best_detection['bbox']
            self.refined_kpts = self.refine_dets(self.rel_keypoints)
            self.uncertainty = self.detector.best_detection['heatmap_uncertainty']
            self.draw_detection(disp)
            if timestamp is not None:
                self.publish_keyponts(timestamp)
            self.blackfly_image = None


    def refine_dets(self, kpts_pred):

        kpts3d = self.object_points
        K = np.array([[1,0,256], [0,1,256], [0,0,1]]).astype(np.float32)
        x = np.zeros(9)

        def convert_joints3d(joints3d, r, s, t) -> np.ndarray:
            """Apply rotation, scale and translation to joints3d to generate glob_joints_3d
            Parameters
            ----------
            kpts_format: str
                Format of used kpt (coco or spin)
            """

            R, _ = cv2.Rodrigues(r)
            joints3d = joints3d @ R
            joints3d_tr = (
                joints3d * s + t
            )
            return joints3d_tr
        def f(x):
            r, s, t = x[:3], x[3:6], x[6:]
            j3d_tr = convert_joints3d(kpts3d, r, s, t)
            j2d, _ = cv2.projectPoints(j3d_tr, np.zeros(3), np.zeros(3), K, None)
            proj = j2d[:, 0, :]

            # proj = K@convert_joints3d(kpts3d, r, s, t)
            loss = np.sum(np.square(proj-kpts_pred))
            return loss

        opt = minimize(f, x, method="SLSQP")
        x = opt.x

        r, s, t = x[:3], x[3:6], x[6:]
        j3d_tr = convert_joints3d(kpts3d, r, s, t)
        kpts_refined, _ = cv2.projectPoints(j3d_tr, np.zeros(3), np.zeros(3), K, None)
        print(kpts_pred, kpts_refined)
        kpts_refined = kpts_refined[:, 0, :]+self.bbox[0:2]
        return kpts_refined





    def draw_detection(self, frame):
        """Draw keypoints and uncertainty ellipses for 1,2 and 3 sigma"""
        colors = [(255,0,0), (0,255,0), (0,0,255), (255,255,0)]
        # for i, pt in enumerate(self.keypoints):
        for i, (kp, rkp) in enumerate(zip(self.keypoints, self.refined_kpts)):
            print(rkp.shape, rkp)
            cv2.circle(frame, (int(kp[0]), int(kp[1])), 15-2*i, colors[i], -1)
            cv2.circle(frame, (int(rkp[0]), int(rkp[1])), 5, (255,255,255), -1)
        xmin, ymin, xmax, ymax =  self.detector.best_detection['bbox']
        crop = frame[ymin:ymax, xmin:xmax]
        cv2.imwrite('/root/catkin_ws/src/charger_kpts_train/src/test/out.jpg', cv2.resize(crop, (720,720)))
        
    def publish_keyponts(self, stamp):
        out_msg = KeypointsWithCovarianceStamped()
        out_msg.header.stamp = stamp
        out_msg.num_points = self.num_points_front
        out_msg.object_points = np.reshape(self.object_points, (-1))
        out_msg.covariance = np.reshape(self.uncertainty, (-1))
        out_msg.keypoints = []

        for idx, kp in enumerate(self.keypoints):
            out_msg.keypoints.append(kp[0]/self.scale)
            out_msg.keypoints.append(kp[1]/self.scale)

        self.keypointsPublisher.publish(out_msg)


if __name__ == '__main__':
    det_node = DetectorNode()
    det_node.start()
