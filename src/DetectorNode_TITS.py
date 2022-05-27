#!/usr/bin/env python3

import os
import pandas as pd
import numpy as np
# try:
#     from cv2 import cv2
# except ImportError:
#     pass
import cv2
from DetectorHM1 import Detector
from Detector import Detector
from Detector_mmpose import DetectorMmpose
from Detector_mmpose_unc import DetectorMmposeUnc
from Detector_AMCS import DetectorAMCS
from PoseEstimator import PoseEstimator
import rospy
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import PoseStamped
from custom_solvepnp_msg.msg import KeypointsWithCovarianceStamped

import xml.etree.ElementTree as ET
import time
from scipy.optimize import minimize, least_squares, NonlinearConstraint
from scipy.spatial.transform import Rotation
from datetime import datetime

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
        self.scale = [1.0, 1.0]
        self.num_points_front = 4
        # Set True to equalize histogram of input image
        self.equalize_histogram = False
        # Camera intrinsics for local SolvePnP. Not used in default pipeline. See: custom_solvepnp repository
        # self.blackfly_camera_matrix = np.array([[4885.3110509, 0, 2685.5111516],
        #                                         [0, 4894.72687634, 2024.08742622],
        #                                         [0, 0, 1]]).astype(np.float64)
        # self.blackfly_camera_matrix = np.array([[4947.48755782, 0, 2620.30371328],
        #                                         [0, 4961.12502108, 1888.56860792],
        #                                         [0, 0, 1]]).astype(np.float64)

        # self.blackfly_camera_distortion = (-0.10912282,  0.10204657,  0.00090473, -0.00106435)
        self.blackfly_camera_distortion = (-0.12819854,  0.14240317, -0.00049096, -0.00664523)

        # CALIBRATION AFTER CHANGE 17.02.2021
        self.blackfly_camera_matrix = np.array([[5030.637829822024, 0, 2817.644564009482],
                                                [0, 5036.48155498098, 1829.1542473494844],
                                                [0, 0, 1]]).astype(np.float64)
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
        self.read_from_dir_path = "/root/share/tf/dataset/final_localization/corners_1.0/val/"  # '/root/share/tf/dataset/artag/out_close'\
        self.blackfly_frame_shape = [3648, 5472] #self.get_image_shape(self.blackfly_topic)
        print('SHAPE:', self.blackfly_frame_shape)
        # Initialize detector
        # self.detector = Detector(path_to_model_front,
        #                          path_to_pole_model,
        #                          self.num_points_front)  # , path_to_model_bottom=path_to_model_bottom)
        # self.detector = DetectorMmpose("/root/share/tf/mmpose_checkpoints/c_litehrnet30_512_hm256/epoch_7.pth", path_to_pole_model)
        # self.detector = DetectorMmpose("/root/share/tf/mmpose_checkpoints/c_litehrnet30_512_hm128/epoch_10.pth", path_to_pole_model)
        # self.detector = DetectorMmpose("/root/share/tf/mmpose_checkpoints/c_litehrnet30_512_hm128_repr_v2/epoch_10.pth", path_to_pole_model)
        # self.detector = DetectorMmpose("/root/share/tf/mmpose_checkpoints/c_litehrnet30_512_hm128_repr_v3/epoch_9.pth", path_to_pole_model)
        # self.detector = DetectorMmpose("/root/share/tf/mmpose_checkpoints/c_litehrnet_udp_512_hm128_b4/epoch_10.pth", path_to_pole_model)
        # self.detector = DetectorMmpose("/root/share/tf/mmpose_checkpoints/c_litehrnet30_512_hm128_repr_v4/epoch_7.pth", path_to_pole_model)
        # self.detector = DetectorMmpose("/root/share/tf/mmpose_checkpoints/c_litehrnet30_512_hm128_repr_v4/epoch_10.pth", path_to_pole_model)
        # self.detector = DetectorMmpose("/root/share/tf/mmpose_checkpoints/c_litehrnet30_512_hm128_repr_v6/epoch_10.pth", path_to_pole_model)
        # self.detector = DetectorMmpose("/root/share/tf/mmpose_checkpoints/c_litehrnet30_512_hm128_repr_v6/epoch_7.pth", path_to_pole_model)
        # self.detector = DetectorMmpose("/root/share/tf/mmpose_checkpoints/c_litehrnet30_512_hm128_repr_v6/epoch_15.pth", path_to_pole_model)
        # self.detector = DetectorMmpose("/root/share/tf/mmpose_checkpoints/c_litehrnet30_512_hm512/epoch_8.pth", path_to_pole_model)
        # self.detector = DetectorMmpose("/root/share/tf/mmpose_checkpoints/c_litehrnet30_512_hm512_b4/epoch_9.pth", path_to_pole_model)
        # self.detector = DetectorMmpose("/root/share/tf/mmpose_checkpoints/c_litehrnet30_512_hm512_d0.5/epoch_8.pth", path_to_pole_model)
        # self.detector = DetectorMmpose("/root/share/tf/mmpose_checkpoints/c_litehrnet30_512_hm256_repr_v1/epoch_20.pth", path_to_pole_model)

        # self.detector = DetectorMmpose("/root/share/tf/mmpose_checkpoints/c_hrnet48_udp_512_hm512/epoch_7.pth", path_to_pole_model)
        # self.detector = DetectorMmpose("/root/share/tf/mmpose_checkpoints/c_hrnet48_udp_512_hm256/epoch_3.pth", path_to_pole_model)
        # self.detector = DetectorMmpose("/root/share/tf/mmpose_checkpoints/c_hrnet48_udp_512_hm128_c2/epoch_8.pth", path_to_pole_model)
        # self.detector = DetectorMmpose("/root/share/tf/mmpose_checkpoints/c_hrnet48_udp_512_hm1024/epoch_10.pth", path_to_pole_model)
        # self.detector = DetectorMmpose("/root/share/tf/mmpose_checkpoints/c_hrnet48_udp_512_hm128_repr_v1/epoch_6.pth", path_to_pole_model)
        # self.detector = DetectorMmpose("/root/share/tf/mmpose_checkpoints/c_hrnet48_udp_512_hm128_repr_v1/epoch_10.pth", path_to_pole_model)
        # self.detector = DetectorMmpose("/root/share/tf/mmpose_checkpoints/c_hrnet48_udp_512_hm256_c9/epoch_7.pth", path_to_pole_model)
        # self.detector = DetectorMmpose("/root/share/tf/mmpose_checkpoints/c_hrnet48_udp_512_hm128/epoch_7.pth", path_to_pole_model)
        # self.detector = DetectorMmpose("/root/share/tf/mmpose_checkpoints/c_hrnet48_udp_512_hm128_c5/epoch_9.pth", path_to_pole_model)
        # self.detector = DetectorMmpose("/root/share/tf/mmpose_checkpoints/c_hrnet48_udp_512_hm128_c5_d25/epoch_8.pth", path_to_pole_model)
        # self.detector = DetectorMmpose("/root/share/tf/mmpose_checkpoints/c_hrnet48_udp_512_hm128_repr_v2/epoch_9.pth", path_to_pole_model)
        
        # self.detector = DetectorMmpose("/root/share/tf/mmpose_checkpoints/c_hrnet32_udp_512_hm256_repr_v2/epoch_18.pth", path_to_pole_model)
        # self.detector = DetectorMmpose("/root/share/tf/mmpose_checkpoints/c_hrnet32_udp_512_hm256/epoch_8.pth", path_to_pole_model)
        # self.detector = DetectorMmpose("/root/share/tf/mmpose_checkpoints/c_hrnet32_udp_512_hm128/epoch_8.pth", path_to_pole_model)
        # self.detector = DetectorMmpose("/root/share/tf/mmpose_checkpoints/c_hrnet32_udp_512_hm256_repr_v1/epoch_20.pth", path_to_pole_model)
        # 512
        # self.detector = DetectorMmpose("/root/share/tf/mmpose_checkpoints/c_hrnet32_udp_512_hm512_e20/epoch_18.pth", path_to_pole_model)
        # self.detector = DetectorMmpose("/root/share/tf/mmpose_checkpoints/c_hrnet32_udp_512_hm512/epoch_8.pth", path_to_pole_model)
        # self.detector = DetectorMmpose("/root/share/tf/mmpose_checkpoints/c_hrnet32_udp_512_hm512_repr_v2/epoch_15.pth", path_to_pole_model)
        # self.detector = DetectorMmpose("/root/share/tf/mmpose_checkpoints/c_hrnet32_udp_512_hm128_repr_v1/epoch_10.pth", path_to_pole_model)
        # self.detector = DetectorMmpose("/root/share/tf/mmpose_checkpoints/c_hrnet32_udp_512_hm128_repr_v2/epoch_15.pth", path_to_pole_model)
        # self.detector = DetectorMmpose("/root/share/tf/mmpose_checkpoints/c_hrnet32_udp_512_hm128/epoch_9.pth", path_to_pole_model)
        # self.detector = DetectorMmpose("/root/share/tf/mmpose_checkpoints/c_hrnet32_udp_512_hm128_repr_v3/epoch_20.pth", path_to_pole_model)
        # self.detector = DetectorMmpose("/root/share/tf/mmpose_checkpoints/c_hrnet32_udp_512_hm128_repr_v3/epoch_12.pth", path_to_pole_model)
        # 256
        # self.detector = DetectorMmpose("/root/share/tf/mmpose_checkpoints/c_hrnet32_udp_512_hm256_e20/epoch_20.pth", path_to_pole_model)
        # self.detector = DetectorMmpose("/root/share/tf/mmpose_checkpoints/c_hrnet32_udp_512_hm256_repr_v2/epoch_18.pth", path_to_pole_model)
        # self.detector = DetectorMmpose("/root/share/tf/mmpose_checkpoints/c_hrnet32_udp_512_hm512_repr_v1/epoch_17.pth", path_to_pole_model)
        # self.detector = DetectorMmpose("/root/share/tf/mmpose_checkpoints/c_hrnet32_udp_512_hm512_repr_v1/epoch_10.pth", path_to_pole_model)
        # self.detector = DetectorMmpose("/root/share/tf/mmpose_checkpoints/c_hrnet32_udp_512_hm512_repr_v2/epoch_19.pth", path_to_pole_model)
        # 128
        self.detector = DetectorMmpose("/root/share/tf/mmpose_checkpoints/c_hrnet32_udp_512_hm128/epoch_15.pth", path_to_pole_model)
        # self.detector = DetectorMmpose("/root/share/tf/mmpose_checkpoints/c_hrnet32_udp_512_hm128_repr_v4/epoch_13.pth", path_to_pole_model)
        # self.detector = DetectorMmpose("/root/share/tf/mmpose_checkpoints/c_hrnet32_udp_512_hm128_c2/epoch_14.pth", path_to_pole_model)
        # self.detector = DetectorMmposeUnc("/root/share/tf/mmpose_checkpoints/c_hrnet32_udp_512_hm128_unc/epoch_7.pth", path_to_pole_model)
        
        
        # self.detector = DetectorAMCS("/root/share/tf/AMCS_checkpoints/model_2_1_121.pth", path_to_pole_model)
        self.inside=0
        self.outside=0
        
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
            images = os.listdir(os.path.join(self.read_from_dir_path, "images"))
            images = iter(images)
        else:
            rospy.Subscriber(self.blackfly_topic, CompressedImage, self.update_blackfly_image, queue_size=1, buff_size=2**24)
        while not rospy.is_shutdown():  # and self.pointgrey_image is not None:
            if self.read_from_dir_path is not None:
                self.read_from_dir(images, self.read_from_dir_path, True)
            if self.blackfly_image is not None:
                print("detect")
                self.detect(self.blackfly_image, self.blackfly_image_timestamp)
            k = cv2.waitKey(10)
            if k == ord('q') or k == 27:
                exit(0)

        self.detector.yolo.close_session()

    def read_from_dir(self, images, base_path, read_gt=False):
        """Reads image from directory. Ready to read gt from annotations"""
        self.gt_keypoints = []
        fname = next(images)
        img_fname = os.path.join(base_path, "images", fname)
        self.blackfly_image = cv2.imread(img_fname)
        if read_gt:
            ann_fname = os.path.join(base_path, 'annotations/', fname[:-4]+".txt")
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
        # image = cv2.resize(image, (0,0), fx=self.scale, fy=self.scale)
        if self.equalize_histogram:
            img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            img_yuv[:, :, 2] = cv2.equalizeHist(img_yuv[:, :, 2])
            image = cv2.cvtColor(img_yuv, cv2.COLOR_HSV2BGR)
        self.blackfly_image = image
        # print("Image update time", time.time()-self.last_image_timestamp)
        self.last_image_timestamp = time.time()



    def detect(self, frame, timestamp):
        """Run detector, draw and publish results"""

        tic = time.perf_counter()
        disp = np.copy(frame)
        working_copy = np.copy(frame)
        # print("sssh", working_copy.shape)
        self.detector.detect(working_copy)
        if self.detector.best_detection is not None:
            self.keypoints = self.detector.best_detection['keypoints'] 
            self.rel_keypoints = self.detector.best_detection['rel_keypoints'] 
            self.bbox = self.detector.best_detection['bbox']
            self.scale = self.detector.best_detection['scale']
            # tic_ref = time.perf_counter()
            # self.refined_kpts, cost, tr_vec = self.refine_dets_v2(self.keypoints)
            # toc_ref = time.perf_counter()
            self.uncertainty = self.detector.best_detection['uncertainty']
            self.draw_detection(disp)
            if timestamp is not None:
                self.publish_keyponts(timestamp)
            self.blackfly_image = None
        toc = time.perf_counter()
        print(f"Detection time: {toc - tic:0.4f}") #, ref_time: {toc_ref - tic_ref:0.4f}")


    def refine_dets(self, kpts_pred):

        kpts3d = self.object_points
        x = np.array([0.4,0,0,0,0,30]).astype(np.float32)
        # K = np.array([[4950,0,2620], [0,4960,1888], [0,0,1]]).astype(np.float32)
        K = self.blackfly_camera_matrix

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
        def fun(x):
            r, t = x[:3], x[3:]
            j3d_tr = convert_joints3d(kpts3d, r, np.ones(3), t)
            j2d, _ = cv2.projectPoints(j3d_tr, np.zeros(3), np.zeros(3), K, None)
            proj = j2d[:, 0, :]
            loss = np.sum(np.square(np.linalg.norm(proj-kpts_pred, axis=1)))
            return loss

        def fun_ls(x):
            r, t = x[:3], x[3:]
            j3d_tr = convert_joints3d(kpts3d, r, np.ones(3), t)
            j2d, _ = cv2.projectPoints(j3d_tr, np.zeros(3), np.zeros(3), K, None)
            proj = j2d[:, 0, :].reshape((-1))

            loss = proj-kpts_pred.reshape((-1))
            return loss

        bounds = [[-np.pi, np.pi],
                [-np.pi, np.pi],
                [-np.pi, np.pi],
                [-50, 50],
                [-20,20],
                [0,50]
                ]
        bounds = ((-np.pi/4,-np.pi/4,-np.pi/4,-50,-20, 0),
                  (np.pi/4, np.pi/4, np.pi/4, 50, 20, 50))
        def const_x(r):
            r = r[:3]
            x, y, z = Rotation.from_rotvec(r).as_euler('xyz', degrees=True)
            return x
        # constraint = NonlinearConstraint(const_x, 0, 90)
        opt = least_squares(fun_ls, x, bounds=bounds)#, method='lm')
        # opt = minimize(fun, x, method='COBYLA', bounds=bounds, constraints=[NonlinearConstraint(const_x, 0, 90)])
        x = opt.x

        r, t = x[:3], x[3:]
        # print("rot", Rotation.from_rotvec(r).as_euler('xyz', degrees=True), "rotvec", r, "tr", t)
        j3d_tr = convert_joints3d(kpts3d, r, np.ones(3), t)
        kpts_refined, _ = cv2.projectPoints(j3d_tr, np.zeros(3), np.zeros(3), K, None)
        kpts_refined = kpts_refined[:, 0, :]
        return kpts_refined, opt.cost, t

    def refine_dets_v2(self, kpts_pred):

        kpts3d = self.object_points
        x = np.array([0.4,0,0,0,0,30]).astype(np.float32)
        K = self.blackfly_camera_matrix

        def convert_joints3d(joints3d, r, s, t) -> np.ndarray:
            R, _ = cv2.Rodrigues(r)
            joints3d = joints3d @ R
            joints3d_tr = (
                joints3d * s + t
            )
            return joints3d_tr

        def fun_ls(x):
            r, t = x[:3], x[3:]
            j3d_tr = convert_joints3d(kpts3d, r, np.ones(3), t)
            j2d, _ = cv2.projectPoints(j3d_tr, np.zeros(3), np.zeros(3), K, None)
            proj = j2d[:, 0, :].reshape((-1))

            loss = proj-kpts_pred.reshape((-1))
            return loss

        bounds = ((-np.pi/4,-np.pi/4,-np.pi/4,-50,-20, 0),
                  (np.pi/4, np.pi/4, np.pi/4, 50, 20, 50))
        opt = least_squares(fun_ls, x, bounds=bounds)
        x = opt.x

        r, t = x[:3], x[3:]
        j3d_tr = convert_joints3d(kpts3d, r, np.ones(3), t)
        kpts_refined, _ = cv2.projectPoints(j3d_tr, np.zeros(3), np.zeros(3), K, None)
        kpts_refined = kpts_refined[:, 0, :]
        dist = np.linalg.norm(kpts_pred-kpts_refined, axis=1)
        # print("dist", dist.shape)
        if np.max(dist)>2*np.mean(dist[np.arange(len(dist))!=np.argmax(dist)]):
            print("refok")
            return kpts_refined, opt.cost, t
        else:
            return kpts_pred, opt.cost, t

    def draw_detection(self, frame):
        """Draw keypoints and uncertainty ellipses for 1,2 and 3 sigma"""
        colors = [(255,0,0), (0,255,0), (0,0,255), (255,255,0)]
        colorsd = [(100,0,0), (0,100,0), (0,0,100), (100,100,0)]
        for i, kp in enumerate(self.keypoints):
            cv2.circle(frame, (int(kp[0]), int(kp[1])), 25, colors[i], 5)
        xmin, ymin, xmax, ymax =  self.detector.best_detection['bbox']
        # crop = frame[ymin:ymax, xmin:xmax]
        cv2.imwrite('/root/catkin_ws/src/charger_kpts_train/src/test/out.jpg', cv2.resize(frame, (720,720)))
        # cv2.imwrite('/root/catkin_ws/src/charger_kpts_train/src/test/unc/'+str(self.blackfly_image_timestamp.secs)+'.jpg', cv2.resize(crop, (720,720)))
        # time.sleep(2)

    def publish_keyponts(self, stamp):
        out_msg = KeypointsWithCovarianceStamped()
        out_msg.header.stamp = stamp
        out_msg.num_points = self.num_points_front
        out_msg.object_points = np.reshape(self.object_points, (-1))
        out_msg.covariance = np.reshape(self.uncertainty, (-1))
        out_msg.scale = self.scale
        out_msg.keypoints = []

        # for idx, kp in enumerate(self.refined_kpts):
        for idx, kp in enumerate(self.keypoints):
            out_msg.keypoints.append(kp[0])
            out_msg.keypoints.append(kp[1])

        self.keypointsPublisher.publish(out_msg)


if __name__ == '__main__':
    det_node = DetectorNode()
    det_node.start()
