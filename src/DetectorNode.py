#!/usr/bin/env python3

import os
import numpy as np
# try:
#     from cv2 import cv2
# except ImportError:
#     pass
import cv2
import math
from Detector import Detector
from PoseEstimator import PoseEstimator
import rospy
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import PoseStamped, Quaternion
from scipy.spatial.transform import Rotation
from custom_solvepnp_msg.msg import KeypointsWithCovarianceStamped

import xml.etree.ElementTree as ET
import time


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
        self.posePublisher_back = rospy.Publisher('/pose_estimator/charger_pose/detection_back', PoseStamped,
                                                  queue_size=1)
        self.keypointsPublisher = rospy.Publisher('/pose_estimator/keypoints', KeypointsWithCovarianceStamped,
                                                  queue_size=1)
        # Topic names of camera input image
        self.blackfly_topic = '/blackfly/camera/image_color/compressed'
        self.pointgrey_topic = '/pointgrey/camera/image_color/compressed'
        rospy.init_node('deep_pose_estimator', log_level=rospy.DEBUG)
        # Paths to trained models: front, bask and YOLO pole detector
        # path_to_model_bottom = "/root/share/tf/Keras/09_05_bottom_PP"
        path_to_model_bottom = "/root/share/tf/Keras/18_06_PP_4_wo_mask_bigger_head"
        path_to_model_front = "/root/share/tf/Keras/15_10_heatmap_final"
        # path_to_model_front = "/root/share/tf/Keras/28_11_roof"
        # path_to_model_front = "/root/share/tf/Keras/28_11_roof/charger20201202T1055/mask_rcnn_charger_0004.h5"
        # path_to_model_front = "/root/share/tf/Keras/2_11_AR"
	# path_to_pole_model = os.path.join('/root/share/tf/YOLO/', '18_09trained_weights_final.h5')
    #     path_to_pole_model = os.path.join('/root/share/tf/YOLO/', '13_10trained_weights_final.h5')
        path_to_pole_model = os.path.join('/root/share/tf/YOLO/', '24_11trained_weights_final.h5')
        # path_to_pole_model = os.path.join('/root/share/tf/YOLO/', '24_11ep001-loss5.416-val_loss5.967.h5')

        self.num_points_front = 4
        # Set True to equalize histogram of input image
        self.equalize_histogram = False
        # Camera intrinsics for local SolvePnP. Not used in default pipeline. See: custom_solvepnp repository
        # self.blackfly_camera_matrix = np.array([[4885.3110509, 0, 2685.5111516],
        #                                         [0, 4894.72687634, 2024.08742622],
        #                                         [0, 0, 1]]).astype(np.float64)
        self.blackfly_camera_matrix = np.array([[4905.87212594, 0, 2826.08722841],
                                                [0, 4902.16329531, 1895.3021881],
                                                [0, 0, 1]]).astype(np.float64)
        # self.blackfly_camera_distortion = (-0.10912282,  0.10204657,  0.00090473, -0.00106435)
        self.blackfly_camera_distortion = (-0.1162506, 0.10820695, 0.00122427, -0.0002685)
        self.pointgrey_camera_matrix = np.array([[671.18360957, 0, 662.70347005],
                                                 [0, 667.94555172, 513.26149201],
                                                 [0, 0, 1]]).astype(np.float64)
        self.pointgrey_camera_distortion = (-0.04002205, 0.04100822, 0.00137423, 0.00464031, 0.0)
        # 3D coordinates of points depending of used variant: 5 points-corners, 4-point black rectangles, 9-both
        if self.num_points_front == 4:
            # self.object_points = [[-0.31329, 0.02334, -0.62336], [-0.04812, 0.26695, -0.619169],
                                #   [0.09679, 0.26416, -0.61086], [0.34823, 0.01992, -0.605189]]       #roof
            self.object_points = [[-0.35792, 0.02384, -0.63703], [0.3991, 0.01473, -0.60985],
                                  [2.82224, 0.90896, -0.05048], [-0.10018, 0.74608, -0.05833]]       #corners
            # self.object_points = [[0.01755, 0.30737, -0.624479], [2.71971, -0.65149, -0.05814],
                                  # [2.3428, 0.66646, -0.03803], [0.24037, 0.67778, -0.05468]]       #tape
        elif self.num_points_front == 8:
            self.object_points = [[0.01755, 0.30737, -0.624479], [2.71971, -0.65149, -0.05814],
                                  [2.3428, 0.66646, -0.03803], [0.24037, 0.67778, -0.05468],
                                  [-0.35792, 0.02384, -0.63703], [0.3991, 0.01473, -0.60985],
                                  [2.82224, 0.90896, -0.05048], [-0.10018, 0.74608, -0.05833]]       #tape
        # if self.num_points_front == 5:
        #     self.object_points = [[-0.3838, -0.0252, -0.6103], [0.3739, -0.0185, -0.6131], [2.7607, 0.7064, -0.1480],
        #                           [2.8160, -0.9127, -0.1428], [-0.1048, -0.7433, -0.0434]]
        # elif self.num_points_front == 4:
        #     self.object_points = [[0.0145, -0.1796, -0.6472], [2.7137, 0.7782, -0.0808], [2.3398, -0.5353, -0.0608],
        #                           [0.2374, -0.5498, -0.0778]]
        # elif self.num_points_front == 9:
        #     self.object_points = [[0.0145, -0.1796, -0.6472], [2.7137, 0.7782, -0.0808], [2.3398, -0.5353, -0.0608],
        #                           [0.2374, -0.5498, -0.0778], [-0.3838, -0.0252, -0.6103], [0.3739, -0.0185, -0.6131],
        #                           [2.7607, 0.7064, -0.1480], [2.8160, -0.9127, -0.1428], [-0.1048, -0.7433, -0.0434]]
        self.keypoints = None
        self.gt_keypoints = []
        self.blackfly_image_timestamp = None
        self.blackfly_image = None
        self.pointgrey_image_timestamp = None
        self.pointgrey_image = None
        # Uncertainty estimation
        self.uncertainty = None

        # To read images from directory insert here path to directory with images and annotations folder inside
        self.read_from_dir_path = None  # '/root/share/tf/dataset/artag/out_close'
        # self.blackfly_frame_shape = (512, 512)
        self.blackfly_frame_shape = self.get_image_shape(self.blackfly_topic)
        # Waiting for pointgrey frame. Comment when there are no data from pointgrey camera.
        # self.pointgrey_frame_shape = self.get_image_shape(self.pointgrey_topic)
        # Initialize detector
        self.detector = Detector(path_to_model_front,
                                 path_to_pole_model,
                                 self.num_points_front)  # , path_to_model_bottom=path_to_model_bottom)
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
            rospy.Subscriber(self.pointgrey_topic, CompressedImage, self.update_pointgrey_image, queue_size=1)
        while not rospy.is_shutdown():  # and self.pointgrey_image is not None:
            if self.read_from_dir_path is not None:
                self.read_from_dir(images, self.read_from_dir_path)
            if self.blackfly_image is not None:
                starttime = time.time()
                if self.detector.bottom:
                    self.detect(self.pointgrey_image, self.pointgrey_image_timestamp)
                else:
                    self.detect(self.blackfly_image, self.blackfly_image_timestamp)
                print("processing time: ", time.time()-starttime, " seconds")
                print("Current time-image time: ", time.time()-self.blackfly_image_timestamp.to_sec(), " seconds")
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
        np_arr = np.fromstring(image_msg.data, np.uint8)
        image = cv2.imdecode(np_arr, -1)
        image = cv2.undistort(image, self.blackfly_camera_matrix, self.blackfly_camera_distortion)
        if self.equalize_histogram:
            img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            img_yuv[:, :, 2] = cv2.equalizeHist(img_yuv[:, :, 2])
            image = cv2.cvtColor(img_yuv, cv2.COLOR_HSV2BGR)
        self.blackfly_image = image

    def update_pointgrey_image(self, image_msg):
        """Read image and timestamp. Remove distortion"""
        self.pointgrey_image_timestamp = image_msg.header.stamp
        np_arr = np.fromstring(image_msg.data, np.uint8)
        image = cv2.imdecode(np_arr, -1)
        image = cv2.undistort(image, self.pointgrey_camera_matrix, self.pointgrey_camera_distortion)
        self.pointgrey_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)  # cv2.resize(image, None, fx=0.7, fy=0.7)

    def detect(self, frame, timestamp):
        """Run detector, draw and publish results"""
        disp = np.copy(frame)
        working_copy = np.copy(frame)
        self.detector.detect(working_copy)
        if self.detector.best_detection is not None:
            self.keypoints = self.detector.best_detection['keypoints']
            self.uncertainty = self.detector.best_detection['uncertainty']
            self.draw_detection(disp)
            if timestamp is not None:
                self.publish_keyponts(timestamp)
                self.publish_pose(timestamp)
            self.blackfly_image = None

    def draw_detection(self, frame):
        """Draw keypoints and uncertainty ellipses for 1,2 and 3 sigma"""
        for i, pt in enumerate(self.keypoints):
            cv2.circle(frame, (int(pt[0]), int(pt[1])), 10, (255, 255, 0), -1)
            cv2.ellipse(frame, (int(pt[0]), int(pt[1])), (
                int(np.ceil(np.sqrt(self.uncertainty[i, i]))), int(np.ceil(np.sqrt(self.uncertainty[i + 1, i + 1])))),
                        angle=0, startAngle=0, endAngle=360, color=(0, 0, 255), thickness=1)
            cv2.ellipse(frame, (int(pt[0]), int(pt[1])), (int(np.ceil(2 * np.sqrt(self.uncertainty[i, i]))),
                                                          int(np.ceil(2 * np.sqrt(self.uncertainty[i + 1, i + 1])))),
                        angle=0, startAngle=0, endAngle=360, color=(0, 255, 255), thickness=1)
            cv2.ellipse(frame, (int(pt[0]), int(pt[1])), (int(np.ceil(3 * np.sqrt(self.uncertainty[i, i]))),
                                                          int(np.ceil(3 * np.sqrt(self.uncertainty[i + 1, i + 1])))),
                        angle=0, startAngle=0, endAngle=360, color=(0, 255, 0), thickness=1)
        cv2.imshow("detection", cv2.resize(frame, self.detector.slice_size))
        # cv2.imshow("detection", disp[2000:, 2000:])

    def publish_pose(self, stamp):
        if self.detector.bottom:
            camera_matrix = self.pointgrey_camera_matrix
        else:
            camera_matrix = self.blackfly_camera_matrix
        tvec, rvec = self.pose_estimator.calc_PnP_pose(self.keypoints, self.object_points, camera_matrix)
        rot = Rotation.from_rotvec(np.squeeze(rvec))
        out_msg = PoseStamped()
        out_msg.header.stamp = stamp
        out_msg.header.frame_id = "camera"
        out_msg.pose.position.x = tvec[0]
        out_msg.pose.position.y = tvec[1]
        out_msg.pose.position.z = tvec[2]
        np_quat = rot.as_quat()
        ros_quat = Quaternion(np_quat[0], np_quat[1], np_quat[2], np_quat[3])

        out_msg.pose.orientation = ros_quat
        if self.detector.bottom:
            self.posePublisher_back.publish(out_msg)
        else:
            self.posePublisher_front.publish(out_msg)

    def publish_keyponts(self, stamp):
        out_msg = KeypointsWithCovarianceStamped()
        out_msg.header.stamp = stamp
        out_msg.num_points = self.num_points_front
        out_msg.object_points = np.reshape(self.object_points, (-1))
        print(self.uncertainty)
        # # Fill uncertainty using 2x2 patches on the diagonal
        # raw_uncertainty = self.uncertainty
        # self.uncertainty = np.zeros((self.num_points_front*2, self.num_points_front*2))
        # for i, unc in enumerate(raw_uncertainty):
        #     self.uncertainty[i*2:i*2+2, i*2:i*2+2] = unc
        out_msg.covariance = np.reshape(self.uncertainty, (-1))
        out_msg.keypoints = []

        for idx, kp in enumerate(self.keypoints):
            out_msg.keypoints.append(kp[0])
            out_msg.keypoints.append(kp[1])

        self.keypointsPublisher.publish(out_msg)


if __name__ == '__main__':
    det_node = DetectorNode()
    det_node.start()
