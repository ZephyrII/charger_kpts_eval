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
from sensor_msgs.msg import CompressedImage, NavSatFix
from ublox_msgs.msg import NavRELPOSNED
from geometry_msgs.msg import PoseStamped, Quaternion
from scipy.spatial.transform import Rotation
from custom_solvepnp_msg.msg import KeypointsWithCovarianceStamped

import xml.etree.ElementTree as ET
import time


# import math

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

def rotate(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    ox, oy, oz = origin
    px, py, pz = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (pz - oz)
    qz = oz + math.sin(angle) * (px - ox) + math.cos(angle) * (pz - oz)
    qy = py
    return [qx, qy, qz]

class DetectorNode:
    def __init__(self):
        self.posePublisher_front = rospy.Publisher('/pose_estimator/charger_pose/detection_front', PoseStamped,
                                                   queue_size=1)
        self.posePublisher_back = rospy.Publisher('/pose_estimator/charger_pose/detection_back', PoseStamped,
                                                  queue_size=1)
        self.keypointsPublisher = rospy.Publisher('/pose_estimator/keypoints', KeypointsWithCovarianceStamped,
                                                  queue_size=1)
        self.gpsPublisher = rospy.Publisher('/pose_estimator/gps', NavSatFix,
                                            queue_size=1)
        self.headPublisher = rospy.Publisher('/pose_estimator/head', NavRELPOSNED,
                                             queue_size=1)
        self.blackfly_topic = '/blackfly/camera/image_color/compressed'
        self.pointgrey_topic = '/pointgrey/camera/image_color/compressed'
        # self.camera_topic = '/video_player/compressed'
        self.imu_topic = '/xsens/data'
        self.gt_pose_topic = '/dgps_ublox/dgps_base/fix'
        self.gt_head_topic = '/dgps_ublox/dgps_rover/navrelposned'
        rospy.init_node('deep_pose_estimator', log_level=rospy.DEBUG)
        # path_to_model_bottom = "/root/share/tf/Keras/09_05_bottom_PP"
        path_to_model_bottom = "/root/share/tf/Keras/18_06_PP_4_wo_mask_bigger_head"
        # path_to_model_front = "/root/share/tf/Keras/2_10_8x8"
        # path_to_model_front = "/root/share/tf/Keras/15_10_heatmap_final"
        path_to_model_front = "/root/share/tf/Keras/2_11_AR"
        # path_to_pole_model = os.path.join("/root/share/tf/Faster/pole/model_Inea_3", 'frozen_inference_graph.pb')
        path_to_pole_model = os.path.join('/root/share/tf/YOLO/', '18_09trained_weights_final.h5')
        # path_to_pole_model = os.path.join('/root/share/tf/YOLO/', '13_10trained_weights_final.h5')
        self.num_points_front = 4
        self.equalize_histogram = False
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

        if self.num_points_front == 5:
            self.object_points = [[-0.3838, -0.0252, -0.6103], [0.3739, -0.0185, -0.6131], [2.7607, 0.7064, -0.1480],
                                  [2.8160, -0.9127, -0.1428], [-0.1048, -0.7433, -0.0434]]
        elif self.num_points_front == 4:
            self.object_points = [[0.0145, -0.1796, -0.6472], [2.7137, 0.7782, -0.0808], [2.3398, -0.5353, -0.0608],
                                  [0.2374, -0.5498, -0.0778]]
        elif self.num_points_front == 9:
            self.object_points = [[0.0145, -0.1796, -0.6472], [2.7137, 0.7782, -0.0808], [2.3398, -0.5353, -0.0608],
                                  [0.2374, -0.5498, -0.0778], [-0.3838, -0.0252, -0.6103], [0.3739, -0.0185, -0.6131],
                                  [2.7607, 0.7064, -0.1480], [2.8160, -0.9127, -0.1428], [-0.1048, -0.7433, -0.0434]]
        # for idx, pt in enumerate(self.object_points):
        #     self.object_points[idx] = rotate((0, 0, 0), pt, math.radians(5.5))
        # self.pitch = None
        # self.frame_pitch = None
        self.keypoints = None
        self.uncertainty = None
        self.gt_keypoints = []
        # self.gt_keypoints_slice = []
        self.blackfly_image_msg = None
        self.blackfly_image = None
        self.pointgrey_image_msg = None
        self.pointgrey_image = None
        self.gt_pose = None
        # self.gt_mat = None
        self.frame_gt = None
        # self.frame_scale = None
        # Detection performance
        self.all_frames = 0
        self.frames_sent_to_detector = 0
        self.detected_frames = 0
        # Uncertainty estimation
        self.failed_detections = 0
        self.kp_predictions = []
        self.cov_matrices = np.empty((8, 8, 0), np.float)
        self.prediction_errors = []

        # Initialize detector
        self.pointgrey_frame_shape = (5, 5)  # self.get_image_shape(self.pointgrey_topic)
        self.frame_shape = (960, 960)
        # self.frame_shape = self.get_image_shape(self.blackfly_topic)
        self.detector = Detector(path_to_model_front,
                                 path_to_pole_model,
                                 self.num_points_front)  # , path_to_model_bottom=path_to_model_bottom)
        self.detector.init_size(self.frame_shape)
        # self.detector.init_size((5000,5000))
        self.pose_estimator = PoseEstimator(self.blackfly_camera_matrix)

    def get_image_shape(self, camera_topic):
        im = rospy.wait_for_message(camera_topic, CompressedImage)
        np_arr = np.fromstring(im.data, np.uint8)
        image_shape = cv2.imdecode(np_arr, -1).shape[:2]
        return list(image_shape)


    def start(self):
        # rospy.Subscriber(self.blackfly_topic, CompressedImage, self.update_blackfly_image, queue_size=1)
        # rospy.Subscriber(self.pointgrey_topic, CompressedImage, self.update_pointgrey_image, queue_size=1)
        # rospy.Subscriber(self.imu_topic, Imu, self.get_pitch, queue_size=1)
        rospy.Subscriber(self.gt_pose_topic, NavSatFix, self.update_gt, queue_size=1)
        rospy.Subscriber(self.gt_head_topic, NavRELPOSNED, self.update_head, queue_size=1)
        # rospy.wait_for_message(self.pointgrey_topic, CompressedImage)
        # rospy.wait_for_message(self.blackfly_topic, CompressedImage)
        # base_path = '/root/share/tf/dataset/14_08_4pts_960_ROI_448'
        base_path = '/root/share/tf/dataset/artag/out_close'
        images = os.listdir(os.path.join(base_path, 'annotations'))
        images = iter(images)
        while not rospy.is_shutdown():
            self.gt_keypoints = []
            # try:
            fname = next(images)
            # if 'train1' in fname:
            img_fname = os.path.join(base_path, 'images/', fname[:-4] + '.png')
            ann_fname = os.path.join(base_path, 'annotations/', fname)
            print(img_fname)

            tree = ET.parse(ann_fname)
            root = tree.getroot()
            if not os.path.exists(img_fname):
                continue
            self.blackfly_image = cv2.imread(img_fname)
            #
            #         size = root.find('size')
            #         w = int(size.find('width').text)
            #         h = int(size.find('height').text)
            #         for object in root.findall('object'):
            #             scale = float(object.find('scale').text)
            #             # print("scale", scale)
            #             bbox = object.find('bndbox')
            #
            #             xmin = int(float(bbox.find('xmin').text) * w)
            #             ymin = int(float(bbox.find('ymin').text) * h)
            #             xmax = int(float(bbox.find('xmax').text) * w)
            #             ymax = int(float(bbox.find('ymax').text) * h)
            #
            #             # image = self.blackfly_image[ymin:ymax, xmin:xmax]
            #             # self.blackfly_image = cv2.resize(image, (w, h))
            #
            #             kps = object.find('keypoints')
            #             # keypoints = []
            #             # print(kps.find('keypoint6'))
            #             for i in range(4):
            #                 kp = kps.find('keypoint' + str(i))
            #                 self.gt_keypoints.append(
            #                     (int(float(kp.find('x').text) * w), int((float(kp.find('y').text) * h))))
            #             # for i, kp in enumerate(keypoints):
            #             #     self.gt_keypoints.append(
            #             #         ((int(kp[0] * w)), (int(kp[1] * h))))
            #     except StopIteration:
            #         # sum_cov_mtx = np.zeros((10, 10))
            #         sum_cov_mtx = np.zeros((8, 8))
            #         for single_img_pred in self.prediction_errors:
            #             cov_matrix = np.matmul(single_img_pred, np.transpose(single_img_pred))
            #             sum_cov_mtx += cov_matrix
            #         print(sum_cov_mtx / len(self.prediction_errors))
            #         print("Fails ratio", self.failed_detections / self.frames_sent_to_detector * 100)
            #         break



            if self.blackfly_image is not None:  # and self.pointgrey_image is not None:
                self.frame_gt = self.gt_pose
                # self.frame_pitch = self.pitch
                # self.frame_scale = self.detector.scale
                k = cv2.waitKey(5)
                if k == ord('q') or k == 27:
                    exit(0)
                if k == ord('z'):
                    # sum_cov_mtx = np.zeros((10, 10))
                    sum_cov_mtx = np.zeros((8, 8))
                    for single_img_pred in self.prediction_errors:
                        cov_matrices = np.matmul(single_img_pred, np.transpose(single_img_pred))
                        sum_cov_mtx += cov_matrices
                    print(sum_cov_mtx / len(self.prediction_errors))
                    print("Fails ratio", self.failed_detections / self.frames_sent_to_detector * 100)
                if self.detector.bottom:
                    self.detect(self.pointgrey_image, self.pointgrey_image_msg.header.stamp, self.frame_gt)
                else:
                    # self.detect(self.blackfly_image, self.blackfly_image_msg.header.stamp, self.frame_gt, self.gt_head)
                    self.detect(self.blackfly_image, None, None, None)

        # self.detector.yolo.close_session()
        rospy.spin()

    def update_gt(self, gt_msg):
        self.gt_pose = gt_msg

    def update_head(self, gt_msg):
        self.gt_head = gt_msg

    def update_blackfly_image(self, image_msg):
        self.all_frames += 1
        self.blackfly_image_msg = image_msg
        self.frame_gt = self.gt_pose
        self.frame_head = self.gt_head
        # self.frame_pitch = self.pitch
        np_arr = np.fromstring(image_msg.data, np.uint8)
        image = cv2.imdecode(np_arr, -1)
        image = cv2.undistort(image, self.blackfly_camera_matrix, self.blackfly_camera_distortion)
        # cv2.imwrite("/root/share/tf/image.png", image)
        # self.frame_shape = list(image.shape[:2])
        if self.equalize_histogram:
            img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            img_yuv[:, :, 2] = cv2.equalizeHist(img_yuv[:, :, 2])
            image = cv2.cvtColor(img_yuv, cv2.COLOR_HSV2BGR)
        self.blackfly_image = image
        # self.blackfly_image = cv2.resize(image, None, fx=self.detector.scale, fy=self.detector.scale)

    def update_pointgrey_image(self, image_msg):
        self.pointgrey_image_msg = image_msg
        self.frame_gt = self.gt_pose
        # self.frame_pitch = self.pitch
        np_arr = np.fromstring(image_msg.data, np.uint8)
        image = cv2.imdecode(np_arr, -1)
        # print(image.shape)
        image = cv2.undistort(image, self.pointgrey_camera_matrix, self.pointgrey_camera_distortion)
        # self.pointgrey_frame_shape = list(image.shape[:2])
        self.pointgrey_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)  # cv2.resize(image, None, fx=0.7, fy=0.7)

    def detect(self, frame, stamp, gt_pose, gt_heading):
        start_time = time.time()
        disp = np.copy(frame)
        working_copy = np.copy(frame)
        self.detector.detect(working_copy, gt_pose, self.gt_keypoints)
        self.frames_sent_to_detector += 1
        if self.detector.best_detection is not None:
            self.keypoints = self.detector.best_detection['keypoints']
            # self.uncertainty = self.detector.best_detection['heatmap_uncertainty']
            self.uncertainty = self.detector.best_detection['uncertainty']
            # self.uncertainty = np.zeros((self.num_points_front*2, self.num_points_front*2))
            # for i, unc in enumerate(raw_uncertainty):
            #     self.uncertainty[i*2:i*2+2, i*2:i*2+2] = unc
            # print(self.keypoints)
            # print(np.sqrt(self.uncertainty[..., 0, 0]), np.sqrt(self.uncertainty[..., 1, 1]), self.keypoints)#, self.gt_keypoints)
            # print("sigma:", self.uncertainty)
            for i, pt in enumerate(self.keypoints):
                sigma = self.uncertainty[i]
                # print(np.sqrt(sigma[0, 0]), np.sqrt(sigma[1, 1]))
                # gt_kp = self.gt_keypoints[i]
                # cv2.circle(disp, (int(pt[0]), int(pt[1])), 15, (0, 0, 255), -1)
                # cv2.circle(disp, (int(pt[0]), int(pt[1])), 10, (0, 0, 0), -1)
                # cv2.ellipse(disp, (int(pt[0]), int(pt[1])), (int(np.sqrt(sigma[0, 0])), int(np.sqrt(sigma[1, 1]))),
                #             angle=0, startAngle=0, endAngle=360, color=(0, 255, 255), thickness=3)
                cv2.ellipse(disp, (int(pt[0]), int(pt[1])), (
                int(np.ceil(np.sqrt(self.uncertainty[i, i]))), int(np.ceil(np.sqrt(self.uncertainty[i + 1, i + 1])))),
                            angle=0, startAngle=0, endAngle=360, color=(0, 0, 255), thickness=1)
                cv2.ellipse(disp, (int(pt[0]), int(pt[1])), (int(np.ceil(2 * np.sqrt(self.uncertainty[i, i]))),
                                                             int(np.ceil(2 * np.sqrt(self.uncertainty[i + 1, i + 1])))),
                            angle=0, startAngle=0, endAngle=360, color=(0, 255, 255), thickness=1)
                cv2.ellipse(disp, (int(pt[0]), int(pt[1])), (int(np.ceil(3 * np.sqrt(self.uncertainty[i, i]))),
                                                             int(np.ceil(3 * np.sqrt(self.uncertainty[i + 1, i + 1])))),
                            angle=0, startAngle=0, endAngle=360, color=(0, 255, 0), thickness=1)
                # cv2.circle(disp, gt_kp, 5, (255, 0, 255), -1)
            cv2.imshow("detection", cv2.resize(disp, self.detector.slice_size))
            # cv2.imshow("detection", disp[2000:, 2000:])
            cv2.waitKey(0)
            self.detected_frames += 1
            if self.detector.bottom:
                camera_matrix = self.pointgrey_camera_matrix
            else:
                camera_matrix = self.blackfly_camera_matrix
            # self.publish_keyponts(stamp)
            # self.publish_pose(stamp, camera_matrix)
            # self.gpsPublisher.publish(gt_pose)
            # self.headPublisher.publish(gt_heading)
            self.blackfly_image = None
            # print("detection time:", time.time() - start_time)

    def publish_pose(self, stamp, camera_matrix):
        tvec, rvec = self.pose_estimator.calc_PnP_pose(self.keypoints, camera_matrix)
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
        # out_msg.covariance = np.reshape(self.uncertainty, (-1))
        out_msg.covariance = np.zeros(64)
        out_msg.keypoints = []

        # out_msg.header.frame_id = "camera"

        def calc_dist(x, z):
            # return math.sqrt((x[0]-z[0]) ** 2 + (x[1]-z[1]) ** 2)
            return abs(x[0] - z[0]), abs(x[1] - z[1])

        # single_img_pred = []
        for idx, kp in enumerate(self.keypoints):
            # if calc_dist(kp, self.gt_keypoints[idx])[1] > 50:
            #     self.failed_detections += 1
            #     print("Fails ratio", self.failed_detections / self.frames_sent_to_detector * 100)
            #     return
            # single_img_pred.append(calc_dist(kp, self.gt_keypoints[idx]))
            out_msg.keypoints.append(kp[0])
            out_msg.keypoints.append(kp[1])
        # print("unc", type(np.reshape(self.uncertainty, (-1))), np.reshape(self.uncertainty, (-1)).dtype)
        # for unc in np.reshape(self.uncertainty[idx], (-1)):
        # self.prediction_errors.append(np.array(single_img_pred).reshape((8, 1)))

        self.keypointsPublisher.publish(out_msg)


if __name__ == '__main__':
    det_node = DetectorNode()
    det_node.start()
