#!/usr/bin/env python3

import os
import numpy as np
# try:
#     from cv2 import cv2
# except ImportError:
#     pass
import cv2
from Detector import Detector
from PoseEstimator import PoseEstimator
import rospy
from sensor_msgs.msg import CompressedImage, Imu
from geometry_msgs.msg import PoseStamped, Point32, Quaternion
from scipy.spatial.transform import Rotation
from deep_pose_estimator.msg import ImageKeypoints
import time

from sklearn.mixture import BayesianGaussianMixture
from sklearn.decomposition import PCA

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

class DetectorNode:
    def __init__(self):
        self.posePublisher_front = rospy.Publisher('/pose_estimator/charger_pose/detection_front', PoseStamped,
                                                   queue_size=1)
        self.posePublisher_back = rospy.Publisher('/pose_estimator/charger_pose/detection_back', PoseStamped,
                                                  queue_size=1)
        self.keypointsPublisher = rospy.Publisher('/pose_estimator/keypoints', ImageKeypoints, queue_size=1)
        self.blackfly_topic = '/blackfly/camera/image_color/compressed'
        self.pointgrey_topic = '/pointgrey/camera/image_color/compressed'
        # self.camera_topic = '/video_player/compressed'
        self.imu_topic = '/xsens/data'
        self.gt_pose_topic = '/pose_estimator/charger_pose/location_gps'
        rospy.init_node('deep_pose_estimator', log_level=rospy.DEBUG)
        path_to_model_bottom = "/root/share/tf/Keras/09_05_bottom_PP"
        # path_to_model_bottom = "/root/share/tf/Keras/15_05_bottom_PP_augg_inc"
        path_to_model = "/root/share/tf/Keras/1_06_PP_7"
        # path_to_model = "/root/share/tf/Keras/27_05_PP"
        # path_to_model = "/root/share/tf/Keras/22_05_PP_aug4_2112"
        path_to_pole_model = os.path.join("/root/share/tf/Faster/pole/model_Inea_3", 'frozen_inference_graph.pb')
        self.equalize_histogram = False
        self.blackfly_camera_matrix = np.array([[4996.73451, 0, 2732.95188],
                                                [0, 4992.93867, 1890.88113],
                                                [0, 0, 1]])
        self.blackfly_camera_distortion = (-0.11286, 0.11138, 0.00195, -0.00166, 0.00000)
        self.pointgrey_camera_matrix = np.array(
            [[678.170160908967, 0, 511.0007922166781],  # change also in update_pointgrey_image TODO:fix
             [0, 678.4827850057917, 642.472979517798],
             [0, 0, 1]]).astype(np.float64)
        self.pointgrey_camera_distortion = (-0.030122176489, 0.03641181142582, 0.001298022247894, -0.001118918)

        self.pitch = None
        self.frame_pitch = None
        self.keypoints = None
        self.image_msg = None
        self.image = None
        self.pointgrey_image_msg = None
        self.pointgrey_image = None
        self.gt_pose = None
        self.gt_mat = None
        self.frame_gt = None
        self.frame_scale = None
        self.all_frames = 0
        self.frames_sent_to_detector = 0
        self.detected_frames = 0

        self.pointgrey_frame_shape = self.get_image_shape(self.pointgrey_topic)
        self.frame_shape = self.get_image_shape(self.blackfly_topic)
        self.detector = Detector(path_to_model, path_to_pole_model)  # , path_to_model_bottom=path_to_model_bottom)
        self.detector.init_size(self.frame_shape)
        # self.detector.init_size((5000,5000))
        self.pose_estimator = PoseEstimator(self.blackfly_camera_matrix)

    def get_image_shape(self, camera_topic):
        im = rospy.wait_for_message(camera_topic, CompressedImage)
        np_arr = np.fromstring(im.data, np.uint8)
        image_shape = cv2.imdecode(np_arr, -1).shape[:2]
        print("imsh", image_shape)
        return list(image_shape)

    def start(self):
        rospy.Subscriber(self.blackfly_topic, CompressedImage, self.update_blackfly_image, queue_size=1)
        rospy.Subscriber(self.pointgrey_topic, CompressedImage, self.update_pointgrey_image, queue_size=1)
        rospy.Subscriber(self.imu_topic, Imu, self.get_pitch, queue_size=1)
        rospy.Subscriber(self.gt_pose_topic, PoseStamped, self.update_gt, queue_size=1)
        rospy.wait_for_message(self.pointgrey_topic, CompressedImage)
        rospy.wait_for_message(self.blackfly_topic, CompressedImage)
        while not rospy.is_shutdown():
            if self.image is not None and self.pointgrey_image is not None:
                self.frame_gt = self.gt_pose
                self.frame_pitch = self.pitch
                self.frame_scale = self.detector.scale
                k = cv2.waitKey(1)
                if k == ord('q') or k == 27:
                    exit(0)
                if self.detector.bottom:
                    self.detect(self.pointgrey_image, self.pointgrey_image_msg.header.stamp, self.frame_gt,
                                self.frame_pitch)
                else:
                    self.detect(self.image, self.image_msg.header.stamp, self.frame_gt, self.frame_pitch)

        if rospy.is_shutdown():
            np_plot = np.array(self.pose_estimator.plot_data)
            rospy.loginfo("avg_err_PnP: %f", np.average(np_plot[:, 4]))
            # rospy.loginfo("avg_err_PnP 15000: %f", np.average(np_plot[np_plot[:, 6]>-15000, 4]))
            # rospy.loginfo("avg_err_PnP 20000: %f", np.average(np_plot[np_plot[:, 6]>-20000, 4]))
            # rospy.loginfo("avg_err_PnP 30000: %f", np.average(np_plot[np_plot[:, 6]>-30000, 4]))
            rospy.loginfo("avg_err_PnP >40m: %f", np.average(np_plot[np_plot[:, 1] > 40, 4]))
            rospy.loginfo("avg_err_PnP <40m: %f", np.average(np_plot[np_plot[:, 1] < 40, 4]))
            rospy.loginfo("avg_err_mask: %f", np.average(np_plot[:, 3]))
            rospy.loginfo("detections coverage: %f", self.detected_frames/self.all_frames*100)
            rospy.loginfo("all_frames: %d", self.all_frames)
            rospy.loginfo("frames_sent_to_detector: %d", self.frames_sent_to_detector)
            rospy.loginfo("detected_frames: %d", self.detected_frames)
        rospy.spin()

    def get_pitch(self, imu_msg):
        quat = imu_msg.orientation
        quat = np.array([quat.x, quat.y, quat.z, quat.w], dtype=np.float64)
        r = Rotation.from_quat(quat)
        euler = r.as_euler('xyz')
        self.pitch = euler[1]

    def update_gt(self, gt_msg):
        self.gt_pose = gt_msg
        self.gt_mat = np.identity(4)
        self.gt_mat[0, 3] = self.gt_pose.pose.position.x
        self.gt_mat[1, 3] = self.gt_pose.pose.position.y
        self.gt_mat[2, 3] = self.gt_pose.pose.position.z

    def update_blackfly_image(self, image_msg):
        self.all_frames += 1
        self.image_msg = image_msg
        self.frame_gt = self.gt_pose
        self.frame_pitch = self.pitch
        np_arr = np.fromstring(image_msg.data, np.uint8)
        image = cv2.imdecode(np_arr, -1)
        image = cv2.undistort(image, self.blackfly_camera_matrix, self.blackfly_camera_distortion)
        self.frame_shape = list(image.shape[:2])
        if self.equalize_histogram:
            img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            img_yuv[:, :, 2] = cv2.equalizeHist(img_yuv[:, :, 2])
            image = cv2.cvtColor(img_yuv, cv2.COLOR_HSV2BGR)
        self.image = cv2.resize(image, None, fx=self.detector.scale, fy=self.detector.scale)

    def update_pointgrey_image(self, image_msg):
        self.pointgrey_image_msg = image_msg
        self.frame_gt = self.gt_pose
        self.frame_pitch = self.pitch
        np_arr = np.fromstring(image_msg.data, np.uint8)
        image = cv2.imdecode(np_arr, -1)
        # print(image.shape)
        camera_matrix = np.array([[678.170160908967, 0, 642.472979517798],
                                  [0, 678.4827850057917, 511.0007922166781],
                                  [0, 0, 1]]).astype(np.float64)
        image = cv2.undistort(image, camera_matrix, self.pointgrey_camera_distortion)
        self.pointgrey_frame_shape = list(image.shape[:2])
        self.pointgrey_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)  #cv2.resize(image, None, fx=0.7, fy=0.7)

    def detect(self, frame, stamp, gt_pose, pitch):
        start_time = time.time()
        disp = np.copy(frame)
        working_copy = np.copy(frame)
        self.detector.detect(working_copy, gt_pose)
        self.frames_sent_to_detector += 1
        if self.detector.best_detection is not None:
            self.keypoints = self.detector.best_detection[
                'keypoints']  # [self.detector.best_detection['keypoints'][0], self.detector.best_detection['keypoints'][1],
            # self.detector.best_detection['keypoints'][2], self.detector.best_detection['keypoints'][3],
            # self.detector.best_detection['keypoints'][5], self.detector.best_detection['keypoints'][6]]
            for idx, pt in enumerate(self.keypoints):
                cv2.circle(disp, (int(pt[0]), int(pt[1])), 10, (255, 255, 255), -1)
            self.detected_frames += 1
            # print("self.frame_scale", self.frame_scale)
            # print("self.detector.best_detection", self.detector.best_detection)
            # print("self.detector.best_detection['score']", self.detector.best_detection['score'])
            if self.detector.bottom:
                camera_matrix = self.pointgrey_camera_matrix
            else:
                camera_matrix = self.blackfly_camera_matrix
            if self.detector.best_detection['score'] > 0.5:
                self.keypoints = np.multiply(self.keypoints, 1 / self.frame_scale)
                self.publish_keyponts(stamp, working_copy)
                self.publish_pose(stamp, camera_matrix)
            # cv2.imshow('detection', cv2.resize(disp, (1280, 960)))
        self.image = None
        # print("detection time:", time.time()-start_time)

    def publish_pose(self, stamp, camera_matrix):
        tvec, rvec = self.pose_estimator.calc_PnP_pose(self.keypoints, camera_matrix)
        rot = Rotation.from_rotvec(np.squeeze(rvec))
        out_msg = PoseStamped()
        out_msg.header.stamp = stamp
        out_msg.header.frame_id = "camera"
        out_msg.pose.position.x = tvec[0]
        out_msg.pose.position.y = tvec[1]
        out_msg.pose.position.z = tvec[2]
        # if self.detector.bottom:
        #     out_msg.pose.position.z = tvec[2]-5.0
        np_quat = rot.as_quat()
        ros_quat = Quaternion(np_quat[0], np_quat[1], np_quat[2], np_quat[3])

        out_msg.pose.orientation = ros_quat
        if self.detector.bottom:
            self.posePublisher_back.publish(out_msg)
        else:
            self.posePublisher_front.publish(out_msg)

    def publish_keyponts(self, stamp, frame):
        out_msg = ImageKeypoints()
        out_msg.header.stamp = stamp
        out_msg.offset.x = self.detector.offset[0]
        out_msg.offset.y = self.detector.offset[1]
        out_msg.offset.z = 0
        out_msg.image.header.stamp = stamp
        out_msg.image.format = 'png'
        out_msg.image.data = np.array(cv2.imencode('.png', self.detector.get_slice(frame))[1]).tostring()
        for kp in self.keypoints:
            point = Point32()
            point.x = kp[0]
            point.y = kp[1]
            point.z = 0
            out_msg.keypoints.append(point)
        self.keypointsPublisher.publish(out_msg)


if __name__ == '__main__':
    det_node = DetectorNode()
    det_node.start()
