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

from sklearn.mixture import BayesianGaussianMixture
from sklearn.decomposition import PCA


class DetectorNode:

    def __init__(self):
        self.camera_topic = '/blackfly/camera/image_color/compressed'
        # self.camera_topic = '/video_player/compressed'
        self.imu_topic = '/xsens/data'
        self.gt_pose_topic = '/pose_estimator/charger_pose/gt'
        rospy.init_node('deep_pose_estimator', log_level=rospy.DEBUG)
        self.scale_factor = 1.0
        path_to_model = "/root/share/tf/Keras/11_06"
        path_to_pole_model = os.path.join("/root/share/tf/Faster/pole/model_Inea", 'frozen_inference_graph.pb')
        self.camera_matrix = np.array([[4996.73451*self.scale_factor, 0, 2732.95188*self.scale_factor],
                                       [0, 4992.93867*self.scale_factor, 1890.88113*self.scale_factor],
                                       [0, 0, 1]])
        self.detector = Detector(path_to_model, path_to_pole_model, self.camera_matrix)
        self.frame_shape = self.get_image_shape()
        self.frame_shape = [int(self.frame_shape[0]*self.scale_factor), int(self.frame_shape[1]*self.scale_factor)]
        self.detector.init_size(self.frame_shape)
        self.pitch = None
        self.frame_pitch = None
        self.keypoints = None
        self.image_msg = None
        self.image = None
        self.gt_pose = None
        self.r_mat = None
        self.gt_mat = None
        self.frame_gt = None
        self.all_frames = 0
        self.frames_sent_to_detector = 0
        self.detected_frames = 0
        # self.camera_matrix = np.array([[1929.14559, 0, 1924.38974],
        #                                [0, 1924.07499, 1100.54838],
        #                                [0, 0, 1]])
        self.pose_estimator = PoseEstimator(self.camera_matrix)
        self.posePublisher = rospy.Publisher('/pose_estimator/charger_pose/detection', PoseStamped, queue_size=1)
        self.keypointsPublisher = rospy.Publisher('/pose_estimator/keypoints', ImageKeypoints, queue_size=1)

    def get_image_shape(self):
        im = rospy.wait_for_message(self.camera_topic, CompressedImage)
        np_arr = np.fromstring(im.data, np.uint8)
        image_shape = cv2.imdecode(np_arr, -1).shape[:2]
        return list(image_shape)

    def start(self):
        rospy.Subscriber(self.camera_topic, CompressedImage, self.update_image, queue_size=1)
        rospy.Subscriber(self.imu_topic, Imu, self.get_pitch, queue_size=1)
        rospy.Subscriber(self.gt_pose_topic, PoseStamped, self.update_gt, queue_size=1)
        while not rospy.is_shutdown():
            if self.image is not None:
                self.frame_gt = self.gt_pose
                self.frame_pitch = self.pitch
                self.detect(self.image, self.image_msg.header.stamp, self.frame_gt, self.frame_pitch)
                # if self.detector.best_detection is not None:
                    # self.mask_fv.append(np.reshape(self.detector.best_detection["bcf"], [-1]))
                    # rospy.logdebug(len(self.mask_fv))
                    # if len(self.mask_fv)==40:
                    #     self.GMM.fit(self.mask_fv)
                    # if len(self.mask_fv)>40:

        if rospy.is_shutdown():
            # np.save("/root/share/tf/mask_fv", self.detector.mask_fv)
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

    def update_image(self, image_msg):
        self.all_frames += 1
        self.image_msg = image_msg
        self.frame_gt = self.gt_pose
        self.frame_pitch = self.pitch
        np_arr = np.fromstring(image_msg.data, np.uint8)
        image = cv2.imdecode(np_arr, -1)
        self.image = cv2.resize(image, None, fx=self.scale_factor, fy=self.scale_factor)

        # img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # img_yuv[:, :, 2] = cv2.equalizeHist(img_yuv[:, :, 2])
        # image = cv2.cvtColor(img_yuv, cv2.COLOR_HSV2BGR)
        # self.image = cv2.resize(image, (self.frame_shape[1], self.frame_shape[0]))
        # cv2.imshow('look', cv2.resize(image, (1280, 960)))

    def detect(self, frame, stamp, gt_pose, pitch):
        disp = np.copy(frame)
        working_copy = np.copy(frame)
        self.detector.detect(working_copy, gt_pose)
        self.frames_sent_to_detector += 1
        if self.detector.best_detection is not None:
            # print("gt_mat\n", self.gt_mat)
            # print(-np.matmul(np.linalg.inv(self.r_mat[:3, :3]), self.r_mat[:3, 3]))
            for idx, pt in enumerate(self.detector.best_detection['keypoints']):
                cv2.circle(disp, (int(pt[0]), int(pt[1])), 10, (0,255,0), -1)
                x1, y1, x2, y2 = self.detector.best_detection['abs_rect']
                mask = np.zeros(self.image.shape[:2], np.uint8)
                # mask[y1:y2, x1:x2] = np.where(self.detector.mask_reshaped > 0.1, 1, 0)
                mask[self.detector.offset[0]:self.detector.offset[0]+self.detector.slice_size[0], self.detector.offset[1]:self.detector.offset[1]+self.detector.slice_size[1]] = self.detector.best_detection['mask']
                # imgray = self.detector.mask_reshaped * 255
                # ret, thresh = cv2.threshold(imgray.astype(np.uint8), 27, 255, 0)
                # contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                # if len(contours) > 0:
                #     contour = np.add(contours[0], [x1, y1])
                #     cv2.drawContours(disp, [contour], 0, (255, 250, 250), 2)
                gray = np.zeros_like(frame)  # skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
                disp = np.where(np.expand_dims(mask, -1), gray, disp).astype(np.uint8)
            # rospy.logerr('detection success')
            self.detected_frames += 1

            self.keypoints = self.detector.best_detection['keypoints']
            self.publish_keyponts(stamp, working_copy)
            if self.gt_pose is not None:
                 self.publish_pose(stamp, gt_pose, pitch)
            cv2.imshow('detection', cv2.resize(disp, (1280, 960)))
            cv2.waitKey(10)
        # else:
        #     rospy.logerr('detection failed')
        self.image = None

    def publish_pose(self, stamp, gt_pose, pitch):
        if self.detector.mask_reshaped is not None:
            # feat = self.detector.pca.transform([np.reshape(self.detector.best_detection["bcf"], [-1])])
            # score = self.detector.GMM.score(feat)
            # rospy.logdebug(score)
            self.pose_estimator.calc_error(gt_pose, np.argwhere(self.detector.mask_reshaped > 0.1),
                                           self.detector.best_detection['abs_rect'], self.keypoints, pitch,
                                           self.detector.best_posecnn_detection)
        tvec, rvec = self.pose_estimator.calc_PnP_pose(self.keypoints)
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
        self.posePublisher.publish(out_msg)

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
