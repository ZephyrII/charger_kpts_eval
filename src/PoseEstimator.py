import cv2

# try:
#     from cv2 import cv2
# except ImportError:
#     pass
import math
import numpy as np
from scipy.spatial.transform import Rotation

np.set_printoptions(linewidth=np.inf)
class PoseEstimator:

    def __init__(self, camera_matrix):
        self.camera_matrix = camera_matrix
        self.last_tvec = np.array([0.0, 0.0, 45.0])
        self.last_rvec = np.array([0.0, 0.0, 0.0])
        lastR = Rotation.from_euler('xyz', [np.deg2rad(0), 0, 0])
        self.last_rvec = lastR.as_rotvec()

    def calc_dist(self, x, z):
        return math.sqrt(x ** 2 + z ** 2)

    def calc_PnP_pose(self, image_points, object_points, camera_matrix):
        if image_points is None:
            return None
        PnP_image_points = np.array(image_points).astype(np.float64)
        object_points = np.array(object_points).astype(np.float64)

        retval, rvec, tvec = cv2.solvePnP(object_points, PnP_image_points, camera_matrix, distCoeffs=None,
                                          tvec=self.last_tvec, rvec=self.last_rvec, flags=cv2.SOLVEPNP_ITERATIVE,
                                          useExtrinsicGuess=True)

        rot = Rotation.from_rotvec(rvec)
        rot.as_euler('xyz')
        print('TVEC', tvec)
        print('RVEC', rvec, rot.as_euler('xyz') * 180 / 3.14)
        self.PnP_pose_data = tvec
        if tvec[2] < 50:  # filter out wrong estimations
            self.last_tvec = tvec
            self.last_rvec = rvec
        return tvec, rvec
