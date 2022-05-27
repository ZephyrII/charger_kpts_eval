import os
from os.path import join as opj
import numpy as np
import signal
from tqdm import tqdm
import rosbag

models = ["c_hrnet32_udp_512_hm512_repr_eval_v2"]
BASE_DIR = "/root/share/tf/results_2.0/"
paths = ["2020-11-09", "2020-11-13", "2020-11-25", "2020-12-14", "2020-12-16"]
# paths = ["/home/tomasz/share_docker/eval_ds_crops/2020-12-16"]


for model in models:
        res_dir = opj(BASE_DIR, model)
        pairs = [[0,1], [0,2], [0,3], [1,2], [1,3], [2,3]]
        total_frames = 0
        duplicates = 0
        for day in paths:
                path = opj(res_dir, day)
                os.chdir(path)
                files = os.listdir(path)

                for filename in files:
                        if filename.endswith(".bag"):
                            print(filename)
                            bag_in = rosbag.Bag(filename)
                            for topic, msg, t in bag_in.read_messages("/pose_estimator/keypoints"):
                                    kpts = np.array(msg.keypoints).reshape((4,2))
                                    dup = False
                                    total_frames +=1
                                    for pair in pairs:
                                            dist = np.linalg.norm(kpts[pair[0]]-kpts[pair[1]])
                                            if dist<10:
                                                    dup = True
                                                    # print(msg)
                                    if dup:
                                            duplicates +=1
        print(model+" total_frames:", total_frames, "duplicates:", duplicates, "error ratio:", duplicates/total_frames*100)
