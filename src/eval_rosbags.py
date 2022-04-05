import os
import subprocess
import signal
from tqdm import tqdm

# paths = ["/root/share/tf/dataset/2020-11-09", "/root/share/tf/dataset/2020-11-13", "/root/share/tf/dataset/2020-11-25", "/root/share/tf/dataset/2020-12-14", "/root/share/tf/dataset/2020-12-16"]
paths = ["/root/share/tf/dataset/eval_ds_clean/2020-11-09", "/root/share/tf/dataset/eval_ds_clean/2020-11-13", "/root/share/tf/dataset/eval_ds_clean/2020-11-25", "/root/share/tf/dataset/eval_ds_clean/2020-12-14", "/root/share/tf/dataset/eval_ds_clean/2020-12-16"]

out_path = "/root/share/tf/results_2.0"
version = "AMCS_kpts" #ROSBAG SPEED .25!!!!
for path in tqdm(paths, "Dirs"):
        os.chdir(path)
        day = path.split('/')[-1]
        if not os.path.exists(os.path.join(out_path, version, day)):
                os.makedirs(os.path.join(out_path, version, day))
        files = os.listdir(path)

        for filename in tqdm(files, "Files", leave=False):
                if filename.endswith(".bag"):                      
                        play_command = "rosbag play --topics /dgps_ublox/dgps_base/fix /dgps_ublox/dgps_rover/navrelposned  /processed_can_data /blackfly/camera/image_color/compressed -r 0.004 -s 0 " +os.path.join(path, filename)
                        record_command = "rosbag record /dgps_ublox/dgps_base/fix /dgps_ublox/dgps_rover/navrelposned /pose_estimator/keypoints /processed_can_data -O " + os.path.join(out_path, version, day, filename[:-4]+"_det.bag")
                        play_process = subprocess.Popen(play_command, stdin=subprocess.PIPE, shell=True)
                        record_process = subprocess.Popen(record_command, stdin=subprocess.PIPE, shell=True, preexec_fn=os.setsid)
                        play_process.wait()
                        os.killpg(os.getpgid(record_process.pid), signal.SIGINT)


for path in tqdm(paths, "Dirs"):
        os.chdir(path)
        day = path.split('/')[-1]
        if not os.path.exists(os.path.join(out_path, version, day)):
                os.makedirs(os.path.join(out_path, version, day))
        files = os.listdir(path)
        for filename in tqdm(files, "Files", leave=False):
                if filename.startswith(("_2020-11-09-12-43-04")):                       
                # if filename.startswith(("_2020-11-13-13-43-53")):                       
                        play_command = "rosbag play --topics /dgps_ublox/dgps_base/fix /dgps_ublox/dgps_rover/navrelposned  /processed_can_data /blackfly/camera/image_color/compressed -r 0.2 -s 0 " +os.path.join(path, filename)
                        record_command = "rosbag record /dgps_ublox/dgps_base/fix /dgps_ublox/dgps_rover/navrelposned /pose_estimator/keypoints /processed_can_data -O " + os.path.join(out_path, version, day, filename[:-4]+"_det.bag")
                        play_process = subprocess.Popen(play_command, stdin=subprocess.PIPE, shell=True)
                        record_process = subprocess.Popen(record_command, stdin=subprocess.PIPE, shell=True, preexec_fn=os.setsid)
                        play_process.wait()
                        os.killpg(os.getpgid(record_process.pid), signal.SIGINT)