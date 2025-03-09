import os
import cv2
import json
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path
import torch
from time import time
import concurrent.futures
import subprocess 

import hamer
from hamer.configs import CACHE_DIR_HAMER
from hamer.models import download_models, load_hamer, DEFAULT_CHECKPOINT
from hamer.utils import recursive_to
from hamer.datasets.vitdet_dataset import ViTDetDataset
from hamer.utils.utils_detectron2 import DefaultPredictor_Lazy
from hamer.utils.renderer import cam_crop_to_full
from vitpose_model import ViTPoseModel
from detectron2.config import LazyConfig

# ------------------------------------------------------
def run_shell(command: str):
    print(f"[Running] {command}")
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print("[Error] Command failed:")
        print(result.stderr)
    else:
        print(result.stdout)


def run_in_bash(command: str):
    full_cmd = (
        "bash -i -c '"
        "source /root/miniconda3/etc/profile.d/conda.sh && "
        + command +
        "'"
    )
    run_shell(full_cmd)


def activate_hamer():
    run_in_bash("conda activate hamer && conda info")


# -----------------------------
def linear_interpolate_array(arr1, arr2, steps):
    arr1 = np.asarray(arr1)
    arr2 = np.asarray(arr2)
    if steps <= 0:
        return np.empty((0,) + arr1.shape, dtype=arr1.dtype)

    alphas = np.linspace(0, 1, steps + 2)[1:-1]  # (steps,)
    alphas = alphas[:, None, None]  
    return (1 - alphas)*arr1[None] + alphas*arr2[None]

# -----------------------------
def detect_hands_and_hamer(model, frame_bgr, detector, cpm, model_cfg, f_idx=None):

    det_out = detector(frame_bgr)
    instances = det_out['instances']
    valid_idx = (instances.pred_classes == 0) & (instances.scores > 0.6)
    pred_bboxes = instances.pred_boxes.tensor[valid_idx].cpu().numpy()
    pred_scores = instances.scores[valid_idx].cpu().numpy()
    if len(pred_bboxes) == 0:
        return {}

    vitposes_out = cpm.predict_pose(
        frame_bgr[:, :, ::-1],
        [np.concatenate([pred_bboxes, pred_scores[:, None]], axis=1)]
    )
    bboxes = []
    is_right_list = []
    for vitposes in vitposes_out:
        left_hand_kp = vitposes['keypoints'][-42:-21]
        right_hand_kp = vitposes['keypoints'][-21:]
        valid_left = left_hand_kp[:,2]>0.6
        valid_right = right_hand_kp[:,2]>0.6
        if np.sum(valid_left) > 5:
            bboxes.append([
                left_hand_kp[valid_left,0].min(),
                left_hand_kp[valid_left,1].min(),
                left_hand_kp[valid_left,0].max(),
                left_hand_kp[valid_left,1].max()
            ])
            is_right_list.append(0)  # 0=left
        if np.sum(valid_right) > 5:
            bboxes.append([
                right_hand_kp[valid_right,0].min(),
                right_hand_kp[valid_right,1].min(),
                right_hand_kp[valid_right,0].max(),
                right_hand_kp[valid_right,1].max()
            ])
            is_right_list.append(1) # 1 = right
    if not bboxes:
        return {}

    boxes_np = np.array(bboxes)
    is_right_arr = np.array(is_right_list, dtype=np.int32)
    dataset = ViTDetDataset(model_cfg, frame_bgr, boxes_np, is_right_arr, rescale_factor=2.0)
    loader = torch.utils.data.DataLoader(dataset, batch_size=len(bboxes), shuffle=False, num_workers=0)

    result = {}
    for batch_data in loader:
        batch_data = recursive_to(batch_data, device)
        with torch.no_grad():
            out = model(batch_data)

        pred_cam = out['pred_cam']  # (B,3)
        is_right_hand = batch_data['right'].cpu().numpy()  # (B,)

        if 'pred_keypoints_3d' in out:
            pred_joints_3d = out['pred_keypoints_3d'].cpu().numpy()  # (B,21,3) 
        else:
            pred_joints_3d = None

        multiplier = (2 * batch_data['right'] - 1)
        pred_cam[:, 1] = multiplier * pred_cam[:, 1]

        from hamer.utils.renderer import cam_crop_to_full
        cam_t_full = cam_crop_to_full(
            pred_cam, batch_data["box_center"], batch_data["box_size"], batch_data["img_size"],
            model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * batch_data["img_size"].max()
        ).cpu().numpy()

        B = len(is_right_hand)
        for i in range(B):
            r_label = int(is_right_hand[i])
            is_right_bool = (r_label == 1)
            j3d = None
            if pred_joints_3d is not None:
                j3d = pred_joints_3d[i]
                if not is_right_bool:
                    j3d[:,0] *= -1

            result[r_label] = {
                "joints_3d": j3d,
                "camera_translation": cam_t_full[i],
                "is_right": is_right_bool
            }

    return result

def process_keyframe(f_idx, frame, model, detector, cpm, model_cfg):
    return f_idx, detect_hands_and_hamer(model, frame, detector, cpm, model_cfg, f_idx=f_idx)


import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

def main():
    activate_hamer()

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default=DEFAULT_CHECKPOINT)
    parser.add_argument('--input_video', type=str, default='input.mp4')
    parser.add_argument('--max_frames', type=int, default=None)
    parser.add_argument('--keypoints_output', type=str, default='keypoints.json',
                        help='Output JSON path, containing {joints_3d, camera_translation, is_right}')
    args = parser.parse_args()

    from time import time
    total_start_time = time()

    from hamer.models import download_models, load_hamer
    from hamer.configs import CACHE_DIR_HAMER
    download_models(CACHE_DIR_HAMER)
    model_3d, model_cfg = load_hamer(args.checkpoint)
    model_3d = model_3d.to(device).eval()
    print("HAMER model loaded")

    from detectron2.config import LazyConfig
    from hamer.utils.utils_detectron2 import DefaultPredictor_Lazy
    from vitpose_model import ViTPoseModel
    from pathlib import Path
    import hamer

    cfg_path = Path(hamer.__file__).parent / 'configs' / 'cascade_mask_rcnn_vitdet_h_75ep.py'
    detectron2_cfg = LazyConfig.load(str(cfg_path))
    detectron2_cfg.train.init_checkpoint = (
        "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
    )
    detector = DefaultPredictor_Lazy(detectron2_cfg)
    cpm = ViTPoseModel(device)

    import cv2
    cap = cv2.VideoCapture(args.input_video)
    if not cap.isOpened():
        print(f"Error: cannot open {args.input_video}")
        return
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    print(f"Video loaded: frames={total_frames}, fps={fps}")

    import concurrent.futures
    frames_to_process = []
    keyframe_map = {}
    KEYFRAME_INTERVAL = 10
    for f_idx in range(total_frames):
        ret, frame = cap.read()
        if not ret or (args.max_frames and f_idx >= args.max_frames):
            break
        if f_idx % KEYFRAME_INTERVAL == 0 or f_idx == (total_frames -1):
            frames_to_process.append((f_idx, frame))
    cap.release()

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futs = [
            executor.submit(
                process_keyframe, f_idx, frame,
                model_3d, detector, cpm, model_cfg
            )
            for (f_idx, frame) in frames_to_process
        ]
        for fut in concurrent.futures.as_completed(futs):
            f_idx, data_dict = fut.result()
            keyframe_map[f_idx] = data_dict

    all_data = {}
    sorted_kfs = sorted(keyframe_map.keys())

    for i in range(len(sorted_kfs)-1):
        start_f, end_f = sorted_kfs[i], sorted_kfs[i+1]
        start_dict = keyframe_map[start_f] or {}
        end_dict = keyframe_map[end_f] or {}
        steps = end_f - start_f -1

        if start_dict:
            all_data[str(start_f)] = {}
            for r_label, val in start_dict.items():
                cam_t = val["camera_translation"]
                j3d = val["joints_3d"]
                is_r = val["is_right"]
                all_data[str(start_f)][str(r_label)] = {
                    "camera_translation": cam_t.tolist(),
                    "is_right": is_r
                }
                if j3d is not None:
                    all_data[str(start_f)][str(r_label)]["joints_3d"] = j3d.tolist()

        for r_label, s_val in start_dict.items():
            if r_label in end_dict:
                e_val = end_dict[r_label]
                s_cam = s_val["camera_translation"]
                e_cam = e_val["camera_translation"]
                s_j3d = s_val["joints_3d"]
                e_j3d = e_val["joints_3d"]
                s_isr = s_val["is_right"]

                if steps>0:
                    from time import time
                    cam_t_arr = linear_interpolate_array(s_cam, e_cam, steps)
                    if (s_j3d is not None) and (e_j3d is not None):
                        j3d_arr = linear_interpolate_array(s_j3d, e_j3d, steps)
                    else:
                        j3d_arr = None

                    for t_idx, mid_f in enumerate(range(start_f+1, end_f)):
                        f_str = str(mid_f)
                        if f_str not in all_data:
                            all_data[f_str] = {}
                        all_data[f_str][str(r_label)] = {
                            "camera_translation": cam_t_arr[t_idx].tolist(),
                            "is_right": s_isr
                        }
                        if j3d_arr is not None:
                            all_data[f_str][str(r_label)]["joints_3d"] = j3d_arr[t_idx].tolist()

    if len(sorted_kfs)>0:
        last_f = sorted_kfs[-1]
        last_dict = keyframe_map[last_f] or {}
        all_data[str(last_f)] = {}
        for r_label, val in last_dict.items():
            cam_t = val["camera_translation"]
            j3d = val["joints_3d"]
            is_r = val["is_right"]
            all_data[str(last_f)][str(r_label)] = {
                "camera_translation": cam_t.tolist(),
                "is_right": is_r
            }
            if j3d is not None:
                all_data[str(last_f)][str(r_label)]["joints_3d"] = j3d.tolist()

    import json
    with open(args.keypoints_output, 'w') as f:
        json.dump(all_data, f, indent=2)
    print(f"Saved => {args.keypoints_output}")

    from time import time
    total_elapsed = time() - total_start_time
    print(f"Done. total time={total_elapsed:.2f}s")


if __name__ == "__main__":
    main()
