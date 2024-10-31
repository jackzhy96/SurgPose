import torch
from sam2.build_sam import build_sam2_video_predictor

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import yaml

import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--video_dir", type=str, default="./data/left_frames")
parser.add_argument("--kps_dir", type=str, default="./kps_annotation")
parser.add_argument("--save_vis", action="store_true")
parser.add_argument("--res_vis_dir", type=str, default="./results")

args = parser.parse_args()

ann_config = {
    "psm": "psm1",
    "traj_idx": 000000,
    "left_or_right": "left",
}
traj_idx = ann_config["traj_idx"]
left_or_right = ann_config["left_or_right"]
psm = ann_config["psm"]

# use bfloat16 for the entire notebook
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

from sam2.build_sam import build_sam2_video_predictor

sam2_checkpoint = "./checkpoints/sam2.1_hiera_large.pt" # sam2_checkpoint = "../checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"        # model_cfg = "sam2_hiera_l.yaml"

predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)

def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   

# scan all the JPEG frame names in this directory
frame_names = [
    p for p in os.listdir(args.video_dir)
    # if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG", 'png']
]
frame_names.sort(key=lambda p: int(os.path.splitext(p)[0][5:]))
# take a look the first video frame
frame_idx = 0
plt.figure(figsize=(12, 8))
plt.title(f"frame {frame_idx}")
plt.imshow(Image.open(os.path.join(args.video_dir, frame_names[frame_idx])))
plt.show()
breakpoint()

inference_state = predictor.init_state(video_path=args.video_dir)

predictor.reset_state(inference_state)

prompts = {}  # hold all the clicks we add for visualization

## PSM1
ann_frame_idx = 0  # the frame index we interact with
ann_obj_id = 1  # psm1 shaft
points = np.array([[139, 413], [116, 425]], dtype=np.float32) #np.array([[1223, 542], [1204, 537], [1221, 513], [1253, 542], [1221, 555]], dtype=np.float32)
labels = np.array([1, 0], np.int32)
prompts[ann_obj_id] = points, labels
_, out_obj_ids, out_mask_logits = predictor.add_new_points(inference_state=inference_state, frame_idx=ann_frame_idx, obj_id=ann_obj_id, points=points, labels=labels,)

# ann_frame_idx = 0  # the frame index we interact with
# ann_obj_id = 2  # psm1 wrist
# points = np.array([[1230, 635], [1128, 701], [1032, 763]], dtype=np.float32)
# labels = np.array([1, 1, 1], np.int32)
# prompts[ann_obj_id] = points, labels
# _, out_obj_ids, out_mask_logits = predictor.add_new_points(inference_state=inference_state, frame_idx=ann_frame_idx, obj_id=ann_obj_id, points=points, labels=labels,)

# ann_frame_idx = 0  # the frame index we interact with
# ann_obj_id = 3  # psm1 joint 1
# points = np.array([[874, 667], [903, 659], [941, 637]], dtype=np.float32)
# labels = np.array([1, 1, 0], np.int32)
# prompts[ann_obj_id] = points, labels
# _, out_obj_ids, out_mask_logits = predictor.add_new_points(inference_state=inference_state, frame_idx=ann_frame_idx, obj_id=ann_obj_id, points=points, labels=labels,)

# ann_frame_idx = 0  # the frame index we interact with
# ann_obj_id = 4  # psm1 left tool tip
# points = np.array([[861, 643], [901, 632], [924, 630], [901, 660], [860, 673]], dtype=np.float32)
# labels = np.array([1, 1, 0, 0, 0], np.int32)
# prompts[ann_obj_id] = points, labels
# _, out_obj_ids, out_mask_logits = predictor.add_new_points(inference_state=inference_state, frame_idx=ann_frame_idx, obj_id=ann_obj_id, points=points, labels=labels,)

# ann_frame_idx = 0  # the frame index we interact with
# ann_obj_id = 5  # psm1 right tool tip
# points = np.array([[1184, 509]], dtype=np.float32) # points = np.array([[680, 450], [666, 455]], dtype=np.float32)
# labels = np.array([1], np.int32)
# prompts[ann_obj_id] = points, labels
# _, out_obj_ids, out_mask_logits = predictor.add_new_points(inference_state=inference_state, frame_idx=ann_frame_idx, obj_id=ann_obj_id, points=points, labels=labels,)

# ann_frame_idx = 0  # the frame index we interact with
# ann_obj_id = 6  # psm1 joint 2
# points = np.array([[1313, 536], [1313, 558]], dtype=np.float32)
# labels = np.array([1, 0], np.int32)
# prompts[ann_obj_id] = points, labels
# _, out_obj_ids, out_mask_logits = predictor.add_new_points(inference_state=inference_state, frame_idx=ann_frame_idx, obj_id=ann_obj_id, points=points, labels=labels,)

# ann_frame_idx = 0 # the frame index we interact with
# ann_obj_id = 7  # psm1 joint 3
# points = np.array([[1114, 502]], dtype=np.float32)
# labels = np.array([1], np.int32)
# prompts[ann_obj_id] = points, labels
# _, out_obj_ids, out_mask_logits = predictor.add_new_points(inference_state=inference_state, frame_idx=ann_frame_idx, obj_id=ann_obj_id, points=points, labels=labels,)

# PSM3
# ann_frame_idx = 560  # the frame index we interact with
# ann_obj_id = 8  # psm3 shaft
# points = np.array([[378, 670], [285, 624], [423, 700]], dtype=np.float32)
# labels = np.array([1, 1, 0], np.int32)
# prompts[ann_obj_id] = points, labels
# _, out_obj_ids, out_mask_logits = predictor.add_new_points(inference_state=inference_state, frame_idx=ann_frame_idx, obj_id=ann_obj_id, points=points, labels=labels,)

# ann_frame_idx = 560  # the frame index we interact with
# ann_obj_id = 9 # psm3 wrist
# points = np.array([[423, 700], [447, 711], [442, 724], [378, 670], [455, 734], [450, 755]], dtype=np.float32)
# labels = np.array([1,1, 0, 0, 0, 0], np.int32)
# prompts[ann_obj_id] = points, labels
# _, out_obj_ids, out_mask_logits = predictor.add_new_points(inference_state=inference_state, frame_idx=ann_frame_idx, obj_id=ann_obj_id, points=points, labels=labels,)

# ann_frame_idx = 560  # the frame index we interact with
# ann_obj_id = 10 # psm3 joint 1  
# points = np.array([[455, 734], [450, 755], [423, 700], [440, 735]], dtype=np.float32)
# labels = np.array([1, 0, 0, 0], np.int32)
# prompts[ann_obj_id] = points, labels
# _, out_obj_ids, out_mask_logits = predictor.add_new_points(inference_state=inference_state, frame_idx=ann_frame_idx,obj_id=ann_obj_id,points=points,labels=labels,)

# ann_frame_idx = 560  # the frame index we interact with
# ann_obj_id = 11 # psm3 left tool tip 
# points = np.array([[450, 755], [455, 734], [423, 700]], dtype=np.float32) 
# labels = np.array([1, 0, 0], np.int32)
# prompts[ann_obj_id] = points, labels
# _, out_obj_ids, out_mask_logits = predictor.add_new_points(inference_state=inference_state, frame_idx=ann_frame_idx, obj_id=ann_obj_id, points=points, labels=labels,)

# ann_frame_idx = 0  # the frame index we interact with
# ann_obj_id = 12 # psm3 right tool tip
# points = np.array([[586, 712]], dtype=np.float32)
# labels = np.array([1], np.int32)
# prompts[ann_obj_id] = points, labels
# _, out_obj_ids, out_mask_logits = predictor.add_new_points(inference_state=inference_state, frame_idx=ann_frame_idx, obj_id=ann_obj_id, points=points, labels=labels,)

# ann_frame_idx = 0  # the frame index we interact with
# ann_obj_id = 13 # psm3 joint 2
# points = np.array([[499, 670], [474, 675]], dtype=np.float32)
# labels = np.array([1, 0], np.int32)
# prompts[ann_obj_id] = points, labels
# _, out_obj_ids, out_mask_logits = predictor.add_new_points(inference_state=inference_state, frame_idx=ann_frame_idx, obj_id=ann_obj_id, points=points, labels=labels,)

# ann_frame_idx = 0  # the frame index we interact with
# ann_obj_id = 14 # psm3 joint 3
# points = np.array([[474, 675], [499, 670]], dtype=np.float32) 
# labels = np.array([1, 0], np.int32)
# prompts[ann_obj_id] = points, labels
# _, out_obj_ids, out_mask_logits = predictor.add_new_points(inference_state=inference_state, frame_idx=ann_frame_idx, obj_id=ann_obj_id, points=points, labels=labels,)

# show the results on the current (interacted) frame on all objects
ann_frame_idx = 0
plt.figure(figsize=(12, 8))
plt.title(f"frame {ann_frame_idx}")
plt.imshow(Image.open(os.path.join(args.video_dir, frame_names[ann_frame_idx])))
# show_points(points, labels, plt.gca())
for i, out_obj_id in enumerate(out_obj_ids):
    # show_points(*prompts[out_obj_id], plt.gca())
    show_mask((out_mask_logits[i] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_id)
plt.show()
breakpoint()

# run propagation throughout the video and collect the results in a dict
video_segments = {}  # video_segments contains the per-frame segmentation results
video_keypoints = {}  # video_keypoints contains the per-frame keypoints results
for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
    video_segments[out_frame_idx] = {
        out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
        for i, out_obj_id in enumerate(out_obj_ids)
    }

    temp_keypoints_list = []
    # if out_frame_idx > 505:
    #     break
    for i, out_obj_id in enumerate(out_obj_ids):
        # segmentation
        mask = (video_segments[out_frame_idx][out_obj_id] * 255).squeeze().astype(np.uint8)
        M = cv2.moments(mask)
        if M["m00"] == 0:
            temp_keypoints_list.append(None)
            continue
        cX = float(M["m10"] / M["m00"])
        cY = float(M["m01"] / M["m00"])
        temp_keypoints_list.append([cX, cY])
        # plt.plot(cX, cY, marker='o', color="red") 
        # plt.imshow(mask) 
        # plt.show() 

    video_keypoints[out_frame_idx] = {
        out_obj_id: temp_keypoints_list[i]
        for i, out_obj_id in enumerate(out_obj_ids)
    }

os.makedirs(args.kps_dir, exist_ok=True)

with open(os.path.join(args.kps_dir, f"keypoints_{traj_idx}_{left_or_right}_{psm}.yaml"), "w") as f: 
    yaml.dump(video_keypoints, f)

if args.save_vis:
    # render the segmentation results every few frames
    vis_frame_stride = 1
    os.makedirs(args.res_vis_dir, exist_ok=True)
    plt.close("all")
    for out_frame_idx in range(0, len(frame_names), vis_frame_stride):
        plt.figure(figsize=(6, 4))
        plt.imshow(Image.open(os.path.join(args.video_dir, frame_names[out_frame_idx])))
        for out_obj_id, out_mask in video_segments[out_frame_idx].items():
            show_mask(out_mask, plt.gca(), obj_id=out_obj_id)
        # save the visualization to disk
        plt.axis('off')
        # plt.savefig(f"./results/frame_{out_frame_idx:04d}.png", bbox_inches='tight', pad_inches=0)
        plt.savefig(os.path.join(args.res_vis_dir, f"frame_{out_frame_idx:04d}.png"), bbox_inches='tight', pad_inches=0)
        plt.close()
