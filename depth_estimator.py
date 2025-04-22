from utils.stereo_rectify import StereoRectifier
from RAFT.core.raft import RAFT
from argparse import ArgumentParser, Action
from torchvision.transforms import Resize, InterpolationMode
from collections import OrderedDict
import os
import numpy as np
import torch
import cv2
import shutil
from tqdm import tqdm
import argparse

RAFT_config = {
    "pretrained": "RAFT/pretrained/raft-things.pth",
    "iters": 12,
    "dropout": 0.0,
    "small": False,
    "pose_scale": 1.0,
    "lbgfs_iters": 100,
    "use_weights": True,
    "dbg": False
}

def check_arg_limits(arg_name, n):
    class CheckArgLimits(Action):
        def __call__(self, parser, args, values, option_string=None):
            if len(values) > n:
                parser.error("Too many arguments for " + arg_name + ". Maximum is {0}.".format(n))
            if len(values) < n:
                parser.error("Too few arguments for " + arg_name + ". Minimum is {0}.".format(n))
            setattr(args, self.dest, values)
    return CheckArgLimits

def read_mask(path):
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    mask = mask > 0
    mask = torch.from_numpy(mask).unsqueeze(0)
    return mask


class DepthEstimator(torch.nn.Module):
    def __init__(self, config):
        super(DepthEstimator, self).__init__()
        self.model = RAFT(config).to('cuda')
        self.model.freeze_bn()
        new_state_dict = OrderedDict()
        raft_ckp = config['pretrained']
        try:
            state_dict = torch.load(raft_ckp)
        except RuntimeError:
            state_dict = torch.load(raft_ckp, map_location='cpu')
        for k, v in state_dict.items():
            name = k.replace('module.','')  # remove `module.`
            new_state_dict[name] = v
        self.model.load_state_dict(new_state_dict)
        
    def forward(self, imagel, imager, baseline, upsample=True): # intrinsics,
        n, _, h, w = imagel.shape
        flow = self.model(imagel.to('cuda'), imager.to('cuda'), upsample=upsample)[0][-1]
        baseline = torch.from_numpy(baseline).to('cuda')
        # focal = torch.from_numpy(intrinsics[0,0][None]).to('cuda')
        # cx = torch.from_numpy(intrinsics[0,2][None]).to('cuda')
        # depth = (baseline[:, None, None] * focal) / (-flow[:, 0] + cx)
        depth = (baseline[:, None, None]) / (-flow[:, 0])
        if not upsample:
            depth/= 8.0  # factor 8 of upsampling
        # valid = (depth > 0) & (depth <= 600.0)
        # depth[~valid] = 0.0
        return depth.unsqueeze(1)
        
def reformat_dataset(data_dir, calib_file, img_size): # img_size - multiple of 8
    """
    Reformat the StereoMIS to the same format as EndoNeRF dataset by stereo depth estimation.
    """
    # Load parameters after rectification
    assert os.path.exists(calib_file), "Calibration file not found."
    # rect = StereoRectifier(calib_file, img_size_new=(img_size[1], img_size[0]), mode='conventional')
    rect = StereoRectifier(calib_file, img_size_new=None, mode='conventional')
    calib = rect.get_rectified_calib()
    baseline = calib['bf'].astype(np.float32) # here the baseline this is "baseline * focal length"
    intrinsics = calib['intrinsics']['left'].astype(np.float32)

    # Sort images and masks according to the start and end frame indexes
    frames = sorted(os.listdir(os.path.join(data_dir, 'left_frames')), key=lambda x: int(x.split('.')[0][5:]))
    assert len(frames) > 0, "No frames found."
    resize = Resize(img_size)
    
    # Configurate depth estimator. We follow the settings of RAFT in robust-pose-estimator(https://github.com/aimi-lab/robust-pose-estimator)
    depth_estimator = DepthEstimator(RAFT_config)

    # Create folders
    output_dir = os.path.join(data_dir, 'stereo')
    left_img_resized_dir = os.path.join(output_dir, 'left_frames')
    depth_dir = os.path.join(output_dir, 'depth')
    depth_vis_dir = os.path.join(output_dir, 'depth_vis')

    if not os.path.exists(left_img_resized_dir):
        os.makedirs(left_img_resized_dir)
    if not os.path.exists(depth_dir):
        os.makedirs(depth_dir)
    if not os.path.exists(depth_vis_dir):
        os.makedirs(depth_vis_dir)

    for i, frame in tqdm(enumerate(frames)):
        left_img = torch.from_numpy(cv2.cvtColor(cv2.imread(os.path.join(data_dir, 'left_frames', frame)), cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float()
        right_img = torch.from_numpy(cv2.cvtColor(cv2.imread(os.path.join(data_dir, 'right_frames', frame)), cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float()

        # original size: 1400x986
        w_orig, h_orig = left_img.shape[2], left_img.shape[1]

        # Rectify the images
        left_img, right_img = rect(left_img, right_img)
        # import matplotlib.pyplot as plt
        # plt.imshow(left_img.permute(1, 2, 0).numpy().astype(np.uint8))
        # plt.show()
        # breakpoint()

        # Resize the images
        left_img = resize(left_img)
        right_img = resize(right_img)

        # new size: 640x512
        w_new, h_new = left_img.shape[2], left_img.shape[1]

        scale = w_new/w_orig # scale = x_new / x_orig
        
        with torch.no_grad():
            depth = depth_estimator(left_img[None], right_img[None], baseline[None] * scale,)

            # check the first depth map
            if i == 0:
                import matplotlib.pyplot as plt
                plt.imshow(depth[0, 0].cpu().numpy())
                plt.show()
                breakpoint()

        # Save the data. 
        left_img_np = left_img.permute(1, 2, 0).numpy()
        left_img_bgr = cv2.cvtColor(left_img_np, cv2.COLOR_RGB2BGR)
 
        name_left_img = 'frame-'+str(i).zfill(6)+'.color.png'
        name_depth = 'frame-'+str(i).zfill(6)+'.depth.npy'
        name_depth_vis = 'frame-'+str(i).zfill(6)+'.depth_vis.png'

        cv2.imwrite(os.path.join(left_img_resized_dir, name_left_img), left_img_bgr)
        depth = depth[0, 0].cpu().numpy()
        np.save(os.path.join(depth_dir, name_depth), depth)
        # np.save(os.path.join(depth_dir, name_depth), depth[0, 0].cpu().numpy())

        import matplotlib.pyplot as plt
        # plt.imshow(depth[0, 0].cpu().numpy())

        plt.imshow(np.load(os.path.join(depth_dir, name_depth)))
        plt.axis('off')
        plt.savefig(os.path.join(depth_vis_dir, name_depth_vis), bbox_inches='tight', pad_inches=0) 
        # plt.show()
        # breakpoint()
    
if __name__ == "__main__":
    torch.manual_seed(1234)
    np.random.seed(1234)
    # Set up command line argument parser
    parser = ArgumentParser(description="parameters for dataset format conversions")
    parser.add_argument('--data_dir', '-d', type=str, required=True, help='Path to the dataset directory')
    parser.add_argument('--img_size', '-s', type=int, nargs=2, default=(512, 640), action=check_arg_limits('img_size', 2))
    parser.add_argument('--calib_file', '-c', type=str, required=True, help='Path to the calibration file')
    args = parser.parse_args()
    reformat_dataset(args.data_dir, args.calib_file, args.img_size)