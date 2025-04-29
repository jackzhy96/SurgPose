import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import argparse

depth_data = None

def load_depth_file(depth_dir:str, idx_frame:int)->None:
    '''
    depth_dir : directory of the parent folder
    idx_frame : index of the selected frame
    Load the depth file
    '''
    global depth_data
    depth_path = os.path.join(depth_dir, 'depth', f'frame-{str(idx_frame).zfill(6)}.depth.npy')
    depth_data = np.load(depth_path)



def custom_coord(x: float, y: float) -> str:
    '''
    x : x coordinate
    y : y coordinate
    depth_map: given depth map
    redefine the matplot coordinate formate
    '''
    x = int(x)
    y = int(y)
    if 0 <= y < depth_data.shape[0] and 0 <= x < depth_data.shape[1]:
        z = depth_data[y, x]
        return f'x={x}, y={y}, depth={z: .2f}mm'
    else:
        return f'x={x}, y={y}'


def plot_depth_map(idx_frame:int)->None:
    '''
    idx_frame : index of the selected frame
    Plot the depth map
    '''
    fig, ax = plt.subplots(figsize=(12, 8))
    img = ax.imshow(depth_data, cmap='plasma', vmin=np.percentile(depth_data, 1), vmax=np.percentile(depth_data, 99))
    plt.colorbar(img, label='Depth (mm)')
    ax.set_title(f'Interactive Depth Viewer Frame {idx_frame}')
    ax.axis('off')
    ax.format_coord = custom_coord
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Interactive Depth Viewer")
    parser.add_argument("--depth_dir", type=str, help="Path of the stereo matching generated folder")
    parser.add_argument("--cp_file", type=str, help="Path of the dVRK API cp data file")
    parser.add_argument("--idx_frame", type=str, help="frame index")
    args = parser.parse_args()

    depth_dir = args.depth_dir
    cp_file = args.cp_file
    idx_frame = int(args.idx_frame)

    try:
        load_depth_file(depth_dir, idx_frame)
        plot_depth_map(idx_frame)
    except Exception as e:
        print(e)

