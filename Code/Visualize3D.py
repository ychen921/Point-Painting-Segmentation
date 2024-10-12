import open3d as o3d
import numpy as np
import imageio
import time
import os
from tqdm import tqdm

if __name__ == '__main__':
    pcd_path = '/home/ychen921/PointPainting/Results/painted_clouds/'
    pcd_data = os.listdir(pcd_path)
    pcd_data.sort()
    
    pcd = o3d.io.read_point_cloud(pcd_path + pcd_data[185])
    
    o3d.visualization.draw_geometries([pcd])