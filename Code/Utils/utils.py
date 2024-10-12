import os
import sys
import cv2
# import imageio
from PIL import Image
import shutil
import open3d
import struct
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def loadCalibData(file_path):
    calib_data = {}
    with open(file_path, 'r') as file:
        for row in file:
            key, item = row.split(':')
            vals = np.array([float(x) for x in item.split()])

            if key == "P":
                calib_data[key] = vals.reshape((3,4))
            elif key == "K":
                calib_data[key] = vals.reshape((3,3))
            elif key == "R0":
                calib_data[key] = vals.reshape((3,3))
            elif key == "Tr_cam_to_lidar":
                calib_data[key] = vals.reshape((3,4))
            elif key == "D":
                calib_data[key] = vals.reshape((1,5))

    return calib_data["P"], calib_data["K"], calib_data["R0"], calib_data["Tr_cam_to_lidar"], calib_data["D"]

def computeLiDAR2Cam(file_path):
    P, K, R, Tr_cam_to_lidar, D = loadCalibData(file_path)

    R_cam_to_lidar = Tr_cam_to_lidar[:3,:3].reshape(3,3)
    T_cam_to_lidar = Tr_cam_to_lidar[:3,3].reshape(3,1)

    R_cam_to_lidar_inv = np.linalg.inv(R_cam_to_lidar)
    t_new = -np.dot(R_cam_to_lidar_inv , T_cam_to_lidar)

    Tr_lidar_to_cam = np.vstack((np.hstack((R_cam_to_lidar_inv, t_new)), np.array([0., 0., 0., 1.])))
    R_rect = np.eye(4)
    R_rect[:3, :3] = R.reshape(3, 3)

    P_ = P.reshape((3, 4))

    proj_mat = P_ @ R_rect  @ Tr_lidar_to_cam
    return proj_mat
            
def make_directory(dir_path):

    # Check if directory exists
    if os.path.exists(dir_path):
        # Remove the existing directory
        shutil.rmtree(dir_path)

    os.makedirs(dir_path)

def bin2pcd(bin_path, pcd_path):
    size_float = 4
    pcd_list = []
    with open(bin_path, "rb") as f:
        byte = f.read(size_float * 4)
        while byte:
            x, y, z, intensity = struct.unpack("ffff", byte)
            pcd_list.append([x, y, z])
            byte = f.read(size_float * 4)

    np_pcd = np.asarray(pcd_list)
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(np_pcd)
    open3d.io.write_point_cloud(pcd_path, pcd)

def convertAllBin2Pcd(bins_path, pcd_path):
    bins_list = os.listdir(bins_path)
    print("Converting all bin files to pcd files")
    for i in tqdm(range(len(bins_list))):
        bin_path = os.path.join(bins_path, bins_list[i])
        bin2pcd(bin_path, pcd_path+"/"+str(i)+".pcd")

def inview_point_index(pts, img_size):
    return ((pts[:, 0] >= 0) & (pts[:, 0] < img_size[0]) & (pts[:, 1] >= 0) & (pts[:, 1] < img_size[1]))

def projectLiDAR2Image(P, lidar_pts, img_size):
    # Number of lidar points
    num_pts = lidar_pts.shape[0]
    pts_3d = np.hstack((lidar_pts, np.ones((num_pts, 1))))
    pts_2d = np.dot(pts_3d, P.T)

    depth = pts_3d[:, 2]
    depth[depth==0] = -1e-6

    # Normalize points
    pts_2d[:,0] /= pts_2d[:,2]
    pts_2d[:,1] /= pts_2d[:,2]
    pts_2d = pts_2d[:,:2]

    inview_idx = inview_point_index(pts_2d, img_size)

    return pts_2d[inview_idx], depth[inview_idx], lidar_pts[inview_idx]

def makeVideo(save_path, frames):
    # with imageio.get_writer(save_path, fps=30) as writer:
    #     for frame in frames:
    #         writer.append(frame)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(save_path, fourcc, 15.0, (frames[0].shape[1], frames[0].shape[0]))

    for frame in frames: 
        out.write(frame)

    out.release()

def pcd3dVisualize(pointcloud, f_num, save_path):
    # pcd coordinate and sengmentic label
    coords = pointcloud[:, :3]
    labels = pointcloud[:, 3:]

    # #Initialize Open3D visualizer
    visualizer = open3d.visualization.Visualizer()
    pcd = open3d.geometry.PointCloud()
    visualizer.add_geometry(pcd)

    pcd.points = open3d.utility.Vector3dVector(coords)
    pcd.colors = open3d.utility.Vector3dVector(labels)

    open3d.io.write_point_cloud(save_path + "/" + str(f_num) + ".pcd",pcd)

