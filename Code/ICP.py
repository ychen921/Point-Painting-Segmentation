import os
import cv2
import open3d
import argparse
import numpy as np
from tqdm import tqdm


def P2PICP(src, dst, threshold, index):
    curr_transformation = np.array([
        [1, 0, 0, index],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]])    
    
    result = open3d.pipelines.registration.registration_icp(src, dst, threshold, curr_transformation,
                                                             open3d.pipelines.registration.TransformationEstimationPointToPoint())
    return result


def main():
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--BasePath', default='/home/ychen921/PointPainting//Results/painted_clouds/', 
                        help='Default:/home/ychen921/PointPainting/Results/painted_clouds/')
    Parser.add_argument('--SavePath', default='/home/ychen921/PointPainting/Results', 
                        help='Default:/home/ychen921/PointPainting/Results')
    
    Args = Parser.parse_args()
    pcd_path = Args.BasePath
    save_path = Args.SavePath

    pcd_data = os.listdir(pcd_path)
    pcd_data.sort()
    
    dst_pcd = open3d.io.read_point_cloud(pcd_path+pcd_data[0])
    for i in tqdm(range(1, len(pcd_data))):
        src_pcd = open3d.io.read_point_cloud(pcd_path+pcd_data[i])
        
        icp_result = P2PICP(src=src_pcd, dst=dst_pcd, threshold=5, index=i)
        print(icp_result)

        # Transform the src pcd
        src_pcd = src_pcd.transform(icp_result.transformation)

        # Combine with dst pcd
        dst_pcd += src_pcd

    open3d.visualization.draw_geometries([dst_pcd])
    open3d.io.write_point_cloud(save_path + "/final_pcd.pcd", dst_pcd)

if __name__ == "__main__":
    main()