import argparse
import cv2
import open3d
from tqdm import tqdm
from Utils.utils import *
from predSegmentation import *

def main():
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--DataPath', default='/home/ychen921/PointPainting/Data', 
                        help='Default:/home/ychen921/PointPainting/Data')
    Parser.add_argument('--SavePath', default='/home/ychen921/PointPainting/Results', 
                        help='Default:/home/ychen921/PointPainting/Results')
    Parser.add_argument('--CkptPath', default='/home/ychen921/PointPainting', 
                        help='Default:/home/ychen921/PointPainting')
    Parser.add_argument('--ParseData', type=int, default=0, 
                        help='Parse the raw point cloud data, Default:0')
    
    Args = Parser.parse_args()
    data_path = Args.DataPath
    save_path = Args.SavePath
    ckpt_path = Args.CkptPath
    parse_flag = Args.ParseData

    # Directory to segmentation network checkpoint
    ckpt_path = ckpt_path + '/best_deeplabv3plus_resnet101_cityscapes_os16.pth.tar'
    # Calculate the transformation from LiDAR to camera
    calib_path = data_path + '/calib.txt'
    liDAR2Cam = computeLiDAR2Cam(calib_path)

    # Create directory for saving
    img_save_dir = save_path+'/projected_clouds'
    pcd_save_dir = save_path+'/painted_clouds'
    make_directory(save_path)
    make_directory(img_save_dir)
    make_directory(pcd_save_dir)

    # Create directory for parsed 3D point cloud data
    pcd_src_dir = data_path + '/KITTI-360/data_3d_raw/2013_05_28_drive_0000_sync/velodyne_points/data'
    pcd_data_dir = data_path + '/pcd_data'
    if parse_flag == 1:
        make_directory(pcd_data_dir)
        convertAllBin2Pcd(pcd_src_dir, pcd_data_dir) # Parse point cloud data

    rgb_data_dir = data_path + '/KITTI-360/data_2d_raw/2013_05_28_drive_0000_sync/image_00/data_rect'

    img_list = os.listdir(rgb_data_dir)
    pcd_list = os.listdir(pcd_data_dir)
    img_list.sort()
    pcd_list.sort()
    
    img_paths = []
    pcd_paths = []
    for i in range(len(img_list)):
        img_paths.append(os.path.join(rgb_data_dir, img_list[i]))
        pcd_paths.append(os.path.join(pcd_data_dir, pcd_list[i]))

    # Save processed frames
    video_sequence = []
    for i in tqdm(range(500)):
        # Load image
        img = cv2.imread(img_paths[i])
        
        # Load corresponding pcd data and convert to np array
        pcd = open3d.io.read_point_cloud(pcd_paths[i])
        pcd_arr = np.asarray(pcd.points)

        # Remove points that behind the camera
        index = pcd_arr[:,0] >= 0
        pcd_arr = pcd_arr[index]
        pts_2d, depth, pts_3d_img = projectLiDAR2Image(P=liDAR2Cam, lidar_pts=pcd_arr, img_size=(img.shape[1], img.shape[0]))

        # semantic segmentation
        _, semantic_img = predict_segmentation(ckpt_path, img_paths[i])
        
        pcd_color = np.zeros((pts_3d_img.shape[0], 3), dtype=np.float32)
        img_pcd = img.copy()

        # Assign segmentation label to point cloud
        for j in range(pts_2d.shape[0]):
            if j >= 0:
                x, y = np.int32(pts_2d[j,0]), np.int32(pts_2d[j,1])
                class_color = np.float32(semantic_img[y, x])
                cv2.circle(img_pcd, (int(x), int(y)), 2, color=tuple(int(c) for c in class_color), thickness=1)
                pcd_color[j] = class_color / 255.0
        
        # Some issue with showing stacked image using imshow
        comb_img = np.vstack((img, semantic_img, img_pcd)).astype(np.uint8)
        
        # Visualize segmantic point cloud
        label_pcd = np.hstack((pts_3d_img[:,:3], pcd_color))
        pcd3dVisualize(label_pcd, i, pcd_save_dir)

        video_sequence.append(comb_img)

        # comb_img = Image.fromarray(comb_img)
        # comb_img.show()
        # time.sleep(0.5)
        # comb_img.close()
        cv2.imshow('LiDAR Mapping Test', comb_img)
        if cv2.waitKey(100) & 0xFF == 27:  # Press 'Esc' to break the loop
            break
    cv2.destroyAllWindows()

    # Save painted LiDAR points on camera images to video stream
    makeVideo(save_path=save_path+'/Output.avi', frames=video_sequence)

if __name__ == '__main__':
    main()