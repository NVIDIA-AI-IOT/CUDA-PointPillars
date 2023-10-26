import os
import pickle
import numpy as np
import kitti_util as utils

CLASS_NAMES = ['Car', 'Pedestrian', 'Cyclist']
PI = 3.1415926

def get_lidar_in_image_fov(
    pc_velo, calib, xmin, ymin, xmax, ymax, return_more=False, clip_distance=2.0
):
    """ Filter lidar points, keep those in image FOV """
    pts_2d = calib.project_velo_to_image(pc_velo)
    fov_inds = (
        (pts_2d[:, 0] < xmax)
        & (pts_2d[:, 0] >= xmin)
        & (pts_2d[:, 1] < ymax)
        & (pts_2d[:, 1] >= ymin)
    )
    fov_inds = fov_inds & (pc_velo[:, 0] > clip_distance)
    imgfov_pc_velo = pc_velo[fov_inds, :]
    if return_more:
        return imgfov_pc_velo, pts_2d, fov_inds
    else:
        return imgfov_pc_velo


def project_to_image(pts_3d, P):
    """ Project 3d points to image plane.

    Usage: pts_2d = projectToImage(pts_3d, P)
      input: pts_3d: nx3 matrix
             P:      3x4 projection matrix
      output: pts_2d: nx2 matrix

      P(3x4) dot pts_3d_extended(4xn) = projected_pts_2d(3xn)
      => normalize projected_pts_2d(2xn)

      <=> pts_3d_extended(nx4) dot P'(4x3) = projected_pts_2d(nx3)
          => normalize projected_pts_2d(nx2)
    """
    n = pts_3d.shape[0]
    pts_3d_extend = np.hstack((pts_3d, np.ones((n, 1))))
    # print(('pts_3d_extend shape: ', pts_3d_extend.shape))
    pts_2d = np.dot(pts_3d_extend, np.transpose(P))  # nx3
    pts_2d[:, 0] /= pts_2d[:, 2]
    pts_2d[:, 1] /= pts_2d[:, 2]
    return pts_2d[:, 0:2]

def roty(t):
    """ Rotation about the y-axis. """
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

def compute_box_3d(ry, l, w, h, t, P):
    """ Takes an object and a projection matrix (P) and projects the 3d
        bounding box into the image plane.
        Returns:
            corners_2d: (8,2) array in left image coord.
            corners_3d: (8,3) array in in rect camera coord.
    """
    # compute rotational matrix around yaw axis
    R = roty(ry)

    # 3d bounding box corners
    x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
    y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
    z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]

    # rotate and translate 3d bounding box
    corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
    # print corners_3d.shape
    corners_3d[0, :] = corners_3d[0, :] + t[0]
    corners_3d[1, :] = corners_3d[1, :] + t[1]
    corners_3d[2, :] = corners_3d[2, :] + t[2]
    # print 'cornsers_3d: ', corners_3d
    # only draw 3d bounding box for objs in front of the camera
    if np.any(corners_3d[2, :] < 0.1):
        corners_2d = None
        return corners_2d #, np.transpose(corners_3d)

    # project the 3d bounding box into the image plane
    corners_2d = project_to_image(np.transpose(corners_3d), P)
    # print 'corners_2d: ', corners_2d

    return corners_2d #, np.transpose(corners_3d)

def _read_imageset_file(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    return [int(line) for line in lines]

def createNVOutput():
    val_image_ids = _read_imageset_file('tool/eval/val.txt')
    for image_i in val_image_ids:
        str_i = str(image_i).zfill(6)

        calib_path = 'data/kitti/training/calib/' + str_i + '.txt'
        image_2_path = 'data/kitti/training/image_2/' + str_i + '.png'
        velo_det_path = 'data/kitti/pred_velo/' + str_i + '.txt'
        cam_det_path = 'data/kitti/pred/' + str_i + '.txt'

        calib = utils.Calibration(calib_path)
        image = utils.load_image(image_2_path)
        img_height, img_width, _ = image.shape

        print(str_i, " ", img_width, " ", img_height)

        # detection in velodyne coordinate - x, y, z, w, l, h, rt, id, score
        velo_det = np.loadtxt(velo_det_path).reshape(-1,9)
        img_boxs = calib.project_velo_to_rect(velo_det[:,:3])

        with open(cam_det_path, 'w') as the_file:
            for i in range(0, img_boxs.shape[0]):

                w = velo_det[i][3]
                l = velo_det[i][4]
                h = velo_det[i][5]

                x = img_boxs[i][0]
                y = img_boxs[i][1] + h/2
                z = img_boxs[i][2]

                ry = - velo_det[i][6] - 0.5 * PI

                alpha = -np.arctan2(x, z) + ry

                corners_2d = compute_box_3d(ry, w, l, h, [x, y, z], calib.P)

                if corners_2d is None:
                    continue

                xy_min = np.amin(corners_2d, axis=0)
                xy_max = np.amax(corners_2d, axis=0)

                if (xy_min[0] < 0 or xy_min[0] >= img_width) and \
                    (xy_min[1] < 0 or xy_min[1] >= img_height) and \
                    (xy_max[0] < 0 or xy_max[0] >= img_width) and \
                    (xy_max[1] < 0 or xy_max[1] >= img_height):
                    continue

                if xy_min[0] < 0:
                    xy_min[0] = 0
                elif xy_min[0] >= img_width:
                    xy_min[0] = img_width - 1

                if xy_min[1] < 0:
                    xy_min[1] = 0
                elif xy_min[1] >= img_height:
                    xy_min[1] = img_height - 1

                if xy_max[0] < 0:
                    xy_max[0] = 0
                elif xy_max[0] >= img_width:
                    xy_max[0] = img_width - 1

                if xy_max[1] < 0:
                    xy_max[1] = 0
                elif xy_max[1] >= img_height:
                    xy_max[1] = img_height - 1

                if xy_min[0] == xy_max[0] or xy_min[1] == xy_max[1]:
                    continue

                the_file.write(CLASS_NAMES[int(velo_det[i][7])])    # type
                the_file.write(" 0.00")                             # truncated
                the_file.write(" 0")                                # occluded
                the_file.write(f" {alpha:.2f}")                     # alpha

                the_file.write(f" {xy_min[0]:.2f} {xy_min[1]:.2f} {xy_max[0]:.2f} {xy_max[1]:.2f} ")    # bbox - xmin，ymin，xmax，ymax
                the_file.write(f"{h:.2f} {l:.2f} {w:.2f} ")                                             # dimensions, h l w
                the_file.write(f"{x:.2f} {y:.2f} {z:.2f} ")                                             # location, x y z

                the_file.write(f"{ry:.2f} ")                        # rotation ry around Y-axis in camera coordinates
                the_file.write(f"{velo_det[i][8]:.2f}")             # score
                the_file.write("\n")

def createPCDetOutput():
    with open("tool/eval/pcdet.pkl", "rb") as f:
        dt_annos = pickle.load(f)
        for txt_i in range(0, len(dt_annos)):
            frame = dt_annos[txt_i]
            print(frame['frame_id'])
            cam_det_path = 'data/kitti/pcdet/' + frame['frame_id'] + '.txt'
            with open(cam_det_path, 'w') as the_file:
                for i in range(0, len(frame['name'])):
                    the_file.write(frame['name'][i] + " ")
                    the_file.write(f"{frame['truncated'][i]:.2f} ")
                    the_file.write(str(int(frame['occluded'][i])) + " ")
                    the_file.write(f"{frame['alpha'][i]:.2f} ")
                    the_file.write(f"{frame['bbox'][i][0]:.2f} {frame['bbox'][i][1]:.2f} {frame['bbox'][i][2]:.2f} {frame['bbox'][i][3]:.2f} ")
                    the_file.write(f"{frame['dimensions'][i][1]:.2f} {frame['dimensions'][i][2]:.2f} {frame['dimensions'][i][0]:.2f} ")
                    the_file.write(f"{frame['location'][i][0]:.2f} {frame['location'][i][1]:.2f} {frame['location'][i][2]:.2f} ")
                    the_file.write(f"{frame['rotation_y'][i]:.2f} ")
                    the_file.write(f"{frame['score'][i]:.2f}")
                    the_file.write("\n")

if __name__ == "__main__":
    createPCDetOutput()
    createNVOutput()