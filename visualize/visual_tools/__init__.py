import open3d as o3d
import numpy as np


from .open3d_coordinate import create_coordinate
from .open3d_arrow import create_arrow
from .open3d_box import create_box


def create_box_with_arrow(box, color=None):
    """
    box: list(8) [ x, y, z, dx, dy, dz, yaw]
    """

    box_o3d = create_box(box, color)
    x = box[0]
    y = box[1]
    z = box[2]
    l = box[3]
    yaw = box[6]
    # get direction arrow
    dir_x = l / 2.0 * np.cos(yaw)
    dir_y = l / 2.0 * np.sin(yaw)

    arrow_origin = [x - dir_x, y - dir_y, z]
    arrow_end = [x + dir_x, y + dir_y, z]
    arrow = create_arrow(arrow_origin, arrow_end, color)

    return box_o3d, arrow


def draw_clouds_with_boxes(cloud , boxes):
    """
    cloud: (N, 4)  [x, y, z, intensity]
    boxes: (n,7) np.array = n*7  ( x, y, z, dx, dy, dz, yaw) 
    """
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    # --------------------------------------------------------------
    # create point cloud
    # --------------------------------------------------------------
    points_color = [[0.5, 0.5, 0.5]]  * cloud.shape[0]
    # print(np.unique(cloud[:, 3]))
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(cloud[:,:3])
    pc.colors = o3d.utility.Vector3dVector(points_color)
    vis.add_geometry(pc)

    # --------------------------------------------------------------
    # create boxes with colors with arrow
    # --------------------------------------------------------------
    boxes_o3d = []

    cur_box_color = [1, 0, 0]

    # create boxes
    for box in boxes:
        box_o3d, arrow = create_box_with_arrow(box, cur_box_color)
        boxes_o3d.append(box_o3d)
        boxes_o3d.append(arrow)
    # add_geometry fro boxes
    [vis.add_geometry(element) for element in boxes_o3d]

    # --------------------------------------------------------------
    # coordinate frame
    # --------------------------------------------------------------
    coordinate_frame = create_coordinate(size=2.0, origin=[0, 0, 0])
    vis.add_geometry(coordinate_frame)

    # --------------------------------------------------------------
    # drop the window
    # --------------------------------------------------------------
    vis.get_render_option().point_size = 2
    vis.run()
    vis.destroy_window()
