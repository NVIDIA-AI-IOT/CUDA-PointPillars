import numpy as np
import open3d as o3d


def rotz(t):
    """ Rotation about the z-axis. """
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def box2corners(box):
    """
    box: [x, y, z, dx, dy, dz, yaw]
    """

    # 8 corners: np.array = n*8*3(x, y, z)
    #         7 -------- 6
    #        /|         /|
    #       4 -------- 5 .
    #       | |        | |
    #       . 3 -------- 2
    #       |/         |/
    #       0 -------- 1

    #             ^ dx(l)
    #             |
    #             |
    #             |
    # dy(w)       |
    # <-----------O

    x = box[0]
    y = box[1]
    z = box[2]

    l = box[3]  # dx
    w = box[4]  # dy
    h = box[5]  # dz

    yaw = box[6]

    # 3d bounding box corners
    Box = np.array(
        [
            [-l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2],
            [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2],
            [-h / 2, -h / 2, -h / 2, -h / 2, h / 2, h / 2, h / 2, h / 2],
        ]
    )

    R = rotz(yaw)
    corners_3d = np.dot(R, Box)  # corners_3d: (3, 8)

    corners_3d[0, :] = corners_3d[0, :] + x
    corners_3d[1, :] = corners_3d[1, :] + y
    corners_3d[2, :] = corners_3d[2, :] + z

    return np.transpose(corners_3d)


def create_box_from_corners(corners, color=None):
    """
	corners: 8 corners(x, y, z)
	corners: array = 8*3
	#         7 -------- 6
	#        /|         /|
	#       4 -------- 5 .
	#       | |        | |
	#       . 3 -------- 2
	#       |/         |/
	#       0 -------- 1
	"""
    # 12 lines in a box
    lines = [
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 0],
        [4, 5],
        [5, 6],
        [6, 7],
        [7, 4],
        [0, 4],
        [1, 5],
        [2, 6],
        [3, 7],
    ]

    colors = [color for i in range(len(lines))]
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(corners)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)

    return line_set


def create_box(box, color=None):
    """
    box: list(8) [ x, y, z, dx, dy, dz, yaw]
    """
    box_corners = box2corners(box)
    box = create_box_from_corners(box_corners, color)
    return box
