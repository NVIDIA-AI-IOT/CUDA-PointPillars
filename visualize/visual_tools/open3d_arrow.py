import open3d as o3d
import numpy as np


# def draw_geometries(pcds):
#     """
#     Draw Geometries
#     Args:
#         - pcds (): [pcd1,pcd2,...]
#     """
#     o3d.visualization.draw_geometries(pcds)


# def get_o3d_FOR(origin=[0, 0, 0], size=10):
#     """ 
#     Create a FOR that can be added to the open3d point cloud
#     """
#     mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
#     mesh_frame.translate(origin)
#     return mesh_frame


def vector_magnitude(vec):
    """
    Calculates a vector's magnitude.
    Args:
        - vec (): 
    """
    magnitude = np.sqrt(np.sum(vec ** 2))
    return magnitude


def calculate_zy_rotation_for_arrow(vec):
    """
    Calculates the rotations required to go from the vector vec to the 
    z axis vector of the original FOR. The first rotation that is 
    calculated is over the z axis. This will leave the vector vec on the
    XZ plane. Then, the rotation over the y axis. 

    Returns the angles of rotation over axis z and y required to
    get the vector vec into the same orientation as axis z
    of the original FOR

    Args:
        - vec (): 
    """
    # Rotation over z axis of the FOR
    gamma = np.arctan(vec[1] / vec[0])
    Rz = np.array(
        [
            [np.cos(gamma), -np.sin(gamma), 0],
            [np.sin(gamma), np.cos(gamma), 0],
            [0, 0, 1],
        ]
    )
    # Rotate vec to calculate next rotation
    vec = Rz.T @ vec.reshape(-1, 1)
    vec = vec.reshape(-1)
    # Rotation over y axis of the FOR
    beta = np.arctan(vec[0] / (vec[2] + 1e-8))
    Ry = np.array(
        [[np.cos(beta), 0, np.sin(beta)], [0, 1, 0], [-np.sin(beta), 0, np.cos(beta)]]
    )
    return (Rz, Ry)


def get_arrow(scale=10):
    """
    Create an arrow in for Open3D
    """
    cone_height = scale * 0.2
    cylinder_height = scale * 0.8
    cone_radius = scale / 10
    cylinder_radius = scale / 20
    mesh_frame = o3d.geometry.TriangleMesh.create_arrow(
        cone_radius=0.5,
        cone_height=cone_height,
        cylinder_radius=0.25,
        cylinder_height=cylinder_height,
    )
    return mesh_frame


def create_arrow(origin=[0, 0, 0], end=None, color=[1, 0, 0], vec=None):
    """
    Creates an arrow from an origin point to an end point,
    or create an arrow from a vector vec starting from origin.
    Args:
        - end (): End point. [x,y,z]
        - vec (): Vector. [i,j,k]
    """
    scale = 10
    Ry = Rz = np.eye(3)
    T = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    T[:3, -1] = origin
    if end is not None:
        vec = np.array(end) - np.array(origin)
    elif vec is not None:
        vec = np.array(vec)
    if end is not None or vec is not None:
        scale = vector_magnitude(vec)
        Rz, Ry = calculate_zy_rotation_for_arrow(vec)
    mesh = get_arrow(scale)
    # Create the arrow
    mesh.rotate(Ry, center=np.array([0, 0, 0]))
    mesh.rotate(Rz, center=np.array([0, 0, 0]))
    mesh.translate(origin)

    mesh.paint_uniform_color(color)
    return mesh


# # Create a Cartesian Frame of Reference
# FOR = get_o3d_FOR()
# # Create an arrow from point (5,5,5) to point (10,10,10)
# arrow = get_arrow([5,5,5],[10,10,10])

# # Create an arrow representing vector vec, starting at (5,5,5)
# # arrow = get_arrow([5,5,5],vec=[5,5,5])

# # Create an arrow in the same place as the z axis
# # arrow = get_arrow()

# # Draw everything
# draw_geometries([FOR,arrow])
