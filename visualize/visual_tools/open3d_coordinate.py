import open3d as o3d


def create_coordinate(size=2.0, origin=[0, 0, 0]):
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=2.0, origin=[0, 0, 0]
    )
    return mesh_frame
