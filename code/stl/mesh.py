import pymeshlab as pml
import numpy as np
import open3d as o3d
import pathlib

def gen_mesh(path_list, out_path, samplenum=None, radius=None, bestsamplepool=50, maxholesize=30, plot=False):
    # Load individual xyz-Files 
    meshes = pml.MeshSet()
    meshes.set_verbosity(True)
        
    for path in path_list:
        if type(path) is not str:
            path = path.as_posix()
            
        meshes.load_new_mesh(path)
        
    # Merge
    meshes.flatten_visible_layers(mergevisible=False)

    # Simplify
    if samplenum is not None and radius is not None:
        meshes.point_cloud_simplification(samplenum=samplenum,
                                          radius=pml.AbsoluteValue(radius), 
                                          bestsamplepool=bestsamplepool)
    elif samplenum is not None:
        meshes.point_cloud_simplification(samplenum=samplenum, 
                                              bestsamplepool=bestsamplepool)
    else:
        meshes.point_cloud_simplification(samplenum=100000, 
                                          bestsamplepool=bestsamplepool)

    # Reconstruct surfaces
    meshes.surface_reconstruction_ball_pivoting(deletefaces=True)
        
    # Orient faces
    meshes.re_orient_all_faces_coherentely()

    #meshes.repair_non_manifold_edges(0)

    # Close holes
    meshes.close_holes(maxholesize=maxholesize)        

    # Save as STL
    out_path = pathlib.Path(out_path)
    if out_path.suffix != "stl" and out_path.is_dir():
        out_path = out_path / "mesh_mm.stl"
    meshes.save_current_mesh(out_path.as_posix())

    # Visualize
    if plot:
        mesh = o3d.io.read_triangle_mesh(out_path.as_posix())
        o3d.visualization.draw_geometries([mesh])

def visualize_mesh(path):
    if type(path) is not str:
        path = path.as_posix()
        
    mesh = o3d.io.read_triangle_mesh(path)
    
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10, origin=[0, 0, 0])
    
    o3d.visualization.draw_geometries([mesh, mesh_frame])