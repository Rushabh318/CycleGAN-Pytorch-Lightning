import numpy as np
import open3d
import pathlib

def stl2obj_converter(stl_file, obj_file):
    stl_mesh = open3d.io.read_triangle_mesh(stl_file)
    
    #open3d.io.write_triangle_mesh(obj_file, stl_mesh)
    open3d.visualization.draw_geometries([stl_mesh])
    
if __name__ == "__main__":
    path = pathlib.Path("D:/users/mrx/Lehre/SPP/STLs/")
    
    
    cube_mesh = open3d.io.read_triangle_mesh((path / "POM_cube_2_Al_cylinders_CUBE.STL").as_posix())
    cylinders_mesh = open3d.io.read_triangle_mesh((path / "POM_cube_2_Al_cylinders_cylinders.STL").as_posix())
    #cube_mesh.paint_uniform_color([1,0,0])
    #cylinders_mesh.paint_uniform_color([0,0,1])
       
    #open3d.visualization.draw_geometries([cube_mesh, cylinders_mesh])
    
    open3d.io.write_triangle_mesh("cube.obj", cube_mesh, print_progress=True)
    #stl2obj_converter(path.as_posix(), None)
    