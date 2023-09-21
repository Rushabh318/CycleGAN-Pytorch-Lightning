import numpy as np
import pathlib
import glob
from tqdm import tqdm
import open3d as o3d
from copy import deepcopy
import cv2

from utils.files import load_raw_img, load_img
from analysis.slices import *

def get_slicewise_ISO50_boundary(img, height):
    hist = generate_slice_hist(img)
    iso50, _, _ = get_iso50_value(hist)
    
    mask = np.where(img > iso50, 1, 0)
    
    return get_boundary_from_mask(mask, height)
    
def get_gradient_boundary(img, height):
    pass
    
def get_boundary_from_thr(img, height, thr, **kwargs):
    mask = np.where(img > thr, 1, 0)
    
    return get_boundary_from_mask(mask, height)
    
def get_boundary_from_mask(mask, height, **kwargs):
    # map for simplicity to {0, 1}
    if np.max(mask) > 1:
        mask = np.where(mask >= 1, 1, 0)

    # calculate the gradient to extract the boundary points only
    gradient = np.gradient(mask)
    gradient_map = np.sqrt(gradient[0] * gradient[0] + gradient[1] * gradient[1])
    gradient_map = np.ceil(gradient_map)
    
    # for simplicity (and unambiguousness), the edge boundary is defined 
    # at the outer pixels of the mask; they can be selected as the common 
    # values between the (absolut) gradient map and the mask itself  
    edge_map = np.uint8(mask) & np.uint8(gradient_map) 
    
    edge_x, edge_y = np.where(edge_map > 0)

    edge_list = [[edge_x[i], edge_y[i], height] for i in range(len(edge_x))]

    return edge_list

def get_boundary_mask(mask):
    boundary_points = get_boundary_from_mask(mask, 0)
    mask_b = np.zeros_like(mask)
    for p in boundary_points:
        mask_b[p[0], p[1]] = 1

    return mask_b

def upscale_img_snip(img_snip, scale_factor=2, mode="rescale", **kwargs):
    if mode not in ["rescale", "interpolate"]:
        raise KeyError()

    new_size = ((img_snip.shape[0]-1)*scale_factor + 1, 
                (img_snip.shape[1]-1)*scale_factor + 1)

    if mode == "rescale":
        upscaled_img = cv2.resize(np.float32(img_snip), dsize=new_size)
    else:
        raise NotImplemented()

    return upscaled_img

def get_subpixel_boundary(img, height, thr, mode="filtered", **kwargs):
    if "num_neighbors" not in kwargs.keys():
        num_neighbors = 2
    else:
        num_neighbors = kwargs["num_neighbors"]
        
    if "scale_factor" not in kwargs.keys():
        scale_factor = 2
    else:
        scale_factor = kwargs["scale_factor"]
        
    if mode not in ["all", "closest", "filtered"]:
        raise KeyError()
        
    # get the pixel-level boundary based on a given threshold (e.g. ISO50)
    boundary_points = get_boundary_from_thr(img, height, thr)
    
    # for each boundary point, interpolate a small surrounding area (neighborhood) 
    # and check whether sub-pixel localtions can be extracted
    refined_points = []
    for p in boundary_points:
        # select image snip
        top_left = (max(0, p[0]-num_neighbors), 
                    max(0, p[1]-num_neighbors))
        
        bottom_right = (min(img.shape[0], p[0]+num_neighbors+1),
                        min(img.shape[1], p[1]+num_neighbors+1))
        
        img_snip = img[top_left[0]:bottom_right[0], 
                       top_left[1]:bottom_right[1]]

        # interpolate images snip and determine new boundary points
        resized_snip = upscale_img_snip(img_snip, scale_factor=scale_factor)
        new_points = get_boundary_from_thr(resized_snip, 0, thr)
        
        # if new points are found, correct their position w.r.t. to the global image
        if len(new_points) == 0:
            continue
               
        # depending on the mode, save either all or only the closet point
        if mode == "closest": 
            center_point = np.asarray(resized_snip.shape) // 2
            new_points = np.asarray(new_points)[:, :2]
            dist = np.linalg.norm(new_points - center_point, axis=1)
            closest_point = new_points[np.argmin(dist)]
            
            # restore global point position and add to list
            new_p = (closest_point - center_point) / scale_factor + np.asarray(p[:2])
            refined_points.append([new_p[0], new_p[1], height])
        else:
            # restore global point position and add to list
            new_points = (np.asarray(new_points)[:, :2]  - np.asarray(resized_snip.shape) // 2) / scale_factor + np.asarray(p[:2])
            
            points = [[new_points[i][0], new_points[i][1], height] for i in range(len(new_points))]
            refined_points += points
  
    # the saving of all new points from the neighborhood results in many identical points; 
    # filter them out to reduce the overall point cloud size
    if mode == "filtered":
        refined_points.sort(key= lambda x: x[0])
        filtered_points = []
        tmp = []
        for p in refined_points:
            if len(tmp) == 0:
                tmp.append(p)
                continue
            
            if np.abs(tmp[0][0] - p[0]) < 1e-3:
                tmp.append(p)
            else:
                _, uniq_pos = np.unique(np.asarray(tmp)[:, 1], return_index=True)
                for pos in uniq_pos:
                    filtered_points.append(tmp[pos])
                
                tmp = [p]
  
        return filtered_points
    else:
        return refined_points
    
def get_img_stack_points(img_stack_path, slice_points_func, order=(0,1,2), debug=False, **kwargs):    
    
    file_list = list(img_stack_path.glob("*.png"))
    file_list.sort(key=lambda x: int(x.stem.split("_")[-1]))
    
    point_list = []
    for f in tqdm(file_list):
        # get height as file index
        height = int(f.stem.split("_")[-1])
        
        # load image
        img = load_img(f, mode="I")
                      
        # get edge/surface points
        points = slice_points_func(img, height, **kwargs)
        
        point_list += points
        
        if "point_limit" in kwargs.keys() and kwargs["point_limit"] is not None:
            point_limit = kwargs["point_limit"]
            debug = True
        else:
            point_limit = 10000
        
        if debug and len(point_list) > point_limit:
            break
        
    reordered_list = [[p[order[0]], p[order[1]], p[order[2]]] for p in point_list]           
        
    return reordered_list

def gen_point_cloud(stack_dict, out_path, sizes, scale_factor, slice_points_func, 
                    debug=False, **kwargs):
    order = {"XY": (1,0,2), "XZ": (1,2,0), "YZ": (2,1,0)}
    
    img_stack_points = {}
    for key in stack_dict:
        print("Generate edge points for image stack: {}".format(key))
        img_stack_points[key] = get_img_stack_points(stack_dict[key], 
                                                     slice_points_func,
                                                     order=order[key], 
                                                     debug=debug,
                                                     **kwargs)

    if out_path.is_dir():
        out_path = out_path / "surface_points.xyz"
        
    num_points = 0
    with open(out_path, "w") as f:
        for key in img_stack_points:
            print("Writing points of {} stack".format(key))
            for p in tqdm(img_stack_points[key]):
                tmp = np.asarray(p)
                    
                if "offset_corr" in kwargs.keys() and kwargs["offset_corr"] is not None:
                    offset = kwargs["offset_corr"][key]
                    offset = np.asarray([offset[order[key][0]], 
                                         offset[order[key][1]], 
                                         offset[order[key][2]]]) 
                    tmp += offset
                          
                # escape special case (not fully understood yet; TODO)
                if key == "XZ":
                    tmp[1] = sizes["Y"] - tmp[1]
                    
                tmp = tmp * scale_factor

                line = "{0} {1} {2}\n".format(*list(tmp))

                f.write(line)
                
                num_points += 1
    
    print("Total number of points: {}".format(num_points))
                
def visualize_point_clouds(path_list, downsampling_factor=10):
    colors = [[1,0,0], [0,1,0], [0,0,1]]
    
    pcd_list = []
    for i, path in enumerate(path_list):
        if type(path) is not str:
            path = path.as_posix()
        
        pcd = o3d.io.read_point_cloud(path, format="xyz")
        pcd = pcd.uniform_down_sample(every_k_points=downsampling_factor)
        pcd.paint_uniform_color(colors[i])
        
        pcd_list.append(pcd)
    
    pcd = deepcopy(pcd_list[0])
    for i in range(1, len(pcd_list)):
        pcd += pcd_list[i]
        
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10, origin=[-10, -10, -10])
    o3d.visualization.draw_geometries([pcd, mesh_frame], point_show_normal=False)