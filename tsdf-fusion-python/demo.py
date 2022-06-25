"""Fuse 1000 RGB-D images from the 7-scenes dataset into a TSDF voxel volume with 2cm resolution.
"""

import time

import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import cv2
import numpy as np

import OpenEXR as exr
import Imath

import fusion

def read_depth_exr_file(filepath: str):
    "https://stackoverflow.com/questions/44630793/reading-exr-files-in-opencv"
    exrfile = exr.InputFile(filepath)
    raw_bytes = exrfile.channel('Z', Imath.PixelType(Imath.PixelType.FLOAT))
    depth_vector = np.frombuffer(raw_bytes, dtype=np.float32)
    height = exrfile.header()['displayWindow'].max.y + 1 - exrfile.header()['displayWindow'].min.y
    width = exrfile.header()['displayWindow'].max.x + 1 - exrfile.header()['displayWindow'].min.x
    depth_map = np.reshape(depth_vector, (height, width))
    return depth_map



if __name__ == "__main__":
  # ======================================================================================================== #
  # (Optional) This is an example of how to compute the 3D bounds
  # in world coordinates of the convex hull of all camera view
  # frustums in the dataset
  # ======================================================================================================== #
  print("Estimating voxel volume bounds...")
  n_imgs = 1
  campose = np.load("../data/front3d/70f7c6c1-c48f-4106-bcef-40b80b84bbad/campose_0001.npz")
  cam_intr = campose["intrinsic"]
  print(cam_intr)
  vol_bnds = np.zeros((3,2))
  for i in range(n_imgs):
    # Read depth image and camera pose
    depth_im = read_depth_exr_file("../data/front3d/70f7c6c1-c48f-4106-bcef-40b80b84bbad/depth_0001.exr")
    #depth_im /= 1000.  # depth is saved in 16-bit PNG in millimeters
    #depth_im[depth_im == 65.535] = 0  # set invalid depth to 0 (specific to 7-scenes dataset)
    cam_pose = campose["camera2world"]  # 4x4 rigid transformation matrix

    # Compute camera view frustum and extend convex hull
    view_frust_pts = fusion.get_view_frustum(depth_im, cam_intr, cam_pose)
    vol_bnds[:,0] = np.minimum(vol_bnds[:,0], np.amin(view_frust_pts, axis=1))
    vol_bnds[:,1] = np.maximum(vol_bnds[:,1], np.amax(view_frust_pts, axis=1))
  # ======================================================================================================== #

  # ======================================================================================================== #
  # Integrate
  # ======================================================================================================== #
  # Initialize voxel volume
  print("Initializing voxel volume...")
  tsdf_vol = fusion.TSDFVolume(vol_bnds, voxel_size=0.02)

  # Loop through RGB-D images and fuse them together
  t0_elapse = time.time()
  for i in range(n_imgs):
    print("Fusing frame %d/%d"%(i+1, n_imgs))

    # Read RGB-D image and camera pose
    color_image = cv2.cvtColor(cv2.imread("../data/front3d/70f7c6c1-c48f-4106-bcef-40b80b84bbad/rgb_0001.png"), cv2.COLOR_BGR2RGB)
    depth_im = read_depth_exr_file("../data/front3d/70f7c6c1-c48f-4106-bcef-40b80b84bbad/depth_0001.exr")
    #depth_im /= 1000.
    #depth_im[depth_im == 65.535] = 0
    cam_pose = campose["camera2world"]

    # Integrate observation into voxel volume (assume color aligned with depth)
    tsdf_vol.integrate(color_image, depth_im, cam_intr, cam_pose, obs_weight=1.)

  fps = n_imgs / (time.time() - t0_elapse)
  print("Average FPS: {:.2f}".format(fps))

  # Get mesh from voxel volume and save to disk (can be viewed with Meshlab)
  print("Saving mesh to mesh.ply...")
  verts, faces, norms, colors = tsdf_vol.get_mesh()
  fusion.meshwrite("mesh.ply", verts, faces, norms, colors)

  # Get point cloud from voxel volume and save to disk (can be viewed with Meshlab)
  print("Saving point cloud to pc.ply...")
  point_cloud = tsdf_vol.get_point_cloud()
  fusion.pcwrite("pc.ply", point_cloud)