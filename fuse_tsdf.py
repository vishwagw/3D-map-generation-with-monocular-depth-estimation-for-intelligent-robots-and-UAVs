import argparse
import os
import glob
import numpy as np
import open3d as o3d
import cv2


def fuse_tsdf(frames_dir, depths_dir, poses_file, out, width, height, fx, fy=None, cx=None, cy=None, depth_trunc=5.0, voxel_length=0.02):
    if fy is None:
        fy = fx
    if cx is None:
        cx = width / 2.0
    if cy is None:
        cy = height / 2.0

    poses = np.load(poses_file)
    depth_paths = sorted(glob.glob(os.path.join(depths_dir, '*.npy')))
    color_paths = sorted(glob.glob(os.path.join(frames_dir, '*.png')) + glob.glob(os.path.join(frames_dir, '*.jpg')))
    n = min(len(depth_paths), len(color_paths), len(poses))

    intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)
    tsdf = o3d.pipelines.integration.ScalableTSDFVolume(voxel_length=voxel_length, sdf_trunc=voxel_length * 4, color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)

    for i in range(n):
        depth = np.load(depth_paths[i])
        color = cv2.imread(color_paths[i])
        color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
        # ensure depth is HxW float32 in meters
        if depth.dtype != np.float32:
            depth = depth.astype(np.float32)
        depth_o3d = o3d.geometry.Image(depth)
        color_o3d = o3d.geometry.Image(color.astype('uint8'))
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(color_o3d, depth_o3d, depth_scale=1.0, depth_trunc=depth_trunc, convert_rgb_to_intensity=False)
        pose = poses[i]
        extrinsic = np.linalg.inv(pose)  # Open3D expects camera-to-world? use inverse if needed
        tsdf.integrate(rgbd, intrinsic, extrinsic)

    mesh = tsdf.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    o3d.io.write_triangle_mesh(out, mesh)
    print(f"Wrote mesh to {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", required=True)
    parser.add_argument("--depths", required=True)
    parser.add_argument("--poses", required=True)
    parser.add_argument("--out", default='mesh.ply')
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--fx", type=float, required=True)
    parser.add_argument("--fy", type=float)
    parser.add_argument("--cx", type=float)
    parser.add_argument("--cy", type=float)
    parser.add_argument("--depth_trunc", type=float, default=5.0)
    parser.add_argument("--voxel", type=float, default=0.02)
    args = parser.parse_args()
    fuse_tsdf(args.frames, args.depths, args.poses, args.out, args.width, args.height, args.fx, fy=args.fy, cx=args.cx, cy=args.cy, depth_trunc=args.depth_trunc, voxel_length=args.voxel)


if __name__ == '__main__':
    main()
