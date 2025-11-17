import argparse
import os
import sys
import subprocess


def run_cmd(cmd):
    print('RUN:', ' '.join(cmd))
    subprocess.check_call(cmd)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--work_dir", default='data')
    parser.add_argument("--resize", nargs=2, type=int, default=[640, 480])
    parser.add_argument("--fx", type=float, default=500.0)
    parser.add_argument("--min_depth", type=float, default=0.3)
    parser.add_argument("--max_depth", type=float, default=5.0)
    args = parser.parse_args()

    work = args.work_dir
    frames = os.path.join(work, 'frames')
    depths = os.path.join(work, 'depths')
    poses = os.path.join(work, 'poses')
    os.makedirs(frames, exist_ok=True)
    os.makedirs(depths, exist_ok=True)
    os.makedirs(poses, exist_ok=True)

    py = sys.executable
    run_cmd([py, 'capture_to_frames.py', '--video', args.video, '--out_dir', frames, '--resize', str(args.resize[0]), str(args.resize[1])])
    run_cmd([py, 'depth_infer.py', '--frames', frames, '--out_dir', depths, '--min_depth', str(args.min_depth), '--max_depth', str(args.max_depth)])
    run_cmd([py, 'pose_estimate.py', '--frames', frames, '--out_dir', poses, '--fx', str(args.fx), '--fy', str(args.fx)])
    mesh_out = os.path.join(work, 'mesh.ply')
    run_cmd([py, 'fuse_tsdf.py', '--frames', frames, '--depths', depths, '--poses', os.path.join(poses, 'poses.npy'), '--out', mesh_out, '--width', str(args.resize[0]), '--height', str(args.resize[1]), '--fx', str(args.fx)])

    print('Demo finished. Mesh at', mesh_out)


if __name__ == '__main__':
    main()
