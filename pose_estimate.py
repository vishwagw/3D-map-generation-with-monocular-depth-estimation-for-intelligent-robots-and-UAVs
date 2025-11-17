import argparse
import os
import glob
import numpy as np
import cv2

# safe imread with PIL fallback for problematic PNGs
def imread_safe(path):
    img = cv2.imread(path)
    if img is not None:
        return img
    try:
        from PIL import Image
        import numpy as _np
        with Image.open(path) as im:
            im = im.convert('RGB')
            arr = _np.array(im)
            # PIL gives RGB, convert to BGR for OpenCV compatibility
            arr = arr[:, :, ::-1].copy()
            return arr
    except Exception:
        return None


def estimate_poses(frames_dir, out_dir, fx, fy, cx, cy, match_threshold=32):
    os.makedirs(out_dir, exist_ok=True)
    img_paths = sorted(glob.glob(os.path.join(frames_dir, '*.png')) + glob.glob(os.path.join(frames_dir, '*.jpg')))
    N = len(img_paths)
    if N == 0:
        raise RuntimeError('No frames found')
    poses = []
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=float)

    orb = cv2.ORB_create(2000)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    prev_kp = None
    prev_des = None
    prev_img = None
    pose = np.eye(4, dtype=float)
    poses.append(pose.copy())

    for i in range(1, N):
        img1 = imread_safe(img_paths[i-1])
        img2 = imread_safe(img_paths[i])
        if img1 is None:
            print(f"Warning: failed to read image: {img_paths[i-1]}. Skipping frame {i-1} -> {i}.")
            poses.append(pose.copy())
            continue
        if img2 is None:
            print(f"Warning: failed to read image: {img_paths[i]}. Skipping frame {i-1} -> {i}.")
            poses.append(pose.copy())
            continue
        g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        kp1, des1 = orb.detectAndCompute(g1, None)
        kp2, des2 = orb.detectAndCompute(g2, None)
        if des1 is None or des2 is None:
            poses.append(pose.copy())
            continue
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        good = [m for m in matches if m.distance < match_threshold]
        if len(good) < 8:
            poses.append(pose.copy())
            continue
        pts1 = np.array([kp1[m.queryIdx].pt for m in good])
        pts2 = np.array([kp2[m.trainIdx].pt for m in good])

        E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        if E is None:
            poses.append(pose.copy())
            continue
        _, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, K)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t.ravel()
        # chain poses: pose_next = pose_prev @ T
        pose = pose @ T
        poses.append(pose.copy())

    poses = np.stack(poses, axis=0)
    np.save(os.path.join(out_dir, 'poses.npy'), poses)
    print(f"Saved {len(poses)} poses to {out_dir}/poses.npy")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--fx", type=float, default=500.0)
    parser.add_argument("--fy", type=float, default=500.0)
    parser.add_argument("--cx", type=float, default=None)
    parser.add_argument("--cy", type=float, default=None)
    args = parser.parse_args()
    # infer cx/cy from first image if not provided
    import glob, cv2
    img_paths = sorted(glob.glob(os.path.join(args.frames, '*.png')) + glob.glob(os.path.join(args.frames, '*.jpg')))
    if len(img_paths) == 0:
        raise RuntimeError('No frames found')
    img = imread_safe(img_paths[0])
    if img is None:
        raise RuntimeError(f"Failed to read first image '{img_paths[0]}'. Check file integrity or path.")
    h, w = img.shape[:2]
    if args.cx is None:
        args.cx = w / 2.0
    if args.cy is None:
        args.cy = h / 2.0
    estimate_poses(args.frames, args.out_dir, args.fx, args.fy, args.cx, args.cy)


if __name__ == '__main__':
    main()
