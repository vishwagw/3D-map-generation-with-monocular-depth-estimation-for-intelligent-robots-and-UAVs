import argparse
import os
import glob
import numpy as np
from tqdm import tqdm
import torch
import cv2


def run_midas_on_folder(frames_dir, out_dir, model_name='MiDaS_small', device=None, batch=1, min_depth=0.3, max_depth=5.0):
    os.makedirs(out_dir, exist_ok=True)
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    midas = torch.hub.load('intel-isl/MiDaS', model_name)
    midas.to(device)
    midas.eval()
    transforms = torch.hub.load('intel-isl/MiDaS', 'transforms')
    if model_name == 'MiDaS_small':
        transform = transforms.small_transform
    else:
        transform = transforms.default_transform

    img_paths = sorted(glob.glob(os.path.join(frames_dir, '*.png')) + glob.glob(os.path.join(frames_dir, '*.jpg')))
    print(f"Found {len(img_paths)} images in {frames_dir}")
    if len(img_paths) == 0:
        print("No image files found! Check the frames directory path.")
        return
    
    for p in tqdm(img_paths, desc='Depth infer'):
        img = cv2.imread(p)
        if img is None:
            print(f"Warning: Failed to load image {p}, skipping...")
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        input_batch = transform(img_rgb).to(device)
        with torch.no_grad():
            prediction = midas(input_batch.unsqueeze(0)) if input_batch.dim()==3 else midas(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1), size=img_rgb.shape[:2], mode='bicubic', align_corners=False).squeeze()
            depth = prediction.cpu().numpy()
        # normalize to user metric range
        dmin, dmax = depth.min(), depth.max()
        if dmax - dmin > 1e-6:
            depth_n = (depth - dmin) / (dmax - dmin)
        else:
            depth_n = depth * 0.0
        depth_m = min_depth + depth_n * (max_depth - min_depth)

        base = os.path.splitext(os.path.basename(p))[0]
        np.save(os.path.join(out_dir, base + '.npy'), depth_m.astype(np.float32))
        # save visualization
        vis = (255 * (depth_n)).astype('uint8')
        cv2.imwrite(os.path.join(out_dir, base + '_vis.png'), vis)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--model", default='MiDaS_small')
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--min_depth", type=float, default=0.3)
    parser.add_argument("--max_depth", type=float, default=5.0)
    args = parser.parse_args()
    run_midas_on_folder(args.frames, args.out_dir, model_name=args.model, batch=args.batch, min_depth=args.min_depth, max_depth=args.max_depth)


if __name__ == '__main__':
    main()
