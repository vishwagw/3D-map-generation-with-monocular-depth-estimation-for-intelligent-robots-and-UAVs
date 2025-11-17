import argparse
import cv2
import os


def extract_frames(video_path, out_dir, resize=None, start=0, end=None, fps=None):
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    vid_fps = cap.get(cv2.CAP_PROP_FPS)
    if end is None:
        end = total
    frame_idx = 0
    saved = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx < start:
            frame_idx += 1
            continue
        if frame_idx >= end:
            break
        if fps is not None:
            # sample frames according to target fps
            step = max(1, int(round(vid_fps / fps)))
            if (frame_idx - start) % step != 0:
                frame_idx += 1
                continue
        if resize is not None:
            frame = cv2.resize(frame, (resize[0], resize[1]))
        out_path = os.path.join(out_dir, f"frame_{frame_idx:06d}.png")
        cv2.imwrite(out_path, frame)
        saved += 1
        frame_idx += 1
    cap.release()
    print(f"Extracted {saved} frames to {out_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--resize", nargs=2, type=int, help="width height")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int)
    parser.add_argument("--fps", type=float, help="resample to this fps")
    args = parser.parse_args()
    extract_frames(args.video, args.out_dir, resize=args.resize, start=args.start, end=args.end, fps=args.fps)


if __name__ == '__main__':
    main()
