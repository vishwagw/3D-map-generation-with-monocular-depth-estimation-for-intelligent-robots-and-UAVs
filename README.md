# Monocular 3D mapping prototype (offline, pure-Python)

This repository contains a minimal offline prototype to generate a 3D map (mesh/PLY) from a single front-facing camera video using only Python.

Overview
- Extract frames from recorded video
- Run a pretrained MiDaS depth estimator on each frame
- Estimate relative camera poses with a pure-Python lightweight VO (ORB + Essential matrix)
- Fuse depth frames + poses into a TSDF via Open3D and export a mesh

Notes & limitations
- No sensors available -> metric scale is ambiguous. The output mesh will have arbitrary scale unless you provide scale reference.
- MiDaS provides relative depth; we normalize it to a user-chosen metric range (default 0.3–5.0 m). This is a prototype strategy only.

Requirements
Install dependencies (preferably in a virtual env):

```powershell
python -m pip install -r requirements.txt
```

Quick usage

1. Extract frames from a video:

```powershell
python capture_to_frames.py --video input1.mp4 --out_dir data/frames --resize 640 480
```

2. Run depth inference (MiDaS):

```powershell
python depth_infer.py --frames data/frames --out_dir data/depths --model MiDaS_small --batch 4
```

3. Estimate poses (pure-Python VO):

```powershell
python pose_estimate.py --frames data/frames --out_dir data/poses --fx 500 --fy 500 --cx 320 --cy 240
```

4. Fuse into TSDF and export mesh:

```powershell
python fuse_tsdf.py --frames data/frames --depths data/depths --poses out/poses.npy --out mesh.ply --width 640 --height 480 --fx 500
```

Or run the demo script which runs the full pipeline (frames -> depth -> poses -> fusion):

```powershell
python run_demo.py --video path\to\video.mp4 --work_dir data --resize 640 480
```

If you want to tune intrinsics or the depth range, pass arguments to the scripts. See help (`-h`) for each script.
