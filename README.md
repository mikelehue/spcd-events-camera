# SPDC Event Camera (Andor iStar)

Batch snapshots with an **Andor iStar** and basic **event/“cluster”** counting on SPDC images.  
This is real lab code: it connects to the camera, captures data, saves **NPZ/PNG**,  
and runs a simple cluster analysis to compare **exposure time** and **gate width**.

---

## What this shows (for recruiters)

- **Hardware API integration** (Andor SDK3): AOI, gating, gain, buffer queueing.  
- **Mono12Packed → uint16** conversion and reproducible data dumps (NPZ + preview PNG).  
- **Quantitative analysis**: threshold + connectivity → clusters per frame; CSV + plots.  
- Clear **communication of results** with quick visualizations.

---

## Repository structure

.
├─ README.md
├─ snapshot_single_acq.py # single acquisition (smoke test, shows PNG)
├─ snapshot_batch.py # batch of N frames → .npz + first-frame .png
└─ data_analysis.py # counts clusters per frame, saves CSV, makes plots

yaml
Copiar código

> Repo name is `spcd-events-camera` right now; the project is about **SPDC**.

---

## Requirements

- Python 3.9+
- Packages:
numpy
matplotlib
pandas
scikit-image
tqdm
seaborn # optional (used by some plots)

- Camera SDK: `pyAndorSDK3` (install according to your lab environment).

Quick install:
```bash
pip install numpy matplotlib pandas scikit-image tqdm seaborn
```

1) Data capture
1.1 Quick test (single frame)
snapshot_single_acq.py connects, configures a full-frame AOI (2160×2560),
takes one acquisition, and displays it.

python snapshot_single_acq.py
It:

sets ExposureTime, FrameRate, GateMode = "CW Off", MCPGain

unpacks Mono12Packed into uint16

shows a matplotlib preview with a colorbar

Good for checking connection, focus, and signal levels.

1.2 Batch of frames (NPZ + PNG)
snapshot_batch.py performs N acquisitions using DDG gating, then saves:

a PNG of the first frame

an NPZ stack with shape (N, H, W)

How to use (current workflow):

Edit parameters near the top:

python
Copiar código
exposure_times_s = [2.5e-3]   # seconds
gate_widths_s    = [5e-9]     # seconds
gains            = [4095]
frame_count      = 10         # frames per run
add_path         = "2025_10_27_01"
In main() the camera is configured:

python
Copiar código
cam.GateMode = "DDG"
cam.AOIWidth  = 1500
cam.AOIHeight = 900
cam.AOILeft   = 400
cam.AOITop    = 100
cam.PixelEncoding = "Mono12Packed"
Run:

bash
Copiar código
python snapshot_batch.py
Outputs (in acqui-pics/):

PNG: first frame preview (grayscale)

NPZ: images → array (N, H, W) with uint16

Filename pattern:

cpp
Copiar código
gn{gain}_n{frames}_t{exposure}_gate_DDG_width{gate_width}_{YYYY-mm-dd_HH-MM}.npz
Example:

Copiar código
gn4095_n10_t0p0025_gate_DDG_width5p00e-09_2025-10-27_14-30.npz
t0p0025 means 0.0025 with . replaced by p for filesystem safety.

2) Data analysis (cluster counting)
data_analysis.py scans a folder of NPZ files, parses parameters from the filename,
applies a threshold and connectivity labeling to count clusters per frame,
then saves a CSV and generates plots.

Editable parameters at the top:

python
Copiar código
add_dir      = "2025_10_27_00"          # folder containing .npz files
THRESHOLD    = np.arange(104,110,1)     # thresholds to sweep
CONNECTIVITY = 1                         # 4-neighborhood (use 2 for 8-neighborhood)
Run:

bash
Copiar código
python data_analysis.py
It produces:

CSV: cluster_vs_exposure-time.csv in the data folder

Plots: lines (clusters vs exposure / gate width), histograms, etc.

Console logs: parsed gain, n_frames, texp_s, gate_width_s, and more.

How it works:

parse_fname(...) extracts gain, n, texp_s, gate_width_s, date/time from filenames

count_clusters(img, thr, connectivity) uses skimage.measure.label() → returns number of blobs

For each file and each threshold, it computes:

n_clusters (list per frame)

n_clusters_mean, n_clusters_std

appends to a DataFrame → CSV → plots

With a single frame per file, std can appear as NaN — expected.

Practical notes
Cooling: batch script waits until ~2 °C and aborts if TemperatureStatus == "Fault".

AOI: cropping (e.g. 1500×900) speeds up capture and analysis.

Threshold: 104–110 is a reasonable starting range.

Connectivity: 1 = 4-neighbors; 2 = 8-neighbors (if spots appear split).

File size: NPZ files can be large — organize by date (add_path) and clean old runs.

Limitations (honest list)
Parameters are changed by editing the script (no CLI yet).

No metadata.json; experiment info lives in filenames.

Cluster counting is basic (global threshold + connectivity; no peak fitting).

Short roadmap

Minimal CLI for snapshot_batch.py (--n, --exposure, --gate-width, --out)

metadata.json per run (gain, AOI, temperature, etc.)

“Paper-ready” plots: g²(τ), rate maps vs parameters

Load data in a notebook
python
Copiar código
import numpy as np
d = np.load("acqui-pics/2025_10_27_01/gn4095_n10_t0p0025_gate_DDG_width5p00e-09_2025-10-27_14-30.npz")
imgs = d["images"]  # (N, H, W), uint16
imgs.mean(), imgs.std()
