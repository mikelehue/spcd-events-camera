# SPDC Event Camera (Andor iStar)

Batch snapshots from an Andor iStar and basic SPDC event detection/correlations.

## What this shows
- Hardware API integration (camera SDK), robust CLI & logging
- Reproducible analysis pipeline (events, histograms, g^(2) basics)
- Clear plots and docs that non-experts can follow

## Quickstart
```bash
pip install -r requirements.txt
python capture/snapshot_batch.py --help
python analysis/detect_events.py --input data/sample/
