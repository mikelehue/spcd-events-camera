# Capture Module

This part of the project shows how to **control scientific hardware through Python APIs**.

## What this demonstrates (for recruiters)
- Integration with a **real instrument SDK** (Andor iStar camera).
- Ability to define **repeatable acquisition protocols**.
- Logging of **metadata** (exposure, gain, gating, timestamps, run IDs).
- Clean, reusable code instead of one-off lab scripts.

## Planned structure

istar_setup.py # initialize camera, apply parameters safely
snapshot_batch.py # take repeated frames + metadata to disk
settings.json # example config with documented parameters

## Example usage (planned)
```bash
python capture/snapshot_batch.py --n 200 --exposure 5ms --out data/run_001/
