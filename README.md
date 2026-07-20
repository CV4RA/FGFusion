# FGFusion

Dual-source uncertainty for cross-modal retrieval.

## Usage

```bash
pip install -r requirements.txt
python train.py
```

## Structure

- `train.py` — main script
- `options.py` — dataset config
- `custom_dataset.py` — data loading
- `evaluate.py` — mAP metrics
- `class_names.py` — CLIP class names
- `generate_clip_protos.py` — CLIP protos
- `datasets/` — place .mat/.h5py here

## License

MIT
