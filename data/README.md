# Data

Datasets are **not** stored in this repository (size, and most game-derived datasets are licensed for
non-commercial research only). Download them locally into `data/` with the scripts in `scripts/`.

| Dataset (paper) | Content | Used for | License |
| --- | --- | --- | --- |
| GTA-Human (Cai et al., TPAMI 2024) | GTA V, full-body humans, SMPL | Body mode, pipeline tests | Research only |
| SAIL-VOS (Hu et al., CVPR 2019) | GTA V video, instance masks | Body masks + video inpainting | Research only, requires GTA V |
| MOTSynth (Fabbri et al., ICCV 2021) | GTA V crowds, boxes, masks, IDs | Detection + tracking | Research only |
| JTA (Fabbri et al., ECCV 2018) | GTA V pedestrians, poses | Head boxes from poses | Research only, requires GTA V |
| WIDER FACE (Yang et al., CVPR 2016) | Real-world faces | Face detector reference | Research only |
| Own captures | Close-up NPC frames from games | Face mode, fine-tuning, evaluation | Not redistributed |

Suggested layout:

```
data/
├── samples/        # small shared test set (50-100 frames), synced via shared drive
├── gta_human/
├── sail_vos/
└── own/            # own screenshots + CVAT / Label Studio exports
```
