# DRC: Pre-Actuated Dynamic Residual Control for High-Speed Rigid Interception

[![Paper](https://img.shields.io/badge/PDF-Download-red)](paper/DRC_Pre_Actuated_....pdf)
[![License](https://img.shields.io/badge/License-MIT-blue)](LICENSE)

**Zero-tolerance real-world benchmark**: Rigid tray interception at 3-5 m/s, no mechanical caging.

## Overview
- **83% success rate** on real hardware (vs baseline 72%).
- End-to-end latency reduced from 120 ms to 35 ms.
- Key innovation: Velocity-level pre-actuation residual compensation for execution lag.

![Teaser](results/figures/fig1_hardware_interception.gif) <!-- 硬件+序列帧GIF -->

## Highlights
- Frozen NKP for global guidance + parallel residual for temporal compensation.
- Aggressive latency DDR (20-150 ms injection).
- Real robot: AgileX base + xArm + rigid tray.

![Mechanism](results/figures/fig3_pre_actuation.png)

## Results
| Method | Success Rate | Avg. Lag |
|--------|--------------|----------|
| NKP    | 72.0%       | 120 ms  |
| DRC    | **83.0%**   | **35 ms** |

Video demos: [Success](results/videos/success.mp4) | [Failure Analysis](results/videos/failure.mp4)

## Code
- Training: `python train_DCMM.py --config-name=config_task=Catching`
- Requirements: `pip install -r requirements.txt`

## Citation
