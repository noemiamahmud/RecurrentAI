# RecurrentAI — Interactive Hopfield Network Visualizer

An interactive pygame app that lets you **draw patterns**, **store them as associative memories** in a Hopfield network, **corrupt them with noise**, and **watch the network recall** the original pattern in real time.

Built on the Hebbian-learning Hopfield network from `Hopfield_Nets_Noemia.py`.

![Python](https://img.shields.io/badge/Python-3.10+-blue) ![Pygame](https://img.shields.io/badge/Pygame-2.5+-green)

## Quick Start

```bash
pip install -r requirements.txt
python main.py
```

## Controls

| Action | Key / Mouse |
|---|---|
| Paint cell ON | Left-click / drag |
| Paint cell OFF | Right-click / drag |
| Store memory | `S` |
| Add noise | `N` (cycles 10% → 25% → 40%) |
| Auto-recall | `R` |
| Single step | `Space` |
| Clear grid | `C` |
| Erase all memories | `E` |
| Load preset shape | `1` `2` `3` `4` |
| Resize grid | `↑` / `↓` (6×6 to 14×14) |
| Quit | `Q` / `Esc` |

## How It Works

1. **Draw** a pattern on the grid (or load a preset).
2. **Store** it — the network learns via Hebbian weight updates.
3. Store more patterns (up to 6 memories).
4. **Load** a stored pattern, then **add noise** to corrupt it.
5. Hit **Recall** and watch the async updates converge back to the closest stored memory.
6. The **energy graph** tracks the network's energy function decreasing as it settles.

## Features

- Real-time asynchronous Hopfield recall with animated cell flips
- Live energy landscape graph
- Adjustable grid size (6×6 to 14×14 neurons)
- Built-in preset patterns: heart, cross, diamond, checker
- Hamming distance reported on convergence
- Clean dark-theme UI
