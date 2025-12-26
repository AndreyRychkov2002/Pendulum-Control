# main_underactuated.py — README

**Overview**
- **Purpose:** Simulate and visualize a two-link underactuated pendulum (actuation at joint 1 only). The script runs a time-domain simulation, plots state trajectories and control signals, and saves an animation (`pendulum.gif`).

**Requirements**
- **Python:** 3.8+
- **Libraries:** `numpy`, `scipy`, `matplotlib`, `pillow`

**Quick Install**
- Install dependencies (fish shell):
```fish
python -m pip install --user numpy scipy matplotlib pillow
```

**Run**
- From the project root:
```fish
python main_underactuated.py
```

**Outputs**
- `pendulum.gif` — animation saved by the script (if enabled).
- Figures — Matplotlib windows showing angles, velocities and control inputs.

**Configuration**
- Edit top-of-file parameters in `main_underactuated.py` to change simulation settings: `dt`, `t_stop`, `L1`, `L2`, `M1`, `M2`, `NOISE_MAGNITUDE`, etc.
- The script computes LQR gains for both the downward and inverted equilibria and applies the chosen controller.

**Headless / CI**
- For headless environments (no display), add before other matplotlib imports:
```python
import matplotlib
matplotlib.use('Agg')
```
so figures and the GIF can be saved without a GUI.

**Notes & Next Steps**
- The code was cleaned of unnecessary trailing semicolons for style clarity.
- I can add a CLI flag to toggle interactive plotting vs saving outputs, or add automated tests — tell me which you prefer.

**Contact / Help**
- Reply with any requested changes (format, additional examples, or commit message) and I will update the file.
