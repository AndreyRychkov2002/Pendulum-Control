# main_underactuated.py — Underactuated Two-Link Pendulum Simulation

**Overview**
- **Purpose:** Simulate and visualize a two-link underactuated pendulum with LQR control (torque applied only to joint 1) and save an animation.

**Requirements**
- **Python:** 3.8+
- **Libraries:** `numpy`, `scipy`, `matplotlib`, `pillow`

**Install**
- Install dependencies (fish shell):
```fish
python -m pip install --user numpy scipy matplotlib pillow
```

**Run**
- Execute the script from the repository root:
```fish
python main_underactuated.py
```

**Outputs**
- **Animation:** `pendulum.gif` (saved in the working directory by the script).
- **Plots:** Matplotlib windows showing state trajectories and control signals.

**Configuration**
- Edit runtime parameters at the top of `main_underactuated.py`: `dt`, `t_stop`, `L1`, `L2`, `M1`, `M2`, `NOISE_MAGNITUDE`, etc.
- The script computes LQR gains for both the downward (stable) and inverted (unstable) equilibria.

**Notes**
- For headless servers (no display), set a non-interactive backend before other matplotlib imports:
```python
import matplotlib
matplotlib.use('Agg')
```
- The source has been cleaned of unnecessary trailing semicolons for style consistency.

**Contact**
- If you want improvements (save different formats, export data, or change controller), tell me and I can update the README and the script.