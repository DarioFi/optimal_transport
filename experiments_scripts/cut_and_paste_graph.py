import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

# ——— 1) Define your "hand-drawn" control points ———
#    (x, y) coordinates in order, from Source → cut
cp = np.array([
    [0.0, 0.2],   # Source
    [0.8, 0.3],   # kink out
    [0.8, 1.5],   # cut and beginning loop
    [1.4, 1.5],   # across top of loop
    [1.2, 0.3],   # down out of loop
    [2.2, 0.6],   # cut point
])

# Parameterize them by t = 0…N−1
t  = np.arange(len(cp))
# Build a cubic spline through them
spline = make_interp_spline(t, cp, k=3)

# Sample a fine path
t_fine = np.linspace(t.min(), t.max(), 400)
path   = spline(t_fine)
x1, y1 = path[:,0], path[:,1]

# ——— 2) Dashed continuation after the cut ———

CUT_IND = 1
CUT_END = 4

x_cut, y_cut = cp[CUT_IND]
x_cut_end, y_cut_end = cp[CUT_END]



x2 = np.linspace(x_cut, x_cut_end, 100)
y2 = np.linspace(y_cut, y_cut_end, 100)


# ——— 3) Plot everything ———
fig, ax = plt.subplots(figsize=(6,6))

# Dashed tail
ax.plot(x2, y2, lw=2, ls='--', color='black')


# Solid spline path
ax.plot(x1, y1, lw=2, color='black')


# ——— 4) Mark points & labels ———
# Source
ax.plot(*cp[0], 'o', ms=8, color='black')
ax.text(cp[0,0]-0.05, cp[0,1]-0.1, "Source", ha='right', va='top')

# Cut
ax.plot(*cp[CUT_IND], 'o', ms=10, mfc='white', mec='black')
ax.plot(*cp[CUT_END], 'o', ms=10, mfc='white', mec='black')
ax.text(cp[CUT_IND,0]+0.05, cp[CUT_IND,1]-0.1, "Cut", ha='left', va='top')

# γ label somewhere along the rising branch
mid_idx = np.searchsorted(t_fine, 1.1)  # around the vertical kink
ax.text(x1[mid_idx]+0.02, y1[mid_idx]+0.1, r"$\gamma$", fontsize=14)



# ——— 6) Final tweaks ———
ax.set_aspect('equal', 'box')
ax.set_xlim(-0.1, 3.1)
ax.set_ylim(-0.1, 2.6)
ax.axis('off')

plt.tight_layout()
plt.show()
