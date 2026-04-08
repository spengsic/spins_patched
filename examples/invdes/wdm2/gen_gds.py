"""Generate GDS from wdm2 optimisation epsilon, waveguides included via contour."""
import pickle
import numpy as np
import gdspy
import matplotlib.pyplot as plt

GRID_SPACING = 40   # nm
SI_EPS  = 2.20**2  # 4.84
OX_EPS  = 1.50**2  # 2.25
THRESHOLD = (SI_EPS + OX_EPS) / 2  # 3.545

# Crop y to design height only — waveguides extend full x so no x crop needed.
DESIGN_YMIN, DESIGN_YMAX = -1000, 1000

# Load final step epsilon.
with open("my_results/step105.pkl", "rb") as f:
    data = pickle.load(f)

# epsilon shape: (3, 125, 125, 1) — take z-component, real part, squeeze to 2D.
eps = np.array(data["monitor_data"]["epsilon"])
eps_2d = np.real(eps[2, :, :, 0])  # shape (nx, ny) = (125, 125)

# Full coordinate arrays in nm.
nx, ny = eps_2d.shape
x = (np.arange(nx) - nx / 2 + 0.5) * GRID_SPACING
y = (np.arange(ny) - ny / 2 + 0.5) * GRID_SPACING

# Crop y only — keeps full x so waveguides are included in epsilon.
iy = np.where((y >= DESIGN_YMIN) & (y <= DESIGN_YMAX))[0]
eps_crop = eps_2d[:, iy[0]:iy[-1]+1]
x_crop = x
y_crop = y[iy[0]:iy[-1]+1]

print(f"Cropped region: {eps_crop.shape} pixels, "
      f"x=[{x_crop[0]:.0f},{x_crop[-1]:.0f}] nm, "
      f"y=[{y_crop[0]:.0f},{y_crop[-1]:.0f}] nm")

# Pad all 4 sides with oxide so every shape — including waveguides that exit
# through the x boundaries — is forced closed at the edge.
eps_padded = np.pad(eps_crop, ((1, 1), (1, 1)), mode="constant", constant_values=OX_EPS)
x_padded = np.r_[x_crop[0] - GRID_SPACING, x_crop, x_crop[-1] + GRID_SPACING]
y_padded = np.r_[y_crop[0] - GRID_SPACING, y_crop, y_crop[-1] + GRID_SPACING]

# --- Debug plot ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

im = axes[0].imshow(eps_crop.T, origin="lower",
                    extent=[x_crop[0], x_crop[-1], y_crop[0], y_crop[-1]],
                    cmap="viridis", aspect="auto")
axes[0].set_title("Epsilon (y-cropped, full x)")
axes[0].set_xlabel("x (nm)"); axes[0].set_ylabel("y (nm)")
plt.colorbar(im, ax=axes[0])

axes[1].imshow(eps_crop.T, origin="lower",
               extent=[x_crop[0], x_crop[-1], y_crop[0], y_crop[-1]],
               cmap="viridis", aspect="auto")
cs = plt.contour(x_padded, y_padded, eps_padded.T, levels=[THRESHOLD])
axes[1].contour(x_padded, y_padded, eps_padded.T, levels=[THRESHOLD], colors='r')
axes[1].set_title(f"Contour at threshold={THRESHOLD:.2f}")
axes[1].set_xlabel("x (nm)")

paths = cs.get_paths()
axes[2].set_aspect("equal")
colors = plt.cm.tab10(np.linspace(0, 1, max(len(paths), 1)))
for i, path in enumerate(paths):
    polys = path.to_polygons()
    for poly in polys:
        axes[2].fill(poly[:, 0], poly[:, 1], alpha=0.4, color=colors[i % len(colors)])
        axes[2].plot(poly[:, 0], poly[:, 1], color=colors[i % len(colors)], lw=1)
    print(f"Path {i}: {len(polys)} polygon(s), "
          f"x=[{path.vertices[:,0].min():.0f},{path.vertices[:,0].max():.0f}] nm, "
          f"y=[{path.vertices[:,1].min():.0f},{path.vertices[:,1].max():.0f}] nm")
axes[2].set_xlim(x_crop[0], x_crop[-1])
axes[2].set_ylim(y_crop[0], y_crop[-1])
axes[2].set_title(f"Paths ({len(paths)} total, each colour = 1 path)")
axes[2].set_xlabel("x (nm)")

plt.tight_layout()
plt.savefig("my_results/debug_contour.png", dpi=150)
plt.show()
print(f"\nTotal paths: {len(paths)}")
plt.close("all")

# Write GDS.
cell = gdspy.Cell("WDM2_DESIGN", exclude_from_current=True)
for path in paths:
    for poly in path.to_polygons():
        cell.add(gdspy.Polygon(poly, layer=100))

gdspy.write_gds(
    "my_results/spins_design.gds", [cell],
    unit=1.0e-9,
    precision=1.0e-9)
print("Saved my_results/spins_design.gds")
