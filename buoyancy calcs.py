import numpy as np
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────
# Unit conversions
# ─────────────────────────────────────────────────────────────
inch = 0.0254        # m
mm   = 0.001         # m
lb   = 0.45359237    # kg
g_   = 0.001         # kg per gram

# ─────────────────────────────────────────────────────────────
# Material densities  (kg/m³)
# ─────────────────────────────────────────────────────────────
rho_water = 1000
rho_pvc   = 1400
rho_al    = 2700

# ─────────────────────────────────────────────────────────────
# Geometry — all quantities in metres
# ─────────────────────────────────────────────────────────────
# Forward (main) section: 929 mm long, 4.5 in OD, 0.25 in PVC wall
L_fwd     = 929 * mm
OD        = 4.5 * inch          # 0.11430 m
wall      = 0.25 * inch         # 0.00635 m
ID_main   = OD - 2 * wall       # 0.10160 m
R_OD      = OD / 2              # 0.05715 m
R_ID_main = ID_main / 2         # 0.05080 m

# Aft section: 178 mm long, 4.5 in OD, 100 mm ID (prop/motor shroud)
# The centre bore is open to water, so only the annular wall displaces.
L_aft    = 178 * mm
R_ID_aft = 50 * mm              # 100 mm ID → 50 mm radius

L_total  = L_fwd + L_aft        # 1107 mm ≈ 43.6 in

# ─────────────────────────────────────────────────────────────
# Coordinate system
#   Origin at mid-length of full vehicle
#   +x = forward (nose),  −x = aft (tail)
# ─────────────────────────────────────────────────────────────
x_nose      =  L_total / 2          # +0.5535 m
x_junction  =  x_nose - L_fwd       # −0.3755 m  (fwd/aft section boundary)
x_tail      = -L_total / 2          # −0.5535 m

x_ctr_fwd   = (x_nose + x_junction) / 2   # centroid of fwd section
x_ctr_aft   = (x_junction + x_tail) / 2   # centroid of aft section

# ─────────────────────────────────────────────────────────────
# Displaced volumes and Center of Buoyancy
# ─────────────────────────────────────────────────────────────
V_disp_fwd = np.pi * R_OD**2 * L_fwd
V_disp_aft = np.pi * (R_OD**2 - R_ID_aft**2) * L_aft   # annular — centre open
V_disp     = V_disp_fwd + V_disp_aft

x_CB = (V_disp_fwd * x_ctr_fwd + V_disp_aft * x_ctr_aft) / V_disp
z_CB = 0.0

# ─────────────────────────────────────────────────────────────
# Component masses and positions
# ─────────────────────────────────────────────────────────────

# PVC main pipe wall
m_pvc_fwd   = rho_pvc * np.pi * (R_OD**2 - R_ID_main**2) * L_fwd

# PVC aft section wall
m_pvc_aft   = rho_pvc * np.pi * (R_OD**2 - R_ID_aft**2 ) * L_aft

# Aluminium heatsink (110 400 mm³) — electronics bay, aft end of main section
m_heatsink  = rho_al * 110_400e-9   # ≈ 298 g
x_heatsink  = x_junction + 80 * mm # 80 mm forward of aft junction

# Big battery — aft interior of main pipe, flat against aft wall, on the floor
m_big_batt  = 1.5 * lb             # ≈ 680 g
big_L_batt  = 2.7 * inch
big_H_batt  = 1.5 * inch
x_big_batt  = x_junction + big_L_batt / 2
z_big_batt  = -R_ID_main + big_H_batt / 2

# Small battery — also aft, stacked on top of big battery
m_small_batt = 38 * g_
small_L_batt = 4.2 * inch
small_H_batt = 1.2 * inch
x_small_batt = x_junction + small_L_batt / 2
z_small_batt = z_big_batt + big_H_batt / 2 + small_H_batt / 2

# Two prop assemblies behind the aft section
m_prop   = 290 * g_
prop_len = 3 * inch
x_prop1  = x_tail - prop_len / 2
x_prop2  = x_tail - prop_len - prop_len / 2

# ─────────────────────────────────────────────────────────────
# Component table  [name, mass (kg), x (m), z (m)]
# ─────────────────────────────────────────────────────────────
components = [
    ["PVC main pipe",   m_pvc_fwd,    x_ctr_fwd,    0.0         ],
    ["PVC aft section", m_pvc_aft,    x_ctr_aft,    0.0         ],
    ["Al heatsink",     m_heatsink,   x_heatsink,   0.0         ],
    ["big battery",     m_big_batt,   x_big_batt,   z_big_batt  ],
    ["small battery",   m_small_batt, x_small_batt, z_small_batt],
    ["prop 1",          m_prop,       x_prop1,      0.0         ],
    ["prop 2",          m_prop,       x_prop2,      0.0         ],
]

masses = np.array([c[1] for c in components])
xs     = np.array([c[2] for c in components])
zs     = np.array([c[3] for c in components])

M_total  = np.sum(masses)
x_CM_dry = np.sum(masses * xs) / M_total
z_CM_dry = np.sum(masses * zs) / M_total
M_disp   = rho_water * V_disp
net_buoy = M_disp - M_total   # positive → buoyant

# ─────────────────────────────────────────────────────────────
# Ballast: fixed at 10 in from nose (inside pipe floor).
# Compute required mass for neutral buoyancy and resulting CM.
# ─────────────────────────────────────────────────────────────
x_ballast_offset = 10 * inch               # 10 in from nose face
x_ballast = x_nose - x_ballast_offset      # in coordinate frame

if net_buoy > 0:
    m_ballast = net_buoy                    # exact mass for neutral buoyancy
    z_ballast = -R_ID_main                  # resting on pipe floor
    x_CM_final = (M_total * x_CM_dry + m_ballast * x_ballast) / M_disp
    z_CM_final = (M_total * z_CM_dry + m_ballast * z_ballast) / M_disp
elif net_buoy < 0:
    m_ballast  = 0.0
    z_ballast  = None
    x_CM_final = x_CM_dry
    z_CM_final = z_CM_dry
else:
    m_ballast  = 0.0
    z_ballast  = None
    x_CM_final = x_CM_dry
    z_CM_final = z_CM_dry

# ─────────────────────────────────────────────────────────────
# Console summary
# ─────────────────────────────────────────────────────────────
def i(x): return x / inch   # metres → inches helper

print(f"\n{'='*70}")
print(f"  Torpedo Mass & Buoyancy")
print(f"  Fwd: {L_fwd*1000:.0f} mm × ⌀{OD/inch:.2f}\" OD   "
      f"Aft: {L_aft*1000:.0f} mm × ⌀{OD/inch:.2f}\" OD / {R_ID_aft*2*1000:.0f} mm ID")
print(f"  Total length: {L_total*1000:.0f} mm = {i(L_total):.2f} in")
print(f"{'='*70}")

print("\n  Component masses and positions:")
for name, m, x, z in components:
    print(f"    {name:18s}: {m*1000:7.1f} g   "
          f"x = {i(x):+7.2f} in   z = {i(z):+5.2f} in")

print(f"\n  Total dry mass:        {M_total*1000:8.1f} g  ({M_total/lb:.3f} lb)")
print(f"  Displaced water mass:  {M_disp *1000:8.1f} g  ({M_disp /lb:.3f} lb)")
status = ("BUOYANT" if net_buoy > 0 else ("HEAVY" if net_buoy < 0 else "NEUTRAL"))
print(f"  Net buoyancy (dry):    {net_buoy*1000:+8.1f} g  [{status}]")
print(f"\n  Dry CM:  x = {i(x_CM_dry):+.2f} in,  z = {i(z_CM_dry):+.3f} in")
print(f"  CB:      x = {i(x_CB):+.2f} in,  z = {i(z_CB):+.3f} in")

if m_ballast > 0:
    print(f"\n  ── Ballast (neutral buoyancy, fixed 10 in from nose) ───")
    print(f"  Required ballast:   {m_ballast*1000:7.1f} g  =  {m_ballast/lb:.3f} lb")
    print(f"  Ballast position:   {i(x_ballast_offset):.1f} in from nose  (x = {i(x_ballast):+.2f} in, floor of pipe)")
    print(f"  Final CM:           x = {i(x_CM_final):+.2f} in,  z = {i(z_CM_final):+.3f} in")
    print(f"  CB:                 x = {i(x_CB):+.2f} in")
    lead = x_CM_final - x_CB
    print(f"  CM vs CB (pitch):   CM is {abs(i(lead)):.2f} in "
          f"{'FORWARD' if lead > 0 else 'AFT'} of CB")
    yaw_lever = x_CM_final - x_tail
    print(f"  Yaw lever arm:      {i(yaw_lever):.2f} in  (CM to tail face)")
elif net_buoy < 0:
    print(f"\n  Vehicle HEAVY by {-net_buoy*1000:.1f} g — reduce mass or add foam")

stable = z_CM_final < z_CB
print(f"  Roll stability: {'STABLE  (CM below CB)' if stable else 'UNSTABLE — CM not below CB'}")
print()

# ─────────────────────────────────────────────────────────────
# Plot — single figure, side-view layout
# ─────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(18, 8))

# ── Main pipe (solid cylinder, forward section) ──────────────
pipe_xs = [i(x_junction), i(x_nose)]
ax.fill_between(pipe_xs, -i(R_OD), i(R_OD), alpha=0.15, color="steelblue")
ax.plot(pipe_xs, [ i(R_OD),  i(R_OD)], color="steelblue", lw=2.0, label="Main pipe OD")
ax.plot(pipe_xs, [-i(R_OD), -i(R_OD)], color="steelblue", lw=2.0)
ax.plot([i(x_nose),      i(x_nose)     ], [-i(R_OD),  i(R_OD)], color="steelblue", lw=2.0)
ax.plot([i(x_junction),  i(x_junction) ], [-i(R_OD),  i(R_OD)], color="steelblue", lw=1.0, ls="--")

# ── Aft section (annular — outer wall + inner bore outline) ──
aft_xs = [i(x_tail), i(x_junction)]
ax.fill_between(aft_xs, -i(R_OD),     i(R_OD),     alpha=0.10, color="dimgray")
ax.fill_between(aft_xs, -i(R_ID_aft), i(R_ID_aft), alpha=0.30, color="white")
ax.plot(aft_xs, [ i(R_OD),      i(R_OD)     ], color="dimgray", lw=2.0, label="Aft section OD")
ax.plot(aft_xs, [-i(R_OD),     -i(R_OD)     ], color="dimgray", lw=2.0)
ax.plot(aft_xs, [ i(R_ID_aft),  i(R_ID_aft) ], color="dimgray", lw=1.2, ls=":", label="Aft section ID (100 mm)")
ax.plot(aft_xs, [-i(R_ID_aft), -i(R_ID_aft) ], color="dimgray", lw=1.2, ls=":")
# End cap of aft section (annular face)
ax.plot([i(x_tail), i(x_tail)], [-i(R_OD), -i(R_ID_aft)], color="dimgray", lw=2.0)
ax.plot([i(x_tail), i(x_tail)], [ i(R_ID_aft),  i(R_OD)], color="dimgray", lw=2.0)

# ── Prop rectangles ──────────────────────────────────────────
prop_h = 3.5 / 2   # half-height in inches (3.5 in prop OD)
for xp_m in [x_prop1, x_prop2]:
    rx = [i(xp_m - prop_len / 2), i(xp_m + prop_len / 2)]
    ax.fill_between(rx, -prop_h, prop_h, alpha=0.22, color="darkorange")
    ax.plot(rx, [ prop_h,  prop_h], color="darkorange", lw=0.8)
    ax.plot(rx, [-prop_h, -prop_h], color="darkorange", lw=0.8)

# ── Component markers + manually spaced labels ───────────────
COLORS = {
    "PVC main pipe":   "steelblue",
    "PVC aft section": "dimgray",
    "Al heatsink":     "slategray",
    "big battery":     "purple",
    "small battery":   "mediumpurple",
    "prop 1":          "darkorange",
    "prop 2":          "peru",
}
# (label_x_offset_in, label_y_above_OD_in) — hand-tuned to prevent overlaps
LABEL_POS = {
    "PVC main pipe":   ( 0.0,  7.0),
    "PVC aft section": ( 0.0,  4.5),
    "Al heatsink":     ( 3.5,  4.5),
    "big battery":     (-3.0,  4.5),
    "small battery":   ( 0.0,  2.0),
    "prop 1":          ( 1.5,  4.5),
    "prop 2":          (-1.5,  2.0),
}

for name, m, x, z in components:
    col = COLORS.get(name, "black")
    xi, zi = i(x), i(z)
    xoff, yoff = LABEL_POS.get(name, (0, 4.0))
    ax.scatter(xi, zi, s=100, color=col, zorder=5)
    ax.annotate(
        f"{name}\n({m*1000:.0f} g)",
        xy=(xi, zi), xytext=(xi + xoff, i(R_OD) + yoff),
        fontsize=8, ha="center", va="bottom", color=col,
        arrowprops=dict(arrowstyle="-", color=col, lw=0.8, shrinkA=0, shrinkB=3),
        zorder=6,
    )

# ── Key markers: vertical guide lines ────────────────────────
ax.axvline(i(x_CB),     color="navy",   lw=1.0, ls=":",  alpha=0.6, zorder=2)
ax.axvline(i(x_CM_dry), color="tomato", lw=1.0, ls="--", alpha=0.5, zorder=2)
if m_ballast > 0:
    ax.axvline(i(x_ballast),  color="green", lw=1.0, ls="--", alpha=0.4, zorder=2)
    ax.axvline(i(x_CM_final), color="red",   lw=1.4, ls="-",  alpha=0.7, zorder=2)

# ── Scatter markers ───────────────────────────────────────────
ax.scatter(i(x_CM_dry), i(z_CM_dry), s=160, marker="x", color="tomato",
           linewidths=2.0, zorder=9, label=f"Dry CM  x={i(x_CM_dry):+.2f}\"")
ax.scatter(i(x_CB), i(z_CB), s=200, marker="o", facecolors="none",
           edgecolors="navy", linewidths=2.0, zorder=9, label=f"CB  x={i(x_CB):+.2f}\"")
if m_ballast > 0:
    ax.scatter(i(x_ballast), i(z_ballast), s=220, marker="s", color="green",
               zorder=10, label=f"Ballast  {m_ballast*1000:.0f} g @ 10\" from nose")
    ax.scatter(i(x_CM_final), i(z_CM_final), s=260, marker="x", color="red",
               linewidths=2.5, zorder=10, label=f"Final CM  x={i(x_CM_final):+.2f}\"")

# ── Info box (replaces all the bottom arrow annotations) ──────
lead = x_CM_final - x_CB if m_ballast > 0 else 0
info = (
    f"  Dry mass:   {M_total*1000:.0f} g  ({M_total/lb:.2f} lb)\n"
    f"  Displaced:  {M_disp*1000:.0f} g  ({M_disp/lb:.2f} lb)\n"
    f"  Net (dry):  {net_buoy*1000:+.0f} g\n"
)
if m_ballast > 0:
    info += (
        f"\n"
        f"  Ballast:    {m_ballast*1000:.0f} g  ({m_ballast/lb:.2f} lb)\n"
        f"  @ 10 in from nose\n"
        f"\n"
        f"  Final CM:   x = {i(x_CM_final):+.2f} in\n"
        f"              z = {i(z_CM_final):+.3f} in\n"
        f"  CB:         x = {i(x_CB):+.2f} in\n"
        f"  CM vs CB:   {abs(i(lead)):.2f} in {'fwd' if lead > 0 else 'aft'}\n"
        f"  Yaw lever:  {i(x_CM_final - x_tail):.1f} in"
    )
ax.text(0.015, 0.97, info, transform=ax.transAxes,
        fontsize=8.5, va="top", ha="left", family="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow",
                  edgecolor="gray", alpha=0.9), zorder=11)

# ── Yaw lever arm ─────────────────────────────────────────────
if m_ballast > 0:
    yaw_y = -i(R_OD) - 0.5
    ax.annotate("", xy=(i(x_tail), yaw_y), xytext=(i(x_CM_final), yaw_y),
                arrowprops=dict(arrowstyle="<->", color="darkgreen", lw=1.4))
    ax.text((i(x_CM_final) + i(x_tail)) / 2, yaw_y - 0.2,
            f"yaw lever  {i(x_CM_final - x_tail):.1f} in",
            fontsize=8, color="darkgreen", ha="center", va="top")

# ── Geometric centre reference ────────────────────────────────
ax.axvline(0, linestyle=":", linewidth=0.8, color="gray", zorder=1)
ax.text(0.2, i(R_OD) + 0.3, "mid", fontsize=7, color="gray", ha="left", va="bottom")

# ── Title, labels, limits ─────────────────────────────────────
status_str = "BUOYANT" if net_buoy > 0 else ("HEAVY" if net_buoy < 0 else "NEUTRAL")
ax.set_title(f"Torpedo Mass & Buoyancy Layout   "
             f"[{status_str}  dry={M_total*1000:.0f} g  disp={M_disp*1000:.0f} g  net={net_buoy*1000:+.0f} g]",
             fontsize=10, fontweight="bold", loc="left", pad=8)

x_plot_aft = i(x_prop2 - prop_len / 2) - 2.0
x_plot_fwd = i(x_nose) + 2.5
y_top = i(R_OD) + 10.0
y_bot = -i(R_OD) - 2.5

ax.set_xlim(x_plot_aft, x_plot_fwd)
ax.set_ylim(y_bot, y_top)
ax.set_xlabel("x — torpedo axis (forward →)  [inches]", fontsize=9)
ax.set_ylabel("z — vertical  [inches]", fontsize=9)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=8.5, loc="lower right", framealpha=0.9)

plt.tight_layout()
plt.show()
