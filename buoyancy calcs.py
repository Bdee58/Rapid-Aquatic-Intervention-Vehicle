import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Unit conversions
# -----------------------------
inch = 0.0254
ft   = 0.3048
lb   = 0.45359237
g    = 0.001   # kg per gram

# -----------------------------
# Constants / assumptions
# -----------------------------
# Coordinate system:
#   x = long axis, origin at geometric center of PVC tube
#   aft is negative x, forward is positive x
#   z = vertical, positive upward
# Ideal CM is at x = 0 and below CB for roll stability.
rho_water = 1000   # kg/m^3
rho_pvc   = 1400   # kg/m^3
rho_al    = 2700   # kg/m^3
rho_pla   = 1240   # kg/m^3  (solid PLA)

# PLA shell infill estimate.  "Pretty low infill" is typically 10–20%;
# we use 15 % here as a central estimate.  The perimeter walls are
# ~2 shells at ~1.2 mm each; we lump everything into one effective
# density multiplier for simplicity.
SHELL_INFILL_FRACTION = 0.15


# -------------------------------------------------------
# Scenario function
# -------------------------------------------------------
def compute_scenario(L_pipe, OD_pipe_in=4.25, include_pla_shell=False):
    """
    Compute mass balance and buoyancy for one torpedo configuration.

    Parameters
    ----------
    L_pipe : float
        Length of the PVC pipe (m).
    OD_pipe_in : float
        Outer diameter of the PVC pipe in inches (default 4.25).
    include_pla_shell : bool
        If True, a 0.5"-thick low-infill PLA 3-D-printed shell is added
        around the full torpedo (pipe + both aft prop assemblies).
    """
    # ------------------------------------------------------------------
    # PVC pipe geometry
    # ------------------------------------------------------------------
    OD_pipe = OD_pipe_in * inch
    wall    = 0.25 * inch
    ID_pipe = OD_pipe - 2 * wall
    R_outer = OD_pipe / 2
    R_inner = ID_pipe / 2

    V_displaced_pipe = np.pi * R_outer**2 * L_pipe
    V_pvc            = np.pi * (R_outer**2 - R_inner**2) * L_pipe
    m_pvc            = rho_pvc * V_pvc

    # ------------------------------------------------------------------
    # Props — two assemblies mounted back-to-back at the aft end of pipe
    # ------------------------------------------------------------------
    m_prop   = 290 * g
    prop_len = 3 * inch          # each prop assembly length along x

    # Aft face of pipe = x = -L_pipe/2
    # Prop 1: directly behind the pipe
    x_prop1 = -L_pipe / 2 - prop_len / 2
    # Prop 2: directly behind prop 1 (back-to-back, no gap)
    x_prop2 = -L_pipe / 2 - prop_len - prop_len / 2

    # ------------------------------------------------------------------
    # Aluminum rod — 75 % extends outside the aft cap, 25 % inside
    # ------------------------------------------------------------------
    rod_OD      = 1   * inch
    rod_L       = 5   * inch
    rod_outside = 0.75 * rod_L   # 3.75 in outside
    rod_inside  = 0.25 * rod_L   # 1.25 in inside

    # Center of rod (uniform mass distribution):
    #   aft end  at x = -L_pipe/2 - rod_outside
    #   fwd end  at x = -L_pipe/2 + rod_inside
    x_rod = 0.5 * ((-L_pipe / 2 - rod_outside) + (-L_pipe / 2 + rod_inside))
    V_rod = np.pi * (rod_OD / 2)**2 * rod_L
    m_rod = rho_al * V_rod

    # ------------------------------------------------------------------
    # Batteries — both pressed against the AFT INTERIOR wall of the pipe.
    # They rest on the bottom of the pipe interior (z = -R_inner).
    # X-positions are measured from the aft wall inward.
    # ------------------------------------------------------------------
    m_big_batt   = 1.5 * lb
    big_L_batt   = 2.7  * inch   # extent along pipe axis
    big_H_batt   = 1.5  * inch   # height (vertical)

    # Aft face of big battery is flush with the aft wall (x = -L_pipe/2).
    x_big_batt = -L_pipe / 2 + big_L_batt / 2
    # Battery rests on pipe floor; center is big_H/2 above the floor.
    z_big_batt = -R_inner + big_H_batt / 2

    m_small_batt  = 38 * g
    small_L_batt  = 4.2 * inch   # extent along pipe axis
    small_H_batt  = 1.2 * inch   # height (vertical)

    # Small battery is also against aft wall, stacked ON TOP of big battery.
    x_small_batt = -L_pipe / 2 + small_L_batt / 2
    z_small_batt = z_big_batt + big_H_batt / 2 + small_H_batt / 2

    # ------------------------------------------------------------------
    # Component list  [name, mass (kg), x (m), z (m)]
    # ------------------------------------------------------------------
    components = [
        ["PVC shell",     m_pvc,        0.0,          0.0          ],
        ["prop 1",        m_prop,       x_prop1,      0.0          ],
        ["prop 2",        m_prop,       x_prop2,      0.0          ],
        ["aluminum rod",  m_rod,        x_rod,        0.0          ],
        ["big battery",   m_big_batt,   x_big_batt,   z_big_batt   ],
        ["small battery", m_small_batt, x_small_batt, z_small_batt ],
    ]

    # ------------------------------------------------------------------
    # Displaced volume / CB — pipe only by default
    # ------------------------------------------------------------------
    V_displaced = V_displaced_pipe
    x_CB        = 0.0
    z_CB        = 0.0
    shell_info  = None

    if include_pla_shell:
        # Shell spans from pipe forward face to the aft face of prop 2.
        shell_fwd     =  L_pipe / 2
        shell_aft     = -L_pipe / 2 - 2 * prop_len   # aft face of prop 2
        shell_L       = shell_fwd - shell_aft
        x_shell       = (shell_fwd + shell_aft) / 2

        shell_t       = 0.5 * inch
        shell_R_inner = R_outer                       # snug fit around pipe OD
        shell_R_outer = shell_R_inner + shell_t

        # Mass uses effective density = infill fraction × solid PLA density.
        V_shell_solid = np.pi * (shell_R_outer**2 - shell_R_inner**2) * shell_L
        rho_shell_eff = rho_pla * SHELL_INFILL_FRACTION
        m_shell       = rho_shell_eff * V_shell_solid
        components.append(["PLA shell", m_shell, x_shell, 0.0])

        # Displaced volume is the full outer cylinder of the shell.
        V_displaced = np.pi * shell_R_outer**2 * shell_L
        x_CB        = x_shell   # centroid of the new outer cylinder
        z_CB        = 0.0
        shell_info  = dict(fwd=shell_fwd, aft=shell_aft,
                           R_inner=shell_R_inner, R_outer=shell_R_outer)

    # ------------------------------------------------------------------
    # Totals
    # ------------------------------------------------------------------
    masses  = np.array([c[1] for c in components])
    xs      = np.array([c[2] for c in components])
    zs      = np.array([c[3] for c in components])

    M_total     = np.sum(masses)
    x_CM        = np.sum(masses * xs) / M_total
    z_CM        = np.sum(masses * zs) / M_total
    M_displaced = rho_water * V_displaced
    net_buoy    = M_displaced - M_total   # positive → buoyant

    # Required ballast to reach neutral buoyancy
    m_ballast = max(net_buoy, 0.0)
    if m_ballast > 0:
        # Solve for x_ballast such that combined x_CM = 0
        x_ballast = -(M_total * x_CM) / m_ballast
        # Place ballast at the bottom of the pipe interior for max stability
        z_ballast = -R_inner
    else:
        x_ballast = None
        z_ballast = None

    return dict(
        L_pipe=L_pipe, R_outer=R_outer, R_inner=R_inner,
        prop_len=prop_len,
        components=components,
        M_total=M_total, x_CM=x_CM, z_CM=z_CM,
        M_displaced=M_displaced, x_CB=x_CB, z_CB=z_CB,
        net_buoy=net_buoy,
        m_ballast=m_ballast, x_ballast=x_ballast, z_ballast=z_ballast,
        include_pla_shell=include_pla_shell,
        shell_info=shell_info,
    )


# -------------------------------------------------------
# Four scenarios
# -------------------------------------------------------
scenarios = [
    dict(
        label="Scenario 1: 24-in PVC pipe, no outer shell",
        result=compute_scenario(24 * inch),
    ),
    dict(
        label="Scenario 2: 12-in PVC pipe, no outer shell",
        result=compute_scenario(12 * inch),
    ),
    dict(
        label=f"Scenario 3: 12-in PVC pipe + 0.5\"-thick PLA shell ({int(SHELL_INFILL_FRACTION*100)}% infill)",
        result=compute_scenario(12 * inch, include_pla_shell=True),
    ),
    dict(
        label=f"Scenario 4: 24-in PVC pipe + 0.5\"-thick PLA shell ({int(SHELL_INFILL_FRACTION*100)}% infill)",
        result=compute_scenario(24 * inch, include_pla_shell=True),
    ),
]


# -------------------------------------------------------
# Print results
# -------------------------------------------------------
for s in scenarios:
    r = s["result"]
    print(f"\n{'='*65}")
    print(f"  {s['label']}")
    print(f"{'='*65}")
    print("\n  Component masses and positions:")
    for name, m, x, z in r["components"]:
        print(f"    {name:16s}: {m*1000:6.1f} g   x = {x/inch:+6.2f} in   z = {z/inch:+5.2f} in")
    print(f"\n  Total mass:              {r['M_total']*1000:.1f} g  ({r['M_total']/lb:.3f} lb)")
    print(f"  Displaced water mass:    {r['M_displaced']*1000:.1f} g  ({r['M_displaced']/lb:.3f} lb)")
    print(f"  Net buoyancy:            {r['net_buoy']*1000:+.1f} g  ({'BUOYANT' if r['net_buoy']>0 else 'HEAVY' if r['net_buoy']<0 else 'NEUTRAL'})")
    print(f"\n  CM: x = {r['x_CM']/inch:+.2f} in,  z = {r['z_CM']/inch:+.3f} in")
    print(f"  CB: x = {r['x_CB']/inch:+.2f} in,  z = {r['z_CB']/inch:+.3f} in")

    if r["net_buoy"] > 0:
        print(f"\n  Vehicle is BUOYANT — add {r['m_ballast']*1000:.1f} g = {r['m_ballast']/lb:.3f} lb of ballast")
        if r["x_ballast"] is not None:
            print(f"  Optimal ballast location: x = {r['x_ballast']/inch:+.2f} in,  z = {r['z_ballast']/inch:+.2f} in")
            if abs(r["x_ballast"]) > r["L_pipe"] / 2:
                print("  WARNING: computed ballast x is outside the pipe length — distribute ballast or place multiple weights")
    elif r["net_buoy"] < 0:
        print(f"\n  Vehicle is HEAVY by {-r['net_buoy']*1000:.1f} g — reduce mass or add buoyant foam")
    else:
        print("\n  Vehicle is NEUTRALLY BUOYANT")

    stable = r["z_CM"] < r["z_CB"]
    print(f"  Roll stability: {'STABLE (CM below CB)' if stable else 'UNSTABLE — CM not below CB'}")


# -------------------------------------------------------
# Plot — one subplot per scenario
# -------------------------------------------------------
fig, axes = plt.subplots(4, 1, figsize=(16, 24))
fig.suptitle("Underwater Torpedo — Mass & Buoyancy Layout Comparison\n"
             f"PLA shell modelled at {int(SHELL_INFILL_FRACTION*100)}% infill",
             fontsize=13, fontweight="bold")

MARKER_COLORS = {
    "PVC shell":     "steelblue",
    "prop 1":        "darkorange",
    "prop 2":        "peru",
    "aluminum rod":  "dimgray",
    "big battery":   "purple",
    "small battery": "mediumpurple",
    "PLA shell":     "saddlebrown",
}

# Component labels are staggered at these heights above the pipe OD (in inches).
# Cycling through them prevents arrow/text collisions between nearby components.
_LEVELS = [2.0, 4.0, 6.0, 3.0, 5.0, 7.0, 8.0]

for ax, s in zip(axes, scenarios):
    r        = s["result"]
    L        = r["L_pipe"]
    Ro       = r["R_outer"]
    Ri       = r["R_inner"]
    si       = r["shell_info"]
    prop_len = r["prop_len"]

    # x-axis extent in inches
    x_aft_in = (si["aft"] if si else -L / 2 - 2 * prop_len) / inch
    x_fwd_in = L / 2 / inch

    # y-axis limits: leave room for staggered labels above and CM/CB below
    y_top = Ro / inch + max(_LEVELS) + 1.5
    y_bot = -Ro / inch - 5.0

    # ------------------------------------------------------------------
    # Pipe body
    # ------------------------------------------------------------------
    pipe_x = np.array([-L / 2, L / 2]) / inch
    ax.fill_between(pipe_x, -Ro / inch, Ro / inch,
                    alpha=0.15, color="steelblue")
    ax.plot(pipe_x, [ Ro / inch,  Ro / inch], color="steelblue", lw=1.5, label="PVC pipe OD")
    ax.plot(pipe_x, [-Ro / inch, -Ro / inch], color="steelblue", lw=1.5)

    # ------------------------------------------------------------------
    # PLA shell outline
    # ------------------------------------------------------------------
    if si is not None:
        sx = np.array([si["aft"], si["fwd"]]) / inch
        ax.fill_between(sx, -si["R_outer"] / inch, si["R_outer"] / inch,
                        alpha=0.10, color="orange")
        ax.plot(sx, [ si["R_outer"] / inch,  si["R_outer"] / inch],
                color="orange", lw=1.5, ls="--", label="PLA shell OD")
        ax.plot(sx, [-si["R_outer"] / inch, -si["R_outer"] / inch],
                color="orange", lw=1.5, ls="--")

    # ------------------------------------------------------------------
    # Prop rectangles (side-view cross-section)
    # ------------------------------------------------------------------
    prop_h_in = 3.5 / 2   # half-height in inches (3.5 in prop OD)
    for xp_m in [r["components"][1][2], r["components"][2][2]]:
        rx = [(xp_m - prop_len / 2) / inch, (xp_m + prop_len / 2) / inch]
        ax.fill_between(rx, -prop_h_in, prop_h_in, alpha=0.22, color="darkorange")
        ax.plot(rx, [ prop_h_in,  prop_h_in], color="darkorange", lw=0.8)
        ax.plot(rx, [-prop_h_in, -prop_h_in], color="darkorange", lw=0.8)

    # ------------------------------------------------------------------
    # Component markers + staggered annotate labels with leader lines
    # ------------------------------------------------------------------
    for idx, (name, m, x, z) in enumerate(r["components"]):
        col  = MARKER_COLORS.get(name, "black")
        xi   = x / inch
        zi   = z / inch
        laby = Ro / inch + _LEVELS[idx % len(_LEVELS)]
        ax.scatter(xi, zi, s=100, color=col, zorder=5)
        ax.annotate(
            f"{name}  ({m * 1000:.0f} g)",
            xy=(xi, zi),
            xytext=(xi, laby),
            fontsize=8.5, ha="center", va="bottom", color=col,
            arrowprops=dict(arrowstyle="-", color=col, lw=0.9,
                            shrinkA=0, shrinkB=4),
            zorder=6,
        )

    # ------------------------------------------------------------------
    # CM marker + label below pipe
    # ------------------------------------------------------------------
    cm_xi = r["x_CM"] / inch
    cm_zi = r["z_CM"] / inch
    ax.scatter(cm_xi, cm_zi, s=300, marker="x", color="red",
               linewidths=2.5, zorder=8, label="CM")
    ax.annotate(
        f"CM\nx = {cm_xi:+.2f} in\nz = {cm_zi:+.3f} in",
        xy=(cm_xi, cm_zi),
        xytext=(cm_xi - 0.5, -Ro / inch - 1.5),
        fontsize=8.5, fontweight="bold", color="red", ha="center", va="top",
        arrowprops=dict(arrowstyle="->", color="red", lw=1.3),
        zorder=8,
    )

    # ------------------------------------------------------------------
    # CB marker + label below pipe (offset right to avoid CM text)
    # ------------------------------------------------------------------
    cb_xi = r["x_CB"] / inch
    cb_zi = r["z_CB"] / inch
    ax.scatter(cb_xi, cb_zi, s=260, marker="o", facecolors="none",
               edgecolors="navy", linewidths=2.2, zorder=8, label="CB")
    ax.annotate(
        f"CB\nx = {cb_xi:+.2f} in",
        xy=(cb_xi, cb_zi),
        xytext=(cb_xi + 2.5, -Ro / inch - 3.0),
        fontsize=8.5, fontweight="bold", color="navy", ha="center", va="top",
        arrowprops=dict(arrowstyle="->", color="navy", lw=1.3),
        zorder=8,
    )

    # ------------------------------------------------------------------
    # Geometric centre reference line
    # ------------------------------------------------------------------
    ax.axvline(0, linestyle=":", linewidth=1.0, color="gray", zorder=1)
    ax.text(0.4, Ro / inch + 0.5, "x = 0 (ideal)",
            fontsize=7.5, color="gray", ha="left", va="bottom")

    # ------------------------------------------------------------------
    # Ballast marker + label
    # ------------------------------------------------------------------
    if r["m_ballast"] > 0 and r["x_ballast"] is not None:
        bx = r["x_ballast"] / inch
        bz = r["z_ballast"] / inch
        ax.scatter(bx, bz, s=220, marker="s", color="green", zorder=8,
                   label=f"ballast req. ({r['m_ballast']*1000:.0f} g)")
        ax.annotate(
            f"ballast\n{r['m_ballast'] * 1000:.0f} g\nx = {bx:+.2f} in",
            xy=(bx, bz),
            xytext=(bx + 2.0, -Ro / inch - 1.5),
            fontsize=8.5, color="green", ha="center", va="top",
            arrowprops=dict(arrowstyle="->", color="green", lw=1.2),
            zorder=8,
        )

    # ------------------------------------------------------------------
    # Title, limits, labels
    # ------------------------------------------------------------------
    status = ("BUOYANT" if r["net_buoy"] > 0 else
              "HEAVY"   if r["net_buoy"] < 0 else "NEUTRAL")
    info = (f"M_sys = {r['M_total']*1000:.0f} g   "
            f"M_disp = {r['M_displaced']*1000:.0f} g   "
            f"net = {r['net_buoy']*1000:+.0f} g  [{status}]")
    ax.set_title(f"{s['label']}\n{info}", fontsize=9, loc="left", pad=6)

    ax.set_xlim(x_aft_in - 2.5, x_fwd_in + 3.0)
    ax.set_ylim(y_bot, y_top)
    ax.set_xlabel("x — along torpedo axis  (forward →)  [inches]", fontsize=8)
    ax.set_ylabel("z — vertical  [inches]", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="lower right", ncol=2, framealpha=0.85)

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.show()
