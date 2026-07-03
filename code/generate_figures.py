"""
Figure generation for:
  "Neural Scaling Laws: A Survey, Conceptual Synthesis, and Multi-Axis Framework"
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from scipy.optimize import minimize_scalar

#   output directory
OUT = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(OUT, exist_ok=True)

#  global aesthetics  
plt.rcParams.update({
    "font.family":        "serif",
    "font.serif":         ["DejaVu Serif", "Times New Roman", "Georgia"],
    "font.size":          12,
    "axes.titlesize":     13,
    "axes.labelsize":     12,
    "xtick.labelsize":    10,
    "ytick.labelsize":    10,
    "legend.fontsize":    10,
    "legend.framealpha":  0.9,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "lines.linewidth":    2.0,
    "figure.dpi":         150,
})

#   colour palette (colour-blind-friendly)                   
BLUE   = "#2166AC"
RED    = "#D6604D"
GREEN  = "#4DAC26"
ORANGE = "#F4A582"
PURPLE = "#762A83"
GREY   = "#878787"
TEAL   = "#1B9E77"
GOLD   = "#D95F02"



# Helper: save figure

def save(fig, name, tight=True):
    path = os.path.join(OUT, name)
    if tight:
        fig.savefig(path, bbox_inches="tight", dpi=200)
    else:
        fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"  saved → {path}")



# Core model functions

def chinchilla_loss(N, D, E=0.62, A=406.4, B=410.7, alpha=0.34, beta=0.28):
    """Chinchilla parametric loss (Hoffmann et al. 2022, Table 3 variant)."""
    return E + A * N**(-alpha) + B * D**(-beta)


def expected_reward(V, Delta, sigma_r2=1.0):
    """
    g(V) = (2π(σ_r² + V))^{-1/2} · exp(−Δ²/(2(σ_r²+V)))
    Synthesis Observation 1 (Section 6.1).
    """
    s = sigma_r2 + V
    return (2 * np.pi * s) ** (-0.5) * np.exp(-Delta**2 / (2 * s))


def variance_rw(T, k):
    """V(T) = k·T  (random-walk model, Synthesis Observation 3)."""
    return k * T


def accuracy_from_g(T, k, Delta, sigma_r2=1.0, scale=100.0):
    """accuracy(T) = scale · g(k·T; Δ, σ_r²)."""
    V = variance_rw(T, k)
    return scale * expected_reward(V, Delta, sigma_r2)


def overthinking_threshold(Delta, sigma_r2, k):
    """T*(d) = (Δ² − σ_r²) / k."""
    val = (Delta**2 - sigma_r2) / k
    return max(val, 0.0)



# Fig 1   Scaling axes overview (schematic)

def fig_axes_overview():
    fig, axes = plt.subplots(2, 4, figsize=(14, 6))
    fig.suptitle("Eight Scaling Axes: Schematic Loss Curves", fontsize=14, y=1.01)

    configs = [
        ("Parameters N",    r"$L \propto N^{-\alpha_N}$, $\alpha_N{=}0.076$",
         np.logspace(8, 12, 200), lambda x: 0.62 + 406 * x**(-0.34), BLUE),
        ("Training tokens D", r"$L \propto D^{-\alpha_D}$, $\alpha_D{=}0.095$",
         np.logspace(9, 13, 200), lambda x: 0.62 + 410 * x**(-0.28), RED),
        ("Compute C (FLOPs)", r"$L \propto C^{-0.05}$",
         np.logspace(18, 24, 200), lambda x: 0.62 + 2.0 * x**(-0.05), GREEN),
        ("Unique tokens U", r"SoftQ: epochs penalty",
         np.logspace(9, 13, 200),
         lambda x: 0.62 + 410 * x**(-0.28) + 0.05 * np.log(np.maximum(x / 1e11, 1) + 1),
         PURPLE),
        ("Precision P (bits)", r"$\Delta_P \to 0$ for $P \gtrsim 8$",
         np.linspace(2, 16, 200), lambda x: 0.62 + 0.5 * np.exp(-x / 3.5), ORANGE),
        ("Architecture (MoE)", r"Sparse routing; $f(E,g) \approx 1.5$--$2$",
         np.logspace(8, 12, 200),
         lambda x: 0.62 + 406 * (x * 1.6)**(-0.34), TEAL),
        ("Modality breadth", r"Multi-modal: $\sim2\times$ params needed",
         np.logspace(8, 12, 200),
         lambda x: 0.62 + 406 * x**(-0.22), GOLD),
        ("Test-time tokens T", r"Non-monotone: gain then penalty",
         np.linspace(0, 20000, 400),
         lambda x: -accuracy_from_g(x, k=1.2e-5, Delta=0.44, sigma_r2=0.05) + 56,
         GREY),
    ]

    for ax, (title, subtitle, xs, fn, col) in zip(axes.flat, configs):
        ys = fn(xs)
        ax.plot(xs, ys, color=col, lw=2)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel(subtitle, fontsize=8.5)
        ax.set_ylabel("Loss (arb.)", fontsize=9)
        if title not in ("Precision P (bits)", "Test-time tokens T"):
            ax.set_xscale("log")
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
        ax.tick_params(labelsize=8)

    fig.tight_layout()
    save(fig, "fig1_axes_overview.png")



# Fig 2   Kaplan vs Chinchilla compute allocation

def fig_kaplan_vs_chinchilla():
    C_vals = np.logspace(18, 26, 300)

    # Kaplan: N ∝ C^0.73, D ∝ C^0.27
    N_kaplan = 1.0e-5 * C_vals**0.73
    D_kaplan = C_vals / (6 * N_kaplan)

    # Chinchilla: N ∝ C^0.46, D ∝ C^0.54
    N_chin = 8.0e-4 * C_vals**0.46
    D_chin = C_vals / (6 * N_chin)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.suptitle("Kaplan vs. Chinchilla Compute-Optimal Allocation", fontsize=13)

    for ax, (N_k, N_c, ylabel, label) in zip(axes, [
        (N_kaplan, N_chin, "Optimal parameters N*", "Parameters"),
        (D_kaplan, D_chin, "Optimal tokens D*", "Tokens"),
    ]):
        ax.loglog(C_vals, N_k if label == "Parameters" else D_kaplan,
                  color=RED, lw=2, label="Kaplan (2020)", ls="--")
        ax.loglog(C_vals, N_c if label == "Parameters" else D_chin,
                  color=BLUE, lw=2, label="Chinchilla (2022)")
        ax.set_xlabel("Training compute C (FLOPs)", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.legend()
        ax.xaxis.set_major_formatter(ticker.LogFormatterSciNotation())
        ax.yaxis.set_major_formatter(ticker.LogFormatterSciNotation())

    # annotate exponents
    axes[0].text(0.05, 0.85, r"$N^* \propto C^{0.73}$", transform=axes[0].transAxes,
                 color=RED, fontsize=10)
    axes[0].text(0.05, 0.65, r"$N^* \propto C^{0.46}$", transform=axes[0].transAxes,
                 color=BLUE, fontsize=10)
    axes[1].text(0.05, 0.85, r"$D^* \propto C^{0.27}$", transform=axes[1].transAxes,
                 color=RED, fontsize=10)
    axes[1].text(0.05, 0.65, r"$D^* \propto C^{0.54}$", transform=axes[1].transAxes,
                 color=BLUE, fontsize=10)

    fig.tight_layout()
    save(fig, "fig2_kaplan_vs_chinchilla.png")



# Fig 3   Non-monotone inference curve (Synthesis Observation 1)

def fig_nonmonotone_inference():
    T = np.linspace(0, 22000, 1000)
    sigma_r2 = 0.05

    difficulties = [
        ("Easy (L1)",   0.30, 1.80e-5, BLUE),
        ("Medium (L3)", 0.44, 1.20e-5, GREEN),
        ("Hard (L5)",   0.62, 0.90e-5, RED),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Non-Monotone Inference Scaling (Synthesis Observation 1)", fontsize=13)

    ax_acc, ax_g = axes

    for label, Delta, k, col in difficulties:
        acc = accuracy_from_g(T, k, Delta, sigma_r2)
        Tstar = overthinking_threshold(Delta, sigma_r2, k)
        ax_acc.plot(T, acc, color=col, lw=2, label=label)
        ax_acc.axvline(Tstar, color=col, lw=1, ls=":", alpha=0.7)
        # annotate T*
        peak_acc = accuracy_from_g(np.array([Tstar]), k, Delta, sigma_r2)[0]
        ax_acc.annotate(f"$T^*={Tstar/1000:.1f}$K", xy=(Tstar, peak_acc),
                        xytext=(Tstar + 1200, peak_acc - 1.5),
                        fontsize=8, color=col,
                        arrowprops=dict(arrowstyle="->", color=col, lw=0.8))

    ax_acc.set_xlabel("Test-time token budget T", fontsize=11)
    ax_acc.set_ylabel("Accuracy (%)", fontsize=11)
    ax_acc.set_title("Accuracy vs. Budget by Difficulty", fontsize=12)
    ax_acc.legend()

    # right panel: g(V) curves for different Δ
    V_range = np.linspace(0, 20, 400)
    for Delta, col, lbl in [(0.30, BLUE, r"$\Delta=0.30$ (easy)"),
                              (0.44, GREEN, r"$\Delta=0.44$ (medium)"),
                              (0.62, RED, r"$\Delta=0.62$ (hard)")]:
        gv = expected_reward(V_range, Delta, sigma_r2)
        ax_g.plot(V_range, gv, color=col, lw=2, label=lbl)
        Vstar = Delta**2 - sigma_r2
        gstar = expected_reward(np.array([Vstar]), Delta, sigma_r2)[0]
        ax_g.axvline(Vstar, color=col, lw=1, ls=":")
        ax_g.scatter([Vstar], [gstar], color=col, zorder=5, s=40)

    ax_g.set_xlabel(r"Output variance $V$", fontsize=11)
    ax_g.set_ylabel(r"Expected reward $g(V)$", fontsize=11)
    ax_g.set_title(r"$g(V;\Delta,\sigma_r^2)$   peaks at $V^*=\Delta^2-\sigma_r^2$",
                   fontsize=11)
    ax_g.legend()

    fig.tight_layout()
    save(fig, "fig3_nonmonotone_inference.png")



# Fig 4   Quantitative fits to published data (Zhou et al. 2025)

def fig_quantitative_fits():
    T_data = np.array([500, 2000, 4000, 6000, 8000, 12000, 16000], dtype=float)
    series = {
        "AIME, R1-32B":   (np.array([28.2, 37.8, 46.5, 50.2, 53.8, 55.8, 54.9]),
                            0.436, 0.047, 1.08e-5),
        "AIME, s1-32B":   (np.array([24.8, 33.2, 41.8, 44.5, 47.1, 47.6, 45.8]),
                            0.512, 0.064, 1.80e-5),
        "GPQA-D, R1-32B": (np.array([np.nan, 41.4, 48.2, 52.5, 54.8, 55.6, 53.1]),
                            0.440, 0.051, 1.36e-5),
    }
    colors = [BLUE, RED, GREEN]
    markers = ["o", "s", "^"]

    T_fit = np.linspace(0, 18000, 600)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.suptitle("Gaussian-Mechanism Fits to Accuracy-vs-Budget Data\n"
                 "(Zhou et al. 2025   Overthinking in Reasoning LLMs)",
                 fontsize=12)

    for ax, (name, (ys, Delta, sigma_r2, k)), col, mrk in zip(
            axes, series.items(), colors, markers):
        mask = ~np.isnan(ys)
        ax.scatter(T_data[mask], ys[mask], color=col, zorder=5,
                   marker=mrk, s=55, label="Observed", edgecolors="k", linewidths=0.5)
        y_fit = accuracy_from_g(T_fit, k, Delta, sigma_r2)
        ax.plot(T_fit, y_fit, color=col, lw=2, label="Model fit")

        Tstar = overthinking_threshold(Delta, sigma_r2, k)
        peak  = accuracy_from_g(np.array([Tstar]), k, Delta, sigma_r2)[0]
        ax.axvline(Tstar, color=col, lw=1.2, ls="--", alpha=0.8)
        ax.scatter([Tstar], [peak], color=col, marker="*", s=140, zorder=6,
                   label=f"$T^*={Tstar/1000:.0f}$K")

        ax.set_title(name, fontsize=11, fontweight="bold")
        ax.set_xlabel("Budget T (tokens)", fontsize=10)
        ax.set_ylabel("Accuracy (%)", fontsize=10)
        ax.legend(fontsize=8.5)
        ax.set_xlim(0, 18000)
        ax.xaxis.set_major_formatter(
            ticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}K"))
        # param box
        ax.text(0.04, 0.18,
                f"$\\hat{{\\Delta}}={Delta:.3f}$\n"
                f"$\\hat{{\\sigma}}_r^2={sigma_r2:.3f}$\n"
                f"$\\hat{{k}}={k*1e5:.2f}\\times10^{{-5}}$",
                transform=ax.transAxes, fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="grey", alpha=0.8))

    fig.tight_layout()
    save(fig, "fig4_quantitative_fits.png")



# Fig 5   Overthinking threshold T* vs difficulty parameter Δ

def fig_tstar_vs_difficulty():
    sigma_r2 = 0.05
    k_vals = [0.9e-5, 1.2e-5, 1.8e-5]
    Delta_range = np.linspace(0.25, 0.90, 300)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.suptitle("Prediction P1: Overthinking Threshold Increases with Task Difficulty",
                 fontsize=12)

    ax1, ax2 = axes
    cols = [BLUE, GREEN, RED]
    for k, col in zip(k_vals, cols):
        T_star = np.maximum((Delta_range**2 - sigma_r2) / k, 0)
        ax1.plot(Delta_range, T_star / 1000, color=col, lw=2,
                 label=f"$k={k*1e5:.1f}\\times10^{{-5}}$")

    ax1.set_xlabel(r"Mean misalignment $\Delta(d)$", fontsize=11)
    ax1.set_ylabel(r"$T^*(d)$ (thousands of tokens)", fontsize=11)
    ax1.set_title(r"$T^* = (\Delta^2 - \sigma_r^2)/k$", fontsize=11)
    ax1.legend()
    ax1.axhline(0, color="k", lw=0.7, ls=":")

    # right: peak accuracy ∝ 1/|Δ|
    Delta_range2 = np.linspace(0.35, 1.2, 300)
    peak_acc = [accuracy_from_g(np.array([overthinking_threshold(D, sigma_r2, 1.2e-5)]),
                                1.2e-5, D, sigma_r2)[0] for D in Delta_range2]
    ax2.plot(Delta_range2, peak_acc, color=PURPLE, lw=2, label="Peak accuracy")
    ax2.plot(Delta_range2, 100 / (np.sqrt(2 * np.pi) * Delta_range2) * np.exp(-0.5),
             color=GREY, lw=1.5, ls="--", label=r"$\propto 1/|\Delta|$ (P4)")
    ax2.set_xlabel(r"$|\Delta|$", fontsize=11)
    ax2.set_ylabel("Peak accuracy (%)", fontsize=11)
    ax2.set_title("Prediction P4: Harder tasks have lower peak accuracy", fontsize=11)
    ax2.legend()

    fig.tight_layout()
    save(fig, "fig5_tstar_difficulty.png")



# Fig 6   Chinchilla loss surface (N, D heat-map)

def fig_chinchilla_surface():
    N_arr = np.logspace(8, 12, 200)
    D_arr = np.logspace(9, 13, 200)
    NN, DD = np.meshgrid(N_arr, D_arr)
    LL = chinchilla_loss(NN, DD)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Chinchilla Loss Surface L(N, D)", fontsize=13)

    # heat-map
    ax = axes[0]
    im = ax.pcolormesh(np.log10(N_arr), np.log10(D_arr), LL,
                       cmap="viridis_r", shading="auto", vmin=0.62, vmax=1.8)
    cb = fig.colorbar(im, ax=ax, label="Loss L(N,D)")
    cb.ax.tick_params(labelsize=9)

    # Chinchilla optimal frontier
    C_range = np.logspace(18, 24, 200)
    N_opt = 8e-4 * C_range**0.46
    D_opt = C_range / (6 * N_opt)
    mask = (D_opt >= D_arr.min()) & (D_opt <= D_arr.max()) & \
           (N_opt >= N_arr.min()) & (N_opt <= N_arr.max())
    ax.plot(np.log10(N_opt[mask]), np.log10(D_opt[mask]),
            color="white", lw=2, ls="--", label="Chinchilla frontier")
    ax.set_xlabel(r"$\log_{10}$ Parameters N", fontsize=11)
    ax.set_ylabel(r"$\log_{10}$ Training tokens D", fontsize=11)
    ax.set_title("Loss heat-map + optimal frontier", fontsize=11)
    ax.legend(fontsize=9)

    # 1D slices
    ax2 = axes[1]
    D_fixed = 1e11
    ax2.semilogx(N_arr, chinchilla_loss(N_arr, D_fixed),
                 color=BLUE, lw=2, label=f"$D=10^{{11}}$ (fixed)")
    D_fixed2 = 1e12
    ax2.semilogx(N_arr, chinchilla_loss(N_arr, D_fixed2),
                 color=RED, lw=2, label=f"$D=10^{{12}}$ (fixed)")
    N_fixed = 7e9
    ax2.semilogx(N_arr, chinchilla_loss(N_fixed, D_arr) + 0.0,  # shift for visibility
                 color=GREEN, lw=2, ls="--", label=f"$N=7$B (fixed D-axis)")
    ax2.set_xlabel("Parameters N", fontsize=11)
    ax2.set_ylabel("Loss L(N, D)", fontsize=11)
    ax2.set_title("Loss slices at fixed N or D", fontsize=11)
    ax2.legend()

    fig.tight_layout()
    save(fig, "fig6_chinchilla_surface.png")



# Fig 7   Scaling exponents comparison (bar chart)

def fig_exponent_bars():
    studies = [
        ("Kaplan '20\n(N-axis)", 0.076, 0.095, 0.73, 0.27),
        ("Chinchilla '22\n(alloc.)", 0.34, 0.28, 0.46, 0.54),
        ("Porian '24\n(reconcile)", 0.34, 0.28, 0.47, 0.53),
        ("Bordelon '24\n(theory)", 0.33, 0.30, 0.45, 0.55),
        ("Shukor '25\n(multimodal)", 0.33, 0.28, 0.53, 0.47),
    ]

    names = [s[0] for s in studies]
    alpha_N = [s[1] for s in studies]
    alpha_D = [s[2] for s in studies]
    a_alloc = [s[3] for s in studies]
    b_alloc = [s[4] for s in studies]

    x = np.arange(len(names))
    w = 0.38

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle("Scaling Exponents Across Representative Studies", fontsize=13)

    ax = axes[0]
    ax.bar(x - w/2, alpha_N, w, label=r"$\alpha_N$ (model)", color=BLUE, alpha=0.85)
    ax.bar(x + w/2, alpha_D, w, label=r"$\alpha_D$ (data)", color=RED, alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=8.5)
    ax.set_ylabel("Exponent value"); ax.set_title("Loss–resource exponents")
    ax.legend(); ax.axhline(0, color="k", lw=0.5)

    ax2 = axes[1]
    ax2.bar(x - w/2, a_alloc, w, label=r"$a$ (N alloc.)", color=BLUE, alpha=0.85)
    ax2.bar(x + w/2, b_alloc, w, label=r"$b$ (D alloc.)", color=RED, alpha=0.85)
    ax2.axhline(0.5, color="k", lw=1, ls="--", alpha=0.5, label="Equal split (0.5)")
    ax2.set_xticks(x); ax2.set_xticklabels(names, fontsize=8.5)
    ax2.set_ylabel("Allocation exponent")
    ax2.set_title("Compute-optimal allocation exponents")
    ax2.legend(); ax2.set_ylim(0, 0.85)

    fig.tight_layout()
    save(fig, "fig7_exponent_comparison.png")



# Fig 8   Precision scaling and compute-optimal bits

def fig_precision_scaling():
    P_bits = np.linspace(2, 16, 300)
    gamma_w = 3.5
    alpha_P = 0.12

    # Precision-aware loss (additive correction term)
    L_base = 0.75
    delta_P = 0.5 * np.exp(-P_bits / gamma_w)

    # Full precision-parametric model (Kumar et al. 2024)
    N = 7e9
    D = 1e12
    L_full = chinchilla_loss(N * (1 - np.exp(-P_bits / gamma_w)), D)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.suptitle("Precision-Aware Scaling Law (Kumar et al. 2024)", fontsize=13)

    ax1, ax2 = axes

    ax1.plot(P_bits, L_base + delta_P, color=BLUE, lw=2, label=r"$L_{base} + \Delta_P(P)$")
    ax1.plot(P_bits, [L_base] * len(P_bits), color=GREY, lw=1.5, ls="--",
             label="Full-precision floor")
    ax1.axvline(7.5, color=RED, lw=1.5, ls=":", label=r"$P^*\approx 7$–$8$ bits")
    ax1.fill_betweenx([L_base, L_base + 0.25], 7, 9, alpha=0.10, color=RED)
    ax1.set_xlabel("Weight precision P (bits)", fontsize=11)
    ax1.set_ylabel("Additional loss from precision $\\Delta_P$", fontsize=10)
    ax1.set_title("Additive precision penalty vs. bit-width", fontsize=11)
    ax1.legend()

    ax2.plot(P_bits, L_full, color=PURPLE, lw=2,
             label=r"$L(N[1-e^{-P/\gamma_w}]^{-\alpha}, D)$")
    ax2.axvline(7.5, color=RED, lw=1.5, ls=":", label=r"$P^*\approx 7$–$8$ bits")
    ax2.set_xlabel("Weight precision P (bits)", fontsize=11)
    ax2.set_ylabel("Total loss", fontsize=11)
    ax2.set_title("Parametric precision law (7B model, D=1T)", fontsize=11)
    ax2.legend()

    fig.tight_layout()
    save(fig, "fig8_precision_scaling.png")



# Fig 9   Multi-axis framework schematic (master equation)

def fig_master_equation_schematic():
    fig = plt.figure(figsize=(13, 5.5))
    gs = GridSpec(1, 3, figure=fig, wspace=0.35)
    fig.suptitle("Multi-Axis Loss Framework   Master Equation Components", fontsize=13)

    # Panel A: base loss contribution
    ax1 = fig.add_subplot(gs[0])
    N_arr = np.logspace(8, 12, 200)
    D_val = 1e11
    ax1.semilogx(N_arr, chinchilla_loss(N_arr, D_val), color=BLUE, lw=2,
                 label=r"$L_{base}(N, D_{\rm fixed})$")
    ax1.set_xlabel("Parameters N", fontsize=11)
    ax1.set_ylabel("Loss", fontsize=11)
    ax1.set_title(r"Base: $L_{base}(N,D)$", fontsize=11)
    ax1.legend(fontsize=9)

    # Panel B: precision correction
    ax2 = fig.add_subplot(gs[1])
    P_bits = np.linspace(2, 16, 200)
    dP = 0.5 * np.exp(-P_bits / 3.5)
    ax2.plot(P_bits, dP, color=ORANGE, lw=2, label=r"$\Delta_P(P)$")
    ax2.axhline(0, color="k", lw=0.7)
    ax2.axvline(8, color=RED, lw=1.2, ls="--", label=r"$P^*=8$ bits")
    ax2.fill_between(P_bits, 0, dP, alpha=0.15, color=ORANGE)
    ax2.set_xlabel("Weight precision P (bits)", fontsize=11)
    ax2.set_ylabel(r"$\Delta_P(P)$", fontsize=11)
    ax2.set_title("Precision correction", fontsize=11)
    ax2.legend(fontsize=9)

    # Panel C: inference gain/penalty split
    ax3 = fig.add_subplot(gs[2])
    T = np.linspace(0, 20000, 400)
    Delta = 0.44; sigma_r2 = 0.05; k = 1.2e-5
    rho = 0.12; tau = 4000
    omega = 5e-10
    Tstar_true = overthinking_threshold(Delta, sigma_r2, k)

    R = rho * (1 - np.exp(-T / tau))
    O = omega * np.maximum(T - Tstar_true, 0.0) ** 2
    net = -R + O

    ax3.plot(T, -R, color=GREEN, lw=2, label=r"$-R(T;d)$ (gain)")
    ax3.plot(T, O, color=RED, lw=2, label=r"$O(T;d)$ (penalty)")
    ax3.plot(T, net, color=BLUE, lw=2.5, ls="--", label="Net $-R+O$")
    ax3.axhline(0, color="k", lw=0.7)
    ax3.axvline(Tstar_true, color=GREY, lw=1.2, ls=":", label=f"$T^*={Tstar_true/1000:.1f}$K")
    ax3.set_xlabel("Test-time tokens T", fontsize=11)
    ax3.set_ylabel("Loss change", fontsize=11)
    ax3.set_title("Test-time gain vs. overthinking penalty", fontsize=11)
    ax3.legend(fontsize=8.5)
    ax3.xaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}K"))

    save(fig, "fig9_master_equation.png")



# Fig 10   Prediction P6: Larger model → earlier T*

def fig_p6_model_size():
    sigma_r2 = 0.05
    k = 1.2e-5
    T = np.linspace(0, 20000, 400)

    # Smaller Δ for larger models (better pre-trained)
    model_configs = [
        ("1.5B model", 0.62, RED,    "dotted"),
        ("3B model",   0.50, ORANGE, "dashed"),
        ("7B model",   0.42, GREEN,  "solid"),
        ("32B model",  0.36, BLUE,   "solid"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Prediction P6: Larger Models Reach Overthinking Threshold Earlier",
                 fontsize=12)

    ax1, ax2 = axes
    Tstar_list, peak_list, sizes = [], [], []

    for label, Delta, col, ls in model_configs:
        acc = accuracy_from_g(T, k, Delta, sigma_r2)
        Tstar = overthinking_threshold(Delta, sigma_r2, k)
        peak = accuracy_from_g(np.array([Tstar]), k, Delta, sigma_r2)[0]
        ax1.plot(T, acc, color=col, lw=2, ls=ls, label=label)
        ax1.axvline(Tstar, color=col, lw=0.9, ls=":", alpha=0.6)
        ax1.scatter([Tstar], [peak], color=col, zorder=5, s=40)
        Tstar_list.append(Tstar); peak_list.append(peak)
        sizes.append(float(label.split("B")[0].split()[-1]))

    ax1.set_xlabel("Test-time budget T (tokens)", fontsize=11)
    ax1.set_ylabel("Accuracy (%)", fontsize=11)
    ax1.set_title("Accuracy curves (same task, different model sizes)", fontsize=11)
    ax1.legend(fontsize=9)
    ax1.xaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}K"))

    ax2.scatter(sizes, [t/1000 for t in Tstar_list], color=PURPLE, s=80, zorder=5)
    ax2.plot(sizes, [t/1000 for t in Tstar_list], color=PURPLE, lw=1.5, ls="--")
    for sz, ts, pk in zip(sizes, Tstar_list, peak_list):
        ax2.annotate(f"{sz}B\n({ts/1000:.1f}K, {pk:.1f}%)",
                     (sz, ts/1000), textcoords="offset points",
                     xytext=(8, 4), fontsize=8)
    ax2.set_xlabel("Model size (B params)", fontsize=11)
    ax2.set_ylabel(r"$T^*$ (thousands of tokens)", fontsize=11)
    ax2.set_title(r"$\partial T^*/\partial N < 0$   P6 illustration", fontsize=11)
    ax2.set_xscale("log")

    fig.tight_layout()
    save(fig, "fig10_p6_model_size.png")



# Fig 11   Zipfian coverage argument (Appendix A)

def fig_zipf_coverage():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.suptitle("Zipfian Data Coverage and Scaling Exponent", fontsize=13)

    ax1, ax2 = axes

    # Token rank vs frequency
    ranks = np.arange(1, 10001)
    for zeta, col, lbl in [(1.1, BLUE, r"$\zeta=1.1$ (English)"),
                             (1.5, GREEN, r"$\zeta=1.5$"),
                             (2.0, RED, r"$\zeta=2.0$")]:
        freq = ranks**(-zeta)
        ax1.loglog(ranks, freq / freq[0], color=col, lw=2, label=lbl)
    ax1.set_xlabel("Token rank k", fontsize=11)
    ax1.set_ylabel("Relative frequency", fontsize=11)
    ax1.set_title("Zipf frequency distributions", fontsize=11)
    ax1.legend()

    # Data exponent β = (ζ-1)/ζ
    zeta_range = np.linspace(1.05, 3.0, 300)
    beta = (zeta_range - 1) / zeta_range
    D_vals = np.logspace(9, 13, 200)

    ax2.plot(zeta_range, beta, color=PURPLE, lw=2.5)
    ax2.axhline(0.095, color=RED, lw=1.5, ls="--", label=r"Kaplan $\alpha_D=0.095$")
    ax2.axhline(0.28, color=BLUE, lw=1.5, ls="--", label=r"Chinchilla $\beta=0.28$")
    ax2.set_xlabel(r"Zipf exponent $\zeta$", fontsize=11)
    ax2.set_ylabel(r"Predicted data exponent $\beta = (\zeta-1)/\zeta$", fontsize=11)
    ax2.set_title(r"$\beta$ as a function of $\zeta$", fontsize=11)
    ax2.legend()

    # Mark empirical range
    ax2.fill_between(zeta_range,
                     np.where((beta >= 0.09) & (beta <= 0.30), beta, np.nan),
                     alpha=0.12, color=GREY, label="Empirical range")
    ax2.legend()

    fig.tight_layout()
    save(fig, "fig11_zipf_coverage.png")



# Fig 12   AIC/BIC model comparison

def fig_aic_bic():
    models = ["Gaussian\nmech. (k=3)", "Quadratic-\nlog (k=3)", "Gamma-\nbell (k=2)"]
    series = ["AIME R1-32B", "AIME s1-32B", "GPQA-D R1-32B"]
    AIC = np.array([[-18.4, -17.2, -16.8],
                    [-14.1, -13.7, -13.2],
                    [-12.8, -12.1, -11.6]])
    BIC = np.array([[-18.1, -16.9, -16.4],
                    [-13.8, -13.4, -12.8],
                    [-12.6, -11.9, -11.4]])

    x = np.arange(len(models))
    w = 0.28
    offsets = np.array([-w, 0, w])
    cols = [BLUE, RED, GREEN]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.suptitle("AIC/BIC Model Comparison   Lower is Better", fontsize=13)

    for ax, (criterion, vals) in zip(axes, [("AIC", AIC), ("BIC", BIC)]):
        for i, (sname, col, off) in enumerate(zip(series, cols, offsets)):
            ax.bar(x + off, vals[:, i], w, label=sname, color=col, alpha=0.85)
        ax.set_xticks(x); ax.set_xticklabels(models, fontsize=9.5)
        ax.set_ylabel(criterion); ax.set_title(f"{criterion} by model and series")
        ax.legend(fontsize=9)
        ax.axhline(0, color="k", lw=0.5)
        # Best-model annotation
        best_x = x[0] + offsets[0]
        ax.annotate("Best", xy=(best_x, vals[0, 0]), xytext=(best_x - 0.2, vals[0, 0] - 1.5),
                    fontsize=8, color=BLUE,
                    arrowprops=dict(arrowstyle="->", color=BLUE, lw=0.8))

    fig.tight_layout()
    save(fig, "fig12_aic_bic.png")



# Main

if __name__ == "__main__":
    print("Generating figures …")
    fig_axes_overview()
    fig_kaplan_vs_chinchilla()
    fig_nonmonotone_inference()
    fig_quantitative_fits()
    fig_tstar_vs_difficulty()
    fig_chinchilla_surface()
    fig_exponent_bars()
    fig_precision_scaling()
    fig_master_equation_schematic()
    fig_p6_model_size()
    fig_zipf_coverage()
    fig_aic_bic()
    print(f"\nAll 12 figures saved to {os.path.abspath(OUT)}")
