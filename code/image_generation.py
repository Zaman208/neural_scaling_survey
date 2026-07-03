import json, os, warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
warnings.filterwarnings("ignore")

ROOT   = os.path.join(os.path.dirname(__file__), "..")
MERGED = os.path.join(ROOT, "results", "merged", "all_results_merged.json")
FIGS   = os.path.join(ROOT, "figures")
os.makedirs(FIGS, exist_ok=True)

C = dict(
    blue   = "#2166AC", red    = "#D6604D", green  = "#4DAC26",
    purple = "#762A83", teal   = "#1B9E77", grey   = "#888888",
    lemon  = "#CC8800",
)
L1_COL = C["blue"]; L3_COL = C["teal"]; L5_COL = C["red"]
PHI_COL = "#1B6CA8"; QWEN_COL = "#6A1B9A"

plt.rcParams.update({
    "font.family": "DejaVu Serif", "font.size": 11,
    "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.labelsize": 10, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "legend.fontsize": 8.5, "legend.framealpha": 0.9,
    "legend.edgecolor": "#CCCCCC",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linestyle": "--",
    "lines.linewidth": 2.0, "figure.dpi": 150,
})

def load():
    with open(MERGED) as f:
        return json.load(f)

def g_scaled(T, Delta, sigma_r2, k, scale, offset):
    T   = np.asarray(T, dtype=float)
    V   = np.maximum(k * T, 0.0)
    s   = np.maximum(sigma_r2 + V, 1e-12)
    raw = np.exp(-Delta**2 / (2.0 * s)) / np.sqrt(2.0 * np.pi * s)
    return scale * raw + offset


# Figure 1 — 6-panel P1–P6 overview
def figure_summary_predictions(d):
    expA = d["exp_A_multimodel"]; expB = d["exp_B_p6_scaling"]
    expC = d["exp_C_hurst"];      expD = d["exp_D_precision"]
    expE = d["exp_E_prospective"]
    LEVELS  = ["Level_1", "Level_3", "Level_5"]
    LLABELS = ["L1 (easy)", "L3 (med.)", "L5 (hard)"]

    fig = plt.figure(figsize=(15, 9.5)); fig.patch.set_facecolor("white")
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.55, wspace=0.40,
                            top=0.90, bottom=0.09, left=0.07, right=0.97)

    # (a) T* by difficulty
    ax = fig.add_subplot(gs[0, 0])
    p1 = expA.get("P1_per_model", {})
    x  = np.arange(3); w = 0.35
    phi_ts  = [float(p1.get("phi35",  {}).get(lv, np.nan) or np.nan) for lv in LEVELS]
    qwen_ts = [float(p1.get("qwen3b", {}).get(lv, np.nan) or np.nan) for lv in LEVELS]
    ax.bar(x - w/2, phi_ts,  w, color=PHI_COL,  label="Phi-3.5-mini (3.8B)", alpha=0.88)
    ax.bar(x + w/2, qwen_ts, w, color=QWEN_COL, label="Qwen2.5-3B (3.0B)",   alpha=0.88)
    ax.set_xticks(x); ax.set_xticklabels(LLABELS)
    ax.set_ylabel(r"Optimal budget $T^*$ (tokens)")
    ax.set_title("(a)  $T^*$ increases with task difficulty")
    ax.legend(loc="upper left")
    vals_ab = [v for v in phi_ts + qwen_ts if not np.isnan(v)]
    if vals_ab: ax.set_ylim(0, max(vals_ab) * 1.30)

    # (b) quantisation
    ax  = fig.add_subplot(gs[0, 1])
    p3  = expD.get("P3_tests", {})
    PK  = ["fp16", "bf16", "int8", "int4"]
    PL  = ["FP16", "BF16", "INT8", "INT4"]
    xd  = np.arange(4); wd = 0.22; all_v = []
    for i, (lv, col, lbl) in enumerate(zip(LEVELS, [L1_COL, L3_COL, L5_COL], LLABELS)):
        ts = p3.get(lv, {}).get("T_stars", {})
        vs = [float(ts.get(p, np.nan) or np.nan) for p in PK]
        ax.bar(xd + (i-1)*wd, vs, wd, color=col, alpha=0.88, label=lbl)
        all_v += [v for v in vs if not np.isnan(v)]
    ax.set_xticks(xd); ax.set_xticklabels(PL)
    ax.set_ylabel(r"$T^*$ (tokens)")
    ax.set_title("(b)  Quantisation reduces optimal compute budget")
    ax.legend()
    if all_v:
        cap = np.percentile(all_v, 88) * 1.35
        ax.set_ylim(0, cap)
        ax.text(3.0, cap * 0.90, "INT4 bars truncated above",
                ha="center", fontsize=7.5, color=C["grey"], style="italic")

    # (c) peak accuracy
    ax = fig.add_subplot(gs[0, 2])
    for mkey, col, lbl in [("phi35", PHI_COL, "Phi-3.5-mini"),
                            ("qwen3b", QWEN_COL, "Qwen2.5-3B")]:
        raw = expA.get("models", {}).get(mkey, {})
        peaks = []
        for lv in LEVELS:
            ser = raw.get(lv, [])
            peaks.append(max((pt["accuracy"]*100 for pt in ser), default=np.nan) if ser else np.nan)
        ax.plot(LLABELS, peaks, "o-", color=col, lw=2, ms=7, label=lbl)
    ax.set_ylabel("Peak accuracy (%)"); ax.set_ylim(55, 108)
    ax.set_title("(c)  Peak accuracy decreases with difficulty")
    ax.legend(); ax.axhline(100, color=C["grey"], lw=0.8, ls="--", alpha=0.4)

    # (d) Hurst exponent
    ax = fig.add_subplot(gs[1, 0])
    ax.axhline(0.5, color=C["red"], lw=1.5, ls="--",
               label=r"$H=0.5$ (random walk)", alpha=0.8)
    ax.axhspan(0.4, 0.6, alpha=0.07, color=C["red"])
    for mkey, col, lbl in [("phi35", PHI_COL, "Phi-3.5-mini"),
                            ("qwen3b", QWEN_COL, "Qwen2.5-3B")]:
        mdc = expC.get(mkey, {}); hs = []
        for lv in LEVELS:
            ser = mdc.get(lv, {}).get("series", [])
            if len(ser) > 2:
                bts = np.array([pt.get("budget", np.nan) for pt in ser], dtype=float)
                vts = np.array([pt.get("V_ent",  np.nan) for pt in ser])
                mk  = ~np.isnan(vts) & (vts > 0) & (bts > 0)
                if mk.sum() > 2:
                    sl, _ = np.polyfit(np.log(bts[mk]), np.log(vts[mk]), 1)
                    hs.append(sl / 2); continue
            hs.append(np.nan)
        ax.plot(LLABELS, hs, "o-", color=col, lw=1.8, ms=6, label=lbl)
    ax.set_ylabel("Hurst exponent $H$"); ax.set_ylim(-0.2, 1.1)
    ax.set_title(r"(d)  Token-budget variance follows $H \approx 0.5$")
    ax.legend()

    # (e) T* vs model size
    ax = fig.add_subplot(gs[1, 1])
    p6 = expB.get("P6_tests", {})
    MP = {"qwen_1b5": 1500, "qwen_3b": 3000}
    for lv, col, lbl_lv in zip(LEVELS, [L1_COL, L3_COL, L5_COL], LLABELS):
        tsd = p6.get(lv, {}).get("T_stars", {})
        pm  = p6.get(lv, {}).get("param_Ms", MP)
        xs  = [np.log10(pm.get(mk, np.nan)) for mk in ["qwen_1b5","qwen_3b"] if mk in tsd]
        ys  = [tsd[mk] for mk in ["qwen_1b5","qwen_3b"] if mk in tsd]
        if xs: ax.plot(xs, ys, "o-", color=col, lw=2, ms=7, label=lbl_lv)
    ax.set_xlabel(r"$\log_{10}(N_\mathrm{params})$")
    ax.set_ylabel(r"$T^*$ (tokens)")
    ax.set_title(r"(e)  Larger models reach peak earlier ($\partial T^*/\partial N < 0$)")
    ax.legend()

    # (f) Prospective R^2
    ax = fig.add_subplot(gs[1, 2]); has_data = False
    for i, (mkey, col, lbl) in enumerate([("phi35", PHI_COL, "Phi-3.5-mini"),
                                           ("qwen3b", QWEN_COL, "Qwen2.5-3B")]):
        mde = expE.get(mkey, {}); r2s = []; xpos = []
        for j, lv in enumerate(LEVELS):
            r2 = mde.get(lv, {}).get("eval", {}).get("R2_test", np.nan)
            if isinstance(r2, float) and not np.isnan(r2):
                r2s.append(max(r2, -2.0)); xpos.append(j + (0.18 if i==0 else -0.18))
                has_data = True
        if r2s: ax.bar(xpos, r2s, 0.32, color=col, alpha=0.85, label=lbl)
    if not has_data:
        ax.text(0.5, 0.5, "R^2 data unavailable (all NaN)",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=9, color=C["grey"], style="italic")
    ax.axhline(0, color="k", lw=0.8, ls="--", alpha=0.5)
    ax.axhline(1, color=C["green"], lw=0.8, ls=":", alpha=0.5, label=r"$R^2=1$")
    ax.set_xticks(range(3)); ax.set_xticklabels(LLABELS)
    ax.set_ylabel(r"$R^2$ on held-out test budgets")
    ax.set_title("(f)  Prospective prediction quality (g-model)")
    ax.legend()

    fig.text(0.5, 0.965, "Validation of falsifiable predictions P1-P6 across Experiments A-E",
             ha="center", va="top", fontsize=13, fontweight="bold")
    fig.text(0.5, 0.942,
             "Models: Phi-3.5-mini (3.8B), Qwen2.5-3B (3.0B)  |  AIME 2024/2025  |  8-bit quantisation  |  2x T4 GPUs",
             ha="center", va="top", fontsize=9.5, color=C["grey"])

    out = os.path.join(FIGS, "figure_summary_predictions.png")
    fig.savefig(out, bbox_inches="tight", dpi=200); plt.close(fig)
    print(f"  saved: {out}")


# Figure 2 — T* heatmap

def figure_tstar_heatmap(d):
    expA = d["exp_A_multimodel"]; expB = d["exp_B_p6_scaling"]
    p1   = expA.get("P1_per_model", {}); p6 = expB.get("P6_tests", {})
    MODELS  = ["Phi-3.5-mini (3.8B)", "Qwen2.5-3B (3.0B)", "Qwen2.5-1.5B (1.5B)"]
    LEVELS  = ["Level_1", "Level_3", "Level_5"]
    LLABELS = ["L1 (easy)", "L3 (med.)", "L5 (hard)"]
    mat = np.full((3, 3), np.nan)
    for j, lv in enumerate(LEVELS):
        mat[0,j] = float(p1.get("phi35",  {}).get(lv, np.nan) or np.nan)
        mat[1,j] = float(p1.get("qwen3b", {}).get(lv, np.nan) or np.nan)
        mat[2,j] = float(p6.get(lv, {}).get("T_stars", {}).get("qwen_1b5", np.nan) or np.nan)

    fig, ax = plt.subplots(figsize=(7, 3.8)); fig.patch.set_facecolor("white")
    cmap = plt.cm.YlOrRd.copy(); cmap.set_bad("whitesmoke")
    fin  = mat[~np.isnan(mat)]
    im   = ax.imshow(mat, cmap=cmap, aspect="auto",
                     vmin=fin.min() if len(fin) else 0,
                     vmax=fin.max() if len(fin) else 1)
    cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label(r"Optimal budget $T^*$ (tokens)", fontsize=10)
    ax.set_xticks(range(3)); ax.set_xticklabels(LLABELS, fontsize=10)
    ax.set_yticks(range(3)); ax.set_yticklabels(MODELS,  fontsize=10)
    ax.set_title("Fitted optimal budget T* across model-difficulty pairs\n(row = model,  column = difficulty level)",
                 fontsize=11, fontweight="bold", pad=10)
    mn = fin.mean() if len(fin) else 0
    for i in range(3):
        for j in range(3):
            v = mat[i,j]
            if not np.isnan(v):
                fg = "white" if v > mn else "black"
                ax.text(j, i, f"{int(v):,}", ha="center", va="center",
                        fontsize=11, fontweight="bold", color=fg)
            else:
                ax.text(j, i, "N/A", ha="center", va="center",
                        fontsize=9, color=C["grey"], style="italic")
    fig.tight_layout()
    out = os.path.join(FIGS, "figure_tstar_heatmap.png")
    fig.savefig(out, bbox_inches="tight", dpi=200); plt.close(fig)
    print(f"  saved: {out}")


# Figure 3 — Prospective g-model fit (Exp E)
def figure_prospective_fit(d):
    expE = d["exp_E_prospective"]
    T_train_max = 1200
    T_curve = np.linspace(100, 10000, 500)
    CASES = [
        ("phi35",  "Level_3", "Phi-3.5-mini  x  Difficulty L3 (medium)"),
        ("phi35",  "Level_5", "Phi-3.5-mini  x  Difficulty L5 (hard)"),
        ("qwen3b", "Level_1", "Qwen2.5-3B  x  Difficulty L1 (easy)"),
        ("qwen3b", "Level_3", "Qwen2.5-3B  x  Difficulty L3 (medium)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))
    fig.patch.set_facecolor("white")
    axes = axes.flatten()

    for ax, (mkey, lv, title) in zip(axes, CASES):
        mdata    = expE.get(mkey, {}).get(lv, {})
        measured = mdata.get("measured", {})
        fit_info = mdata.get("fit", {})

        T_obs = np.array(sorted(int(k) for k in measured), dtype=float)
        A_obs = np.array([measured[str(int(t))] for t in T_obs])

        ax.axvspan(T_train_max, T_curve[-1]*1.05,
                   alpha=0.10, color=C["lemon"], zorder=0, label="Extrapolation zone")
        ax.axvline(T_train_max, color=C["lemon"], lw=1.5, ls="--", alpha=0.8)

        mtr = T_obs <= T_train_max; mte = T_obs > T_train_max
        ax.scatter(T_obs[mtr], A_obs[mtr], color=C["blue"],  s=60, zorder=5,
                   label="Observed (train)", marker="o")
        if mte.any():
            ax.scatter(T_obs[mte], A_obs[mte], color=C["green"], s=60, zorder=5,
                       label="Observed (test)", marker="s")

        params = fit_info.get("params", [])
        if len(params) == 5:
            Delta, sigma_r2, k, scale, offset = params
            A_fit = g_scaled(T_curve, Delta, sigma_r2, k, scale, offset)
            ax.plot(T_curve, A_fit, color=C["red"], lw=2.2, zorder=4, label="g-model fit")
            T_star = fit_info.get("T_star")
            if T_star is not None and not np.isnan(float(T_star)):
                T_star = float(T_star)
                ax.axvline(T_star, color=C["red"], lw=1.4, ls=":", alpha=0.75)
                ylim = ax.get_ylim()
                yannot = ylim[1]*0.93 if ylim[1] > 0 else 95
                ax.text(min(T_star+300, 9000), yannot,
                        f"T*={T_star:.0f}", fontsize=8.5, color=C["red"])

        r2 = mdata.get("eval", {}).get("R2_test", np.nan)
        r2_str = f"R2_test = {r2:.3f}" if isinstance(r2, float) and not np.isnan(r2) else "R2_test = N/A"
        ax.text(0.97, 0.05, r2_str, ha="right", va="bottom", transform=ax.transAxes,
                fontsize=9, color=C["grey"],
                bbox=dict(fc="white", ec="#CCCCCC", alpha=0.85, boxstyle="round,pad=0.3"))

        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Thinking budget T (tokens)")
        ax.set_ylabel("Accuracy (%)")
        ax.set_xlim(80, T_curve[-1]*1.06); ax.set_ylim(0, 112)
        ax.legend(fontsize=8.5, loc="upper right")

    fig.text(0.5, 0.985,
             "Prospective g-model fit: trained on T <= 1,200, evaluated on T > 1,200",
             ha="center", va="top", fontsize=13, fontweight="bold")
    fig.text(0.5, 0.963,
             "Shaded = extrapolation window  |  circles = training budgets  |  squares = held-out test budgets",
             ha="center", va="top", fontsize=9.5, color=C["grey"])

    fig.tight_layout(rect=[0, 0, 1, 0.955])
    out = os.path.join(FIGS, "figure_prospective_fit_expE.png")
    fig.savefig(out, bbox_inches="tight", dpi=200); plt.close(fig)
    print(f"  saved: {out}")


if __name__ == "__main__":
    print("Image Generation ...\n")
    data = load()
    print("--- (1/3) Summary predictions P1-P6 ---")
    figure_summary_predictions(data)
    print("--- (2/3) T* heatmap ---")
    figure_tstar_heatmap(data)
    print("--- (3/3) Prospective g-model fit (Exp E) ---")
    figure_prospective_fit(data)
    print(f"\nAll figures saved to {os.path.abspath(FIGS)}")
