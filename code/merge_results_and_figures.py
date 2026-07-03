#!/usr/bin/env python3
"""
Reads all JSON files produced by parts 2–7
"""

import json, math, warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import scipy.stats as stats

OUT = Path("/kaggle/working/results_final")
OUT.mkdir(parents=True, exist_ok=True)

#  COLOUR PALETTE 

BLUE   = "#2166AC"
RED    = "#D6604D"
ORANGE = "#F4A582"
GREEN  = "#4DAC26"
PURPLE = "#762A83"
GRAY   = "#878787"
GOLD   = "#B35806"

LEVEL_COLS  = {"Level_1": BLUE,   "Level_3": ORANGE, "Level_5": RED}
LEVEL_LBLS  = {"Level_1": "L1 (easy)", "Level_3": "L3 (med)", "Level_5": "L5 (hard)"}
MODEL_COLS  = {
    "phi35":    GREEN,
    "qwen3b":   PURPLE,
    "gemma2":   GOLD,
    "qwen_1b5": "#1F78B4",
    "qwen_3b":  "#33A02C",
    "qwen_7b":  "#E31A1C",
}
MODEL_LBLS  = {
    "phi35":    "Phi-3.5-mini",
    "qwen3b":   "Qwen2.5-3B",
    "gemma2":   "Gemma-2-2B",
    "qwen_1b5": "Qwen1.5B",
    "qwen_3b":  "Qwen3B",
    "qwen_7b":  "Qwen7B",
}
PREC_COLS  = {"fp16": BLUE, "bf16": GREEN, "int8": ORANGE, "int4": RED}
PREC_LBLS  = {"fp16":"FP16","bf16":"BF16","int8":"INT8","int4":"INT4"}

plt.rcParams.update({
    "font.family":"serif","font.size":10,"axes.titlesize":11,
    "axes.labelsize":10,"legend.fontsize":8,"savefig.dpi":300,
    "axes.spines.top":False,"axes.spines.right":False,
    "axes.grid":True,"grid.alpha":0.22,
    "lines.linewidth":1.8,"lines.markersize":5,
})

def savefig(name, fig=None):
    fig = fig or plt.gcf()
    for ext in ("pdf","png"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  [fig] {name}")

def load_json(path):
    p = Path(path)
    if p.exists():
        with open(p) as f:
            return json.load(f)
    print(f"  [missing] {path}")
    return {}

def tex(v, fmt=".3f"):
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return "—"
    return format(v, fmt)

#  LOAD ALL RESULTS 

R2  = load_json("/kaggle/working/results_part2/part2_results.json")
R3  = load_json("/kaggle/working/results_part3/multimodel_results.json")
R4  = load_json("/kaggle/working/results_part4/p6_results.json")
R5  = load_json("/kaggle/working/results_part5/hurst_results.json")
R6  = load_json("/kaggle/working/results_part6/precision_results.json")
R7  = load_json("/kaggle/working/results_part7/prospective_results.json")

MERGED = {
    "part2_original": R2,
    "exp_A_multimodel": R3,
    "exp_B_p6_scaling": R4,
    "exp_C_hurst": R5,
    "exp_D_precision": R6,
    "exp_E_prospective": R7,
}
with open(OUT / "all_results_merged.json", "w") as f:
    json.dump(MERGED, f, indent=2, default=str)
print("[json] all_results_merged.json")

#  HELPER: extract T* from any result dict 

def get_tstar(fit_dict):
    if not fit_dict or "T_star" not in fit_dict:
        return float("nan")
    return float(fit_dict["T_star"])

def get_r2(fit_dict):
    if not fit_dict or "R2" not in fit_dict:
        return float("nan")
    return float(fit_dict["R2"])

#  FIGURE 1: GRAND SUMMARY  5-panel, one per prediction 

fig = plt.figure(figsize=(18, 10))
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)

#  Panel (a): P1  T* by difficulty, multi-model 
ax = fig.add_subplot(gs[0, 0])
levels  = ["Level_1","Level_3","Level_5"]
models_A = ["phi35","qwen3b","gemma2"]
x = np.arange(len(levels))
w = 0.28

for i, mk in enumerate(models_A):
    fits = R3.get("fits", {}).get(mk, {})
    vals = [get_tstar(fits.get(lv)) for lv in levels]
    valid_vals = [v for v in vals if not math.isnan(v)]
    if not valid_vals:
        continue
    bars = ax.bar(x + (i-1)*w, [v if not math.isnan(v) else 0 for v in vals],
                  w*0.9, color=MODEL_COLS[mk], alpha=0.82,
                  label=MODEL_LBLS[mk], zorder=3)
ax.set_xticks(x)
ax.set_xticklabels([LEVEL_LBLS[lv] for lv in levels], fontsize=9)
ax.set_ylabel("Optimal budget $T^*$ (tokens)")
ax.set_title("(a) P1: $T^*(d)$ increases with difficulty\n[Exp A — 3 models]")
ax.legend(fontsize=7, loc="upper left")

#  Panel (b): P3  T* by precision, Level_3 
ax = fig.add_subplot(gs[0, 1])
precs = ["fp16","bf16","int8","int4"]
p3_tests = R6.get("P3_tests", {})
ts_by_level = {}
for lv in levels:
    p3t = p3_tests.get(lv, {})
    ts_by_level[lv] = {p: p3t.get("T_stars",{}).get(p, float("nan")) for p in precs}

x2  = np.arange(len(precs))
w2  = 0.28
for i, lv in enumerate(levels):
    vals = [ts_by_level[lv].get(p, float("nan")) for p in precs]
    ax.bar(x2 + (i-1)*w2, [v if not math.isnan(v) else 0 for v in vals],
           w2*0.9, color=LEVEL_COLS[lv], alpha=0.82,
           label=LEVEL_LBLS[lv], zorder=3)
ax.set_xticks(x2)
ax.set_xticklabels([PREC_LBLS[p] for p in precs], fontsize=9)
ax.set_ylabel("$T^*$ (tokens)")
ax.set_title("(b) P3: $T^*$ decreases with quantisation\n[Exp D — FP16→BF16→INT8→INT4]")
ax.legend(fontsize=7)

#  Panel (c): P4 peak accuracy vs difficulty 
ax = fig.add_subplot(gs[0, 2])
for i, mk in enumerate(models_A):
    fits = R3.get("fits", {}).get(mk, {})
    peaks = [fits.get(lv, {}).get("peak_acc", float("nan")) for lv in levels]
    valid = [(j, p) for j, p in enumerate(peaks) if not math.isnan(p)]
    if not valid:
        continue
    jj, pp = zip(*valid)
    ax.plot([LEVEL_LBLS[levels[j]] for j in jj], list(pp),
            "o-", color=MODEL_COLS[mk], lw=1.8, ms=7, label=MODEL_LBLS[mk])
ax.set_ylabel("Peak accuracy at $T^*$ (%)")
ax.set_title("(c) P4: peak accuracy decreases with difficulty\n[Exp A — 3 models]")
ax.legend(fontsize=7)

#  Panel (d): P5  H estimates with CI 
ax = fig.add_subplot(gs[1, 0])
hurst_models = ["phi35","qwen3b"]
x3  = np.arange(len(levels))
w3  = 0.35
for i, mk in enumerate(hurst_models):
    Hs   = []
    errl = []
    errh = []
    for lv in levels:
        hr = R5.get(mk, {}).get(lv, {}).get("hurst", {})
        H  = hr.get("H", float("nan"))
        ci = hr.get("H_CI_boot", [float("nan"), float("nan")])
        Hs.append(H if not math.isnan(H) else 0)
        errl.append((H - ci[0]) if not math.isnan(H) and not math.isnan(ci[0]) else 0)
        errh.append((ci[1] - H) if not math.isnan(H) and not math.isnan(ci[1]) else 0)
    xs = x3 + (i - 0.5) * w3
    ax.bar(xs, Hs, w3*0.9, color=MODEL_COLS[mk], alpha=0.82,
           label=MODEL_LBLS[mk], zorder=3)
    ax.errorbar(xs, Hs, yerr=[errl, errh], fmt="none",
                ecolor="black", elinewidth=1.5, capsize=4, zorder=4)
ax.axhline(0.5, color="red", lw=1.5, ls="--", alpha=0.7, label="H=0.5 (theory)")
ax.axhspan(0.35, 0.65, alpha=0.06, color="red")
ax.set_xticks(x3)
ax.set_xticklabels([LEVEL_LBLS[lv] for lv in levels], fontsize=9)
ax.set_ylabel("Hurst exponent $H$")
ax.set_ylim(0, 1.0)
ax.set_title("(d) P5: $H\\approx 0.5$ (variance grows as $T^{2H}$)\n[Exp C — 2 models]")
ax.legend(fontsize=7)

#  Panel (e): P6  T* vs model size 
ax = fig.add_subplot(gs[1, 1])
qwen_keys  = ["qwen_1b5","qwen_3b","qwen_7b"]
qwen_params = [1500, 3000, 7000]
for lv in levels:
    ts_vals = []
    for mk in qwen_keys:
        t = get_tstar(R4.get("fits",{}).get(mk,{}).get(lv,{}))
        ts_vals.append(t)
    valid = [(np.log10(qwen_params[i]), ts_vals[i])
             for i in range(len(ts_vals)) if not math.isnan(ts_vals[i])]
    if not valid:
        continue
    xs, ys = zip(*valid)
    ax.plot(list(xs), list(ys), "o-", color=LEVEL_COLS[lv], lw=1.8, ms=7,
            label=LEVEL_LBLS[lv])
    if len(xs) >= 2:
        m, b_int, _, _, _ = stats.linregress(list(xs), list(ys))
        xr = np.array([min(xs), max(xs)])
        ax.plot(xr, m*xr+b_int, "--", color=LEVEL_COLS[lv], lw=1.0, alpha=0.5)
ax.set_xlabel(r"$\log_{10}(N)$ (params in M)")
ax.set_ylabel("$T^*$ (tokens)")
ax.set_title("(e) P6: $T^*$ decreases with model size\n"
             "[Exp B — Qwen2.5 family]")
ax.legend(fontsize=7)

#  Panel (f): Exp E  prospective R² summary 
ax = fig.add_subplot(gs[1, 2])
prosp_models = ["phi35","qwen3b"]
x4  = np.arange(len(levels))
w4  = 0.35
for i, mk in enumerate(prosp_models):
    r2_vals = [
        R7.get(mk,{}).get(lv,{}).get("eval",{}).get("R2_test", float("nan"))
        for lv in levels
    ]
    ax.bar(x4 + (i - 0.5)*w4,
           [v if not math.isnan(v) else 0 for v in r2_vals],
           w4*0.9, color=MODEL_COLS[mk], alpha=0.82,
           label=MODEL_LBLS[mk], zorder=3)
ax.axhline(0.8, color="red", lw=1.5, ls="--", alpha=0.7, label="$R^2=0.8$ threshold")
ax.set_xticks(x4)
ax.set_xticklabels([LEVEL_LBLS[lv] for lv in levels], fontsize=9)
ax.set_ylabel("$R^2$ on test budgets {2400,4800,9600}")
ax.set_title("(f) Prospective prediction quality\n"
             "[Exp E — fit on {150,300,600,1200}]")
ax.legend(fontsize=7)

fig.suptitle(
    "Summary of all falsifiable predictions (P1–P6) across experiments A–E\n"
    "Phi-3.5-mini, Qwen2.5 family, Gemma-2-2B  |  2×T4, 8-bit quantisation",
    fontsize=12, y=1.01,
)
savefig("figure_summary_predictions", fig)

#  FIGURE 2: T* HEATMAP 

# Collect T* for all (model, level) pairs from Exp A
all_model_keys = ["phi35","qwen3b","gemma2"]
tstar_matrix   = np.full((len(all_model_keys), len(levels)), float("nan"))
for ri, mk in enumerate(all_model_keys):
    for ci_ax, lv in enumerate(levels):
        t = get_tstar(R3.get("fits",{}).get(mk,{}).get(lv,{}))
        tstar_matrix[ri, ci_ax] = t

# Fill any missing from part2 (Phi-only)
if math.isnan(tstar_matrix[0, 0]):
    for ci_ax, lv in enumerate(levels):
        t = get_tstar(R2.get("overthinking_fits",{}).get(lv,{}))
        if not math.isnan(t):
            tstar_matrix[0, ci_ax] = t

if not np.all(np.isnan(tstar_matrix)):
    vmin = np.nanmin(tstar_matrix)
    vmax = np.nanmax(tstar_matrix)
    fig, ax = plt.subplots(figsize=(7, 4))
    im  = ax.imshow(tstar_matrix, aspect="auto", cmap="YlOrRd",
                    vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(levels)))
    ax.set_xticklabels([LEVEL_LBLS[lv] for lv in levels])
    ax.set_yticks(range(len(all_model_keys)))
    ax.set_yticklabels([MODEL_LBLS[mk] for mk in all_model_keys])
    for ri in range(len(all_model_keys)):
        for ci_ax in range(len(levels)):
            v = tstar_matrix[ri, ci_ax]
            if not math.isnan(v):
                ax.text(ci_ax, ri, f"{v:.0f}", ha="center", va="center",
                        fontsize=11, fontweight="bold",
                        color="white" if v > (vmin + (vmax-vmin)*0.6) else "black")
    plt.colorbar(im, ax=ax, label="$T^*$ (tokens)")
    ax.set_title("T* heatmap — model × difficulty level\n"
                 "P1 (horizontal increase) and P4 confirmed if row-wise decrease absent")
    plt.tight_layout()
    savefig("figure_tstar_heatmap", fig)

#  LaTeX: MAIN PREDICTIONS SUMMARY TABLE 

def make_predictions_table():
    lines = [
        "% AUTO-GENERATED by part8_merge_figures.py",
        r"\begin{table}[t]",
        r"\centering\small",
        r"\caption{Summary of all falsifiable predictions tested. "
        r"\checkmark = supported; $\sim$ = mixed; \texttimes = not supported. "
        r"Each row corresponds to one proposition from the theory.}",
        r"\label{tab:predictions-summary}",
        r"\begin{tabular}{clp{5.5cm}l}",
        r"\toprule",
        r"Prop. & Test & Evidence & Result \\",
        r"\midrule",
    ]

    # P1
    p1_results = []
    for mk in ["phi35","qwen3b","gemma2"]:
        p1 = R3.get("P1_per_model",{}).get(mk,{})
        if p1:
            p1_results.append(p1.get("monotonic", False))
    p1_p2 = R2.get("P1_test",{})
    if p1_p2:
        p1_results.append(p1_p2.get("monotonic", False))
    p1_ok = sum(p1_results) / max(len(p1_results), 1)
    p1_sym = r"\checkmark" if p1_ok >= 0.67 else (r"$\sim$" if p1_ok >= 0.33 else r"\texttimes")
    lines.append(
        rf"P1 & $T^*(d)$ monotone-increasing & "
        rf"Exp A: {sum(p1_results)}/{len(p1_results)} models supported & {p1_sym} \\"
    )

    # P3
    p3_ok_count = 0; p3_total = 0
    for lv in ["Level_1","Level_3","Level_5"]:
        p3t = R6.get("P3_tests",{}).get(lv,{})
        if p3t:
            p3_total += 1
            if p3t.get("monotone_decreasing"):
                p3_ok_count += 1
    p3_sym = r"\checkmark" if p3_ok_count == p3_total and p3_total > 0 \
             else (r"$\sim$" if p3_ok_count > 0 else r"\texttimes")
    lines.append(
        rf"P3 & $T^*$ decreases with quantisation & "
        rf"Exp D: {p3_ok_count}/{p3_total} levels supported (FP16>BF16>INT8>INT4) & {p3_sym} \\"
    )

    # P4
    p4_results = []
    for mk in ["phi35","qwen3b","gemma2"]:
        fits = R3.get("fits",{}).get(mk,{})
        peaks = {lv: fits.get(lv,{}).get("peak_acc", float("nan"))
                 for lv in ["Level_1","Level_3","Level_5"]}
        if all(not math.isnan(v) for v in peaks.values()):
            ok = peaks["Level_1"] > peaks["Level_3"] > peaks["Level_5"]
            p4_results.append(ok)
    p4_sym = r"\checkmark" if sum(p4_results) == len(p4_results) and p4_results \
             else (r"$\sim$" if sum(p4_results) > 0 else r"\texttimes")
    lines.append(
        rf"P4 & Peak accuracy $\downarrow$ with difficulty & "
        rf"Exp A: {sum(p4_results)}/{len(p4_results)} models supported & {p4_sym} \\"
    )

    # P5
    h_ok = 0; h_total = 0
    for mk in ["phi35","qwen3b"]:
        for lv in ["Level_1","Level_3","Level_5"]:
            hr = R5.get(mk,{}).get(lv,{}).get("hurst",{})
            if "H" in hr:
                h_total += 1
                if hr.get("supports_P5"):
                    h_ok += 1
    p5_sym = r"\checkmark" if h_ok >= h_total*0.67 and h_total > 0 \
             else (r"$\sim$" if h_ok > 0 else r"\texttimes")
    lines.append(
        rf"P5 & $V(T)\propto T^{{2H}}$, $H\approx 0.5$ & "
        rf"Exp C: $H\approx 0.5$ in {h_ok}/{h_total} (model,level) pairs & {p5_sym} \\"
    )

    # P6
    p6_ok = 0; p6_total = 0
    for lv in ["Level_1","Level_3","Level_5"]:
        p6t = R4.get("P6_tests",{}).get(lv,{})
        if p6t:
            p6_total += 1
            if p6t.get("monotone_decreasing_with_N"):
                p6_ok += 1
    p6_sym = r"\checkmark" if p6_ok == p6_total and p6_total > 0 \
             else (r"$\sim$" if p6_ok > 0 else r"\texttimes")
    lines.append(
        rf"P6 & $\partial T^*/\partial N < 0$ (larger model $\to$ lower $T^*$) & "
        rf"Exp B: {p6_ok}/{p6_total} levels supported (Qwen 1.5B/3B/7B) & {p6_sym} \\"
    )

    # Prospective
    prosp_strong = 0; prosp_total = 0
    for mk in ["phi35","qwen3b"]:
        for lv in ["Level_1","Level_3","Level_5"]:
            r2t = R7.get(mk,{}).get(lv,{}).get("eval",{}).get("R2_test", float("nan"))
            if not math.isnan(r2t):
                prosp_total += 1
                if r2t > 0.8:
                    prosp_strong += 1
    prosp_sym = r"\checkmark" if prosp_strong >= prosp_total*0.67 and prosp_total > 0 \
               else (r"$\sim$" if prosp_strong > 0 else r"\texttimes")
    lines.append(
        rf"— & Prospective prediction ($R^2>0.8$) & "
        rf"Exp E: {prosp_strong}/{prosp_total} (model,level) pairs $R^2>0.8$ & {prosp_sym} \\"
    )

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)

tex_pred = make_predictions_table()
(OUT / "table_predictions_summary.tex").write_text(tex_pred, encoding="utf-8")
print("  [tex] table_predictions_summary.tex")

#  LaTeX: PROSPECTIVE DETAIL TABLE 

lines2 = [
    "% AUTO-GENERATED by part8_merge_figures.py",
    r"\begin{table}[t]",
    r"\centering\small",
    r"\caption{Experiment E prospective prediction results. "
    r"g-model fitted on $T\in\{150,300,600,1200\}$, "
    r"evaluated at $T\in\{2400,4800,9600\}$ without refitting. "
    r"$R^2_{\mathrm{test}}>0.8$ indicates the model captures a "
    r"real phenomenon rather than overfitting.}",
    r"\label{tab:prospective-detail}",
    r"\begin{tabular}{llccc}",
    r"\toprule",
    r"Model & Level & $R^2_{\mathrm{train}}$ & $R^2_{\mathrm{test}}$ & MAE (pp) \\",
    r"\midrule",
]
for mk in ["phi35","qwen3b"]:
    for li, lv in enumerate(["Level_1","Level_3","Level_5"]):
        res    = R7.get(mk,{}).get(lv,{})
        fit_r  = res.get("fit",{}) or {}
        eval_r = res.get("eval",{}) or {}
        mlbl   = MODEL_LBLS[mk] if li == 0 else ""
        lines2.append(
            rf"{mlbl} & {lv.replace('_',' ')} & "
            rf"${tex(fit_r.get('R2_train'))}$ & "
            rf"${tex(eval_r.get('R2_test'))}$ & "
            rf"${tex(eval_r.get('MAE'),'.1f')}$ \\"
        )
    lines2.append(r"\midrule")
lines2[-1] = r"\bottomrule"
lines2 += [r"\end{tabular}", r"\end{table}"]
(OUT / "table_prospective_summary.tex").write_text("\n".join(lines2), encoding="utf-8")
print("  [tex] table_prospective_summary.tex")

#  FINAL PRINT SUMMARY 

print("\n" + "="*68)
print("  FULL EMPIRICAL PACKAGE SUMMARY")
print("="*68)

print("\n  P1 (T* monotone with difficulty):")
for mk in ["phi35","qwen3b","gemma2"]:
    p1 = R3.get("P1_per_model",{}).get(mk,{})
    if p1:
        t1 = p1.get("Level_1",0); t3=p1.get("Level_3",0); t5=p1.get("Level_5",0)
        ok = p1.get("monotonic","?")
        print(f"    {MODEL_LBLS[mk]:22s}  L1={t1:.0f}  L3={t3:.0f}  L5={t5:.0f}  "
              f"{'SUPPORTED' if ok else 'NOT SUPPORTED'}")

print("\n  P3 (T* decreases with quantisation):")
for lv in ["Level_1","Level_3","Level_5"]:
    p3t = R6.get("P3_tests",{}).get(lv,{})
    if p3t:
        ts = p3t.get("T_stars",{})
        row = "  ".join(f"{p}={ts.get(p,float('nan')):.0f}" for p in ["fp16","bf16","int8","int4"] if p in ts)
        print(f"    {lv}: {row}  "
              f"{'SUPPORTED' if p3t.get('monotone_decreasing') else 'NOT SUPPORTED'}")

print("\n  P5 (Hurst H≈0.5):")
for mk in ["phi35","qwen3b"]:
    for lv in ["Level_1","Level_3","Level_5"]:
        hr = R5.get(mk,{}).get(lv,{}).get("hurst",{})
        if "H" in hr:
            ci = hr.get("H_CI_boot",["?","?"])
            print(f"    {MODEL_LBLS[mk]:20s} {lv}: H={hr['H']:.3f} "
                  f"[{ci[0]:.3f},{ci[1]:.3f}]  R²={hr.get('R2',0):.3f}  "
                  f"{'✓' if hr.get('supports_P5') else '✗'}")

print("\n  P6 (T* decreases with N):")
for lv in ["Level_1","Level_3","Level_5"]:
    p6t = R4.get("P6_tests",{}).get(lv,{})
    if p6t:
        ts = p6t.get("T_stars",{})
        row = "  ".join(f"{MODEL_LBLS.get(mk,mk)}={ts[mk]:.0f}" for mk in ts)
        print(f"    {lv}: {row}  "
              f"{'SUPPORTED' if p6t.get('monotone_decreasing_with_N') else 'NOT SUPPORTED'}")

print("\n  Prospective prediction (R²_test):")
for mk in ["phi35","qwen3b"]:
    for lv in ["Level_1","Level_3","Level_5"]:
        r2t = R7.get(mk,{}).get(lv,{}).get("eval",{}).get("R2_test", float("nan"))
        if not math.isnan(r2t):
            mae = R7.get(mk,{}).get(lv,{}).get("eval",{}).get("MAE", float("nan"))
            strength = "STRONG" if r2t > 0.8 else ("MODERATE" if r2t > 0.5 else "WEAK")
            print(f"    {MODEL_LBLS[mk]:20s} {lv}: R²={r2t:.3f}  MAE={mae:.1f}pp  → {strength}")

print(f"\n  Outputs: {OUT}")
for fp in sorted(OUT.iterdir()):
    print(f"    {fp.name:50s}  {fp.stat().st_size/1024:7.1f} KB")
print("\n  DONE — all paper-ready outputs generated.")
