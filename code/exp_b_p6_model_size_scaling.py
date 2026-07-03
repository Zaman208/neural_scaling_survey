#!/usr/bin/env python3
"""
EXPERIMENT B: P6 Model-Size Scaling
=====================================
Tests Proposition P6: dT*/dN < 0 — larger models reach their overthinking
threshold at a lower token budget than smaller models.

Models
  Qwen2.5-1.5B-Instruct   1.5 B
  Qwen2.5-3B-Instruct     3.0 B
  Qwen2.5-7B-Instruct     7.0 B

Protocol
  • 3 difficulty levels  : Level_1, Level_3, Level_5
  • 10 problems / level
  • Budgets              : [200, 500, 1000, 2000, 4000, 8000]
  • 3 independent samples per (problem, budget)
  • 8-bit quantisation (7B ~7 GB, fits on one T4 15GB)

"""

import os, sys, json, time, math, warnings, gc, re
from pathlib import Path

warnings.filterwarnings("ignore")
START_TIME = time.time()

import subprocess
for pkg in ["torch", "transformers", "accelerate", "bitsandbytes",
            "scipy", "scikit-learn", "sympy"]:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg],
                   capture_output=True)

import numpy as np
import scipy.stats as stats
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import torch
import sympy as sp
from sympy.parsing.sympy_parser import (
    parse_expr, standard_transformations, implicit_multiplication_application,
)
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

torch.manual_seed(42); np.random.seed(42)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[Setup] device={DEVICE}  GPUs={torch.cuda.device_count()}")
if DEVICE == "cuda":
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {p.name}  {p.total_memory/1e9:.1f} GB")

OUT = Path("/kaggle/working/results_part4")
OUT.mkdir(parents=True, exist_ok=True)

#  1. CONSTANTS 

BUDGETS   = [200, 500, 1000, 2000, 4000, 8000]
N_SAMPLES = 3
N_PROBS   = 10
SESSION_LIMIT_MIN = 510

# Model N values in millions (for plotting)
MODELS = {
    "qwen_1b5": ("Qwen/Qwen2.5-1.5B-Instruct", 1500),
    "qwen_3b":  ("Qwen/Qwen2.5-3B-Instruct",   3000),
    "qwen_7b":  ("Qwen/Qwen2.5-7B-Instruct",   7000),
}
MODEL_LABELS = {
    "qwen_1b5": "Qwen2.5-1.5B",
    "qwen_3b":  "Qwen2.5-3B",
    "qwen_7b":  "Qwen2.5-7B",
}
MODEL_COLS   = {
    "qwen_1b5": "#1F78B4",
    "qwen_3b":  "#33A02C",
    "qwen_7b":  "#E31A1C",
}
LEVEL_COLS   = {"Level_1":"#2166AC","Level_3":"#F4A582","Level_5":"#D6604D"}

SYS_PROMPT = (
    "You are a precise math assistant. "
    "Think step by step. "
    "Write your final numerical answer after the word ANSWER: on its own line."
)

MATH_PROBLEMS = {
    "Level_1": [
        ("Compute $2^3 + 3^2$.", "17"),
        ("What is $15\\%$ of $200$?", "30"),
        ("If $x + 5 = 12$, find $x$.", "7"),
        ("What is $\\sqrt{49}$?", "7"),
        ("Calculate $3!$.", "6"),
        ("What is $4^2 - 3^2$?", "7"),
        ("Find $x$: $2x = 18$.", "9"),
        ("Compute $7 \\times 8$.", "56"),
        ("What is $2^5$?", "32"),
        ("Find the mean of $4, 6, 8, 10$.", "7"),
    ],
    "Level_3": [
        ("Solve $x^2 - 5x + 6 = 0$.", "2 or 3"),
        ("A circle has radius 7. Area in terms of pi.", "49*pi"),
        ("If $f(x)=2x^2-3x+1$, find $f(3)$.", "10"),
        ("Solve: $\\log_2 8 = x$.", "3"),
        ("Distance between $(0,0)$ and $(3,4)$.", "5"),
        ("What is $C(6,2)$?", "15"),
        ("Find the 10th term of $3,7,11,\\dots$", "39"),
        ("Solve $|x-3|=5$.", "8 or -2"),
        ("Solve $2^x=32$.", "5"),
        ("Sum of first 10 natural numbers.", "55"),
    ],
    "Level_5": [
        ("Compute sum_{k=1}^{inf} 1/(k(k+1)).", "1"),
        ("Evaluate integral_0^pi sin^2(x) dx.", "pi/2"),
        ("Limit as x->0 of sin(3x)/x.", "3"),
        ("Eigenvalues of [[2,1],[1,2]].", "1 and 3"),
        ("Sum_{n=0}^{inf} 1/2^n.", "2"),
        ("Number of divisors of 360.", "24"),
        ("Solve e^{2x} - 3*e^x + 2 = 0.", "0 or log(2)"),
        ("d/dx of ln(x^2+1).", "2*x/(x**2+1)"),
        ("Sum 1^2 + 2^2 + ... + 10^2.", "385"),
        ("Solve x^4 - 5*x^2 + 4 = 0.", "1 or 2"),
    ],
}

#  2. CHECKPOINT I/O 

CKPT_PATH = OUT / "checkpoint.json"

def load_checkpoint():
    if CKPT_PATH.exists():
        with open(CKPT_PATH) as f:
            return json.load(f)
    return {}

def save_checkpoint(ckpt):
    tmp = CKPT_PATH.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(ckpt, f, indent=2, default=str)
    tmp.replace(CKPT_PATH)

def elapsed_min():
    return (time.time() - START_TIME) / 60

def time_ok():
    return elapsed_min() < SESSION_LIMIT_MIN

#  3. ANSWER CHECKING 

_SP_TRANSFORMS = standard_transformations + (implicit_multiplication_application,)

def _clean_latex(s):
    s = s.lower()
    s = re.sub(r"\\frac\s*{([^}]+)}\s*{([^}]+)}", r"(\1)/(\2)", s)
    s = re.sub(r"\\sqrt\s*{([^}]+)}", r"sqrt(\1)", s)
    for old, new in [
        (r"\\pi","pi"),(r"\\ln","log"),(r"\\log","log"),
        (r"\\sin","sin"),(r"\\cos","cos"),(r"\\cdot","*"),
        (r"\\times","*"),(r"\\infty","oo"),
    ]:
        s = re.sub(old, new, s)
    s = re.sub(r"[{}$\\]", " ", s)
    return s.strip()

def _sympy_eq(a, b):
    try:
        ea = parse_expr(a, transformations=_SP_TRANSFORMS)
        eb = parse_expr(b, transformations=_SP_TRANSFORMS)
        d  = sp.simplify(ea - eb)
        return d == 0 or abs(float(d.evalf())) < 1e-6
    except Exception:
        return False

def _extract_answer(text):
    for marker in ["ANSWER:", "Answer:", "= ", "equals "]:
        idx = text.rfind(marker)
        if idx >= 0:
            return text[idx + len(marker):].strip().split("\n")[0]
    return text.split(".")[-1].strip()

def answer_is_correct(generated, expected):
    gen_region = _extract_answer(generated)
    gen_clean  = _clean_latex(gen_region).replace(" ", "")
    parts = [p.strip() for p in expected.split("or") if p.strip()]
    for part in parts:
        part_clean = _clean_latex(part).replace(" ", "")
        if _sympy_eq(gen_clean, part_clean):
            return True
        if part_clean and part_clean in gen_clean:
            return True
    try:
        exp_num = float(sp.sympify(expected).evalf())
        nums = re.findall(r"-?\d+\.?\d*", gen_region)
        for n in nums:
            if abs(float(n) - exp_num) < 1e-4 * (abs(exp_num) + 1):
                return True
    except Exception:
        pass
    return False

# ── 4. g-MODEL FIT ────────────────────────────────────────────────────────────

def g_raw(T, Delta, sigma_r2, k):
    V = np.maximum(k * np.asarray(T, dtype=float), 0.0)
    s = np.maximum(sigma_r2 + V, 1e-12)
    return (1.0 / np.sqrt(2.0 * np.pi * s)) * np.exp(-Delta**2 / (2.0 * s))

def g_scaled(T, Delta, sigma_r2, k, scale, offset):
    return scale * g_raw(T, Delta, sigma_r2, k) + offset

def fit_g(T_arr, A_arr, n_boot=300):
    T = np.asarray(T_arr, dtype=float)
    A = np.asarray(A_arr, dtype=float)
    best_r2, best_p = -np.inf, None
    for Di in [0.3, 0.5, 0.8, 1.2, 2.0]:
        for ki in [1e-5, 5e-5, 1e-4, 5e-4]:
            try:
                T_pk = T[np.argmax(A)]
                gp   = g_raw(np.array([T_pk]), Di, 0.05, ki)[0]
                sc0  = max(A) / max(gp, 1e-10)
                popt, _ = curve_fit(
                    g_scaled, T, A,
                    p0=[Di, 0.05, ki, sc0, 0.0],
                    bounds=([0.01,1e-4,1e-9, 1.0,-50.0],
                            [5.00,5.00,1.00,1e6, 50.0]),
                    maxfev=20000, method="trf",
                )
                ss_r = np.sum((A - g_scaled(T, *popt))**2)
                ss_t = np.sum((A - A.mean())**2)
                r2   = 1.0 - ss_r / ss_t if ss_t > 0 else -np.inf
                if r2 > best_r2:
                    best_r2, best_p = r2, popt
            except Exception:
                continue
    if best_p is None:
        return {"error": "fit_failed"}
    Delta, sr2, k, sc, off = best_p
    T_fine = np.linspace(T.min(), T.max() * 2.5, 8000)
    A_fine = g_scaled(T_fine, *best_p)
    T_star = float(T_fine[np.argmax(A_fine)])
    peak   = float(np.max(A_fine))
    n  = len(T)
    rss= float(np.sum((A - g_scaled(T, *best_p))**2))
    kp = len(best_p)
    # bootstrap CI on T*
    ci  = {}
    bts = []
    rng = np.random.default_rng(42)
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        Tb, Ab = T[idx], A[idx]
        if len(np.unique(Tb)) < 3:
            continue
        try:
            pb, _ = curve_fit(
                g_scaled, Tb, Ab, p0=best_p,
                bounds=([0.01,1e-4,1e-9, 1.0,-50.0],
                        [5.00,5.00,1.00,1e6, 50.0]),
                maxfev=5000, method="trf",
            )
            bts.append(pb)
        except Exception:
            continue
    if len(bts) >= 40:
        ba = np.array(bts)
        bt_s = []
        for pb in ba:
            Tf2 = np.linspace(T.min(), T.max() * 2.5, 2000)
            bt_s.append(float(Tf2[np.argmax(g_scaled(Tf2, *pb))]))
        ci["T_star"] = [float(np.percentile(bt_s, 2.5)),
                        float(np.percentile(bt_s, 97.5))]
    return {
        "Delta":float(Delta),"sigma_r2":float(sr2),"k":float(k),
        "scale":float(sc),"offset":float(off),
        "R2":float(best_r2),"T_star":T_star,"peak_acc":peak,
        "CI":ci,"n_data":int(n),
    }

# ── 5. MODEL / SWEEP HELPERS ──────────────────────────────────────────────────

def load_model_8bit(model_id):
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    bnb = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=bnb,
        device_map="auto", trust_remote_code=False,
    )
    model.eval()
    return model, tok

def release(model):
    del model; gc.collect(); torch.cuda.empty_cache()

def build_prompt(tok, problem, sys_prompt):
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user",   "content": problem},
    ]
    try:
        text = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        text = f"Instruct: {sys_prompt}\n{problem}\nOutput:"
    return tok(text, return_tensors="pt").to(DEVICE)

def run_sweep(model, tok, problems, budgets, n_samples=N_SAMPLES):
    results = []
    for budget in budgets:
        all_correct = []
        for problem, expected in problems:
            inputs = build_prompt(tok, problem, SYS_PROMPT)
            in_len = inputs["input_ids"].shape[1]
            run_correct = []
            for _ in range(n_samples):
                with torch.no_grad():
                    out = model.generate(
                        **inputs,
                        max_new_tokens=budget,
                        do_sample=True, temperature=0.7, top_p=0.9,
                        pad_token_id=tok.eos_token_id,
                    )
                sequences = out.sequences if hasattr(out, "sequences") else out
                gen_ids  = sequences[0][in_len:]
                gen_text = tok.decode(gen_ids, skip_special_tokens=True)
                run_correct.append(int(answer_is_correct(gen_text, expected)))
            all_correct.append(float(np.mean(run_correct)))
        acc = float(np.mean(all_correct))
        results.append({"budget": budget, "accuracy": acc})
        print(f"      T={budget:5d}  acc={acc:.3f}")
    return results

# ── 6. MAIN EXPERIMENT LOOP ───────────────────────────────────────────────────

ckpt = load_checkpoint()
print(f"\n[Checkpoint] {len(ckpt)} (model,level) pairs already done.")

for model_key, (model_id, param_M) in MODELS.items():
    if not time_ok():
        print(f"\n[TIME] session limit approaching — stopping.")
        break
    levels_needed = [lv for lv in MATH_PROBLEMS
                     if f"{model_key}_{lv}" not in ckpt]
    if not levels_needed:
        print(f"\n[SKIP] {model_key} — all levels done.")
        continue
    print(f"\n{'='*66}")
    print(f"  Loading {MODEL_LABELS[model_key]}  ({param_M}M params)  "
          f"[{elapsed_min():.1f} min]")
    print(f"  Levels remaining: {levels_needed}")
    print(f"{'='*66}")
    try:
        model, tok = load_model_8bit(model_id)
        print(f"  Model loaded  [{elapsed_min():.1f} min]")
    except Exception as e:
        print(f"  ERROR loading {model_key}: {e}")
        ckpt[f"{model_key}_load_error"] = str(e)
        save_checkpoint(ckpt)
        continue
    for level in levels_needed:
        ck_key = f"{model_key}_{level}"
        if not time_ok():
            print(f"  [TIME] stopping before {ck_key}")
            break
        problems = MATH_PROBLEMS[level][:N_PROBS]
        print(f"\n  [{model_key}][{level}]  "
              f"{len(problems)} probs × {len(BUDGETS)} budgets × {N_SAMPLES} runs")
        sweep = run_sweep(model, tok, problems, BUDGETS)
        T_arr = np.array([r["budget"]       for r in sweep])
        A_arr = np.array([r["accuracy"]*100 for r in sweep])
        fit   = fit_g(T_arr, A_arr)
        print(f"  -> R²={fit.get('R2',0):.4f}  T*={fit.get('T_star',0):.0f}")
        ckpt[ck_key] = {"sweep": sweep, "fit": fit, "param_M": param_M,
                        "model_id": model_id, "elapsed_min": elapsed_min()}
        save_checkpoint(ckpt)
        print(f"  [ckpt] {ck_key} saved  [{elapsed_min():.1f} min]")
    release(model)
    print(f"  [released] {model_key}  [{elapsed_min():.1f} min]")

# ── 7. ASSEMBLE RESULTS ───────────────────────────────────────────────────────

RESULTS = {"curves": {}, "fits": {}, "P6_tests": {}}

for model_key, (_, param_M) in MODELS.items():
    RESULTS["curves"][model_key] = {}
    RESULTS["fits"][model_key]   = {}
    for level in MATH_PROBLEMS:
        ck_key = f"{model_key}_{level}"
        if ck_key in ckpt and "sweep" in ckpt[ck_key]:
            RESULTS["curves"][model_key][level] = ckpt[ck_key]["sweep"]
            RESULTS["fits"][model_key][level]   = ckpt[ck_key]["fit"]

# P6 test per level: T* decreases with N?
model_keys = list(MODELS.keys())
param_Ms   = [MODELS[mk][1] for mk in model_keys]

for level in MATH_PROBLEMS:
    t_stars = {}
    for mk in model_keys:
        fit = RESULTS["fits"].get(mk, {}).get(level, {})
        if "T_star" in fit:
            t_stars[mk] = fit["T_star"]
    if len(t_stars) < 2:
        continue
    # Check monotone decrease with N (increasing param_M)
    ordered_t = [t_stars.get(mk, float("nan")) for mk in model_keys]
    valid = [v for v in ordered_t if not math.isnan(v)]
    monotone_dec = all(valid[i] >= valid[i+1] for i in range(len(valid)-1))
    RESULTS["P6_tests"][level] = {
        "T_stars": t_stars,
        "param_Ms": {mk: MODELS[mk][1] for mk in t_stars},
        "monotone_decreasing_with_N": monotone_dec,
    }

out_json = OUT / "p6_results.json"
with open(out_json, "w") as f:
    json.dump(RESULTS, f, indent=2, default=str)
print(f"\n[json] p6_results.json  [{elapsed_min():.1f} min]")

# ── 8. FIGURES ────────────────────────────────────────────────────────────────

plt.rcParams.update({
    "font.family":"serif","font.size":11,"axes.titlesize":12,
    "axes.labelsize":11,"legend.fontsize":9,"savefig.dpi":300,
    "axes.spines.top":False,"axes.spines.right":False,
    "axes.grid":True,"grid.alpha":0.25,
})

def savefig(name, fig=None):
    fig = fig or plt.gcf()
    for ext in ("pdf","png"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  [fig] {name}")

# ── Figure B1: Overthinking curves for Level_3 (all 3 models) ────────────────
level_plot = "Level_3"
has_data = any(
    level_plot in RESULTS["curves"].get(mk, {}) for mk in model_keys
)
if has_data:
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for mk in model_keys:
        data = RESULTS["curves"].get(mk, {}).get(level_plot)
        if not data:
            continue
        col = MODEL_COLS[mk]
        T_d = np.array([r["budget"]       for r in data])
        A_d = np.array([r["accuracy"]*100 for r in data])
        ax.scatter(T_d/1000, A_d, color=col, s=50, zorder=6)
        ax.plot(T_d/1000, A_d, "o-", color=col, lw=1.3, alpha=0.4)
        fit = RESULTS["fits"].get(mk, {}).get(level_plot, {})
        if "Delta" in fit:
            popt = [fit["Delta"],fit["sigma_r2"],fit["k"],fit["scale"],fit["offset"]]
            Tf   = np.linspace(T_d.min(), T_d.max()*1.4, 2000)
            ax.plot(Tf/1000, g_scaled(Tf, *popt), color=col, lw=2.2,
                    label=f"{MODEL_LABELS[mk]}  T*={fit['T_star']:.0f}")
            ax.axvline(fit["T_star"]/1000, color=col, lw=0.9, ls=":", alpha=0.7)
            ci = fit.get("CI", {})
            if "T_star" in ci:
                ax.axvspan(ci["T_star"][0]/1000, ci["T_star"][1]/1000,
                           alpha=0.07, color=col)
    ax.set_xlabel("Budget $T$ (thousands of tokens)")
    ax.set_ylabel("Accuracy (%) — Level 3 (medium)")
    ax.set_title("Experiment B: P6 model-size scaling\n"
                 r"Prediction: $T^*$ decreases as $N$ increases")
    ax.legend(fontsize=9)
    savefig("figure_p6_scaling", fig)

# ── Figure B2: T* vs log(N) scatter for all levels ───────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=False)
for ax_i, level in enumerate(["Level_1","Level_3","Level_5"]):
    ax = axes[ax_i]
    p6t = RESULTS["P6_tests"].get(level, {})
    t_stars = p6t.get("T_stars", {})
    xs, ys, cs = [], [], []
    for mk in model_keys:
        if mk in t_stars:
            xs.append(np.log10(MODELS[mk][1]))
            ys.append(t_stars[mk])
            cs.append(MODEL_COLS[mk])
    if xs:
        ax.scatter(xs, ys, c=cs, s=120, zorder=6)
        for mk, x, y in zip([m for m in model_keys if m in t_stars], xs, ys):
            ax.annotate(MODEL_LABELS[mk], (x, y),
                        textcoords="offset points", xytext=(5,5), fontsize=8)
        if len(xs) >= 2:
            m, b, r, p, _ = stats.linregress(xs, ys)
            xr = np.array([min(xs)-0.1, max(xs)+0.1])
            ax.plot(xr, m*xr+b, "k--", lw=1.5, alpha=0.5,
                    label=f"slope={m:.0f}, R²={r**2:.2f}")
            ax.legend(fontsize=8)
    ax.set_xlabel(r"$\log_{10}(N)$ (model params in M)")
    ax.set_ylabel("$T^*$ (tokens)")
    ax.set_title(f"{level.replace('_',' ')} — "
                 f"P6 {'✓' if p6t.get('monotone_decreasing_with_N') else '✗'}")
plt.suptitle("P6 Test: T* vs model size (Qwen2.5 family)\n"
             "P6 predicts: larger N → smaller T*", y=1.02, fontsize=11)
plt.tight_layout()
savefig("figure_p6_tstar", fig)

# ── 9. LaTeX TABLE ────────────────────────────────────────────────────────────

def tex(v, fmt=".3f"):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    return format(v, fmt)

lines = [
    "% AUTO-GENERATED by part4_p6_scaling.py",
    r"\begin{table}[t]",
    r"\centering\small",
    r"\caption{Experiment B (P6): g-model fits for Qwen2.5 family. "
    r"P6 predicts $\partial T^*/\partial N < 0$: larger models reach the "
    r"overthinking threshold sooner. "
    r"{\checkmark} = monotone-decreasing T* with N.}",
    r"\label{tab:p6-scaling}",
    r"\begin{tabular}{llccc}",
    r"\toprule",
    r"Model ($N$) & Level & $T^*$ (tokens) & 95\% CI & P6 \\",
    r"\midrule",
]
for mk, (model_id, param_M) in MODELS.items():
    for li, level in enumerate(["Level_1","Level_3","Level_5"]):
        fit  = RESULTS["fits"].get(mk, {}).get(level, {})
        ci   = fit.get("CI", {})
        t_ci = ci.get("T_star", [float("nan"), float("nan")])
        p6t  = RESULTS["P6_tests"].get(level, {})
        p6c  = r"\checkmark" if p6t.get("monotone_decreasing_with_N") else r"\texttimes"
        mlbl = f"{MODEL_LABELS[mk]} ({param_M}M)" if li == 0 else ""
        p6col= p6c if li == 2 else ""
        if "T_star" not in fit:
            lines.append(rf"{mlbl} & {level.replace('_',' ')} & — & — & {p6col} \\")
        else:
            lines.append(
                rf"{mlbl} & {level.replace('_',' ')} & "
                rf"${tex(fit['T_star'],'.0f')}$ & "
                rf"$[{tex(t_ci[0],'.0f')},{tex(t_ci[1],'.0f')}]$ & "
                rf"{p6col} \\"
            )
    lines.append(r"\midrule")
lines[-1] = r"\bottomrule"
lines += [r"\end{tabular}", r"\end{table}"]
(OUT / "table_p6.tex").write_text("\n".join(lines), encoding="utf-8")
print("  [tex] table_p6.tex")

# ── 10. SUMMARY ───────────────────────────────────────────────────────────────

print("\n" + "="*66)
print("  EXPERIMENT B (P6) SUMMARY")
print("="*66)
for level in ["Level_1","Level_3","Level_5"]:
    p6t = RESULTS["P6_tests"].get(level, {})
    t_stars = p6t.get("T_stars", {})
    ok = p6t.get("monotone_decreasing_with_N","?")
    vals = "  ".join(
        f"{MODEL_LABELS[mk]}={t_stars[mk]:.0f}" for mk in model_keys if mk in t_stars
    )
    print(f"  {level}: {vals}  P6={'SUPPORTED' if ok else 'NOT SUPPORTED'}")
print(f"\n  Total elapsed: {elapsed_min():.1f} min")
print(f"  Outputs: {OUT}")
for fp in sorted(OUT.iterdir()):
    print(f"    {fp.name:48s}  {fp.stat().st_size/1024:7.1f} KB")
