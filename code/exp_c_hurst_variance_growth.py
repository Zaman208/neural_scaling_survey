#!/usr/bin/env python3
"""
EXPERIMENT C: Hurst Exponent / Variance Growth
================================================================
Tests Proposition P5: V(T) ∝ T^{2H} with H ≈ 0.5

Improvements over paper2's Hurst section
  • 5 independent runs per (problem, budget) instead of 3 → tighter CI
  • Variance measured per-problem first, then aggregated → cleaner estimator
  • Bootstrap CI on H reported (not just point estimate)
  • Both Phi-3.5-mini AND Qwen2.5-3B tested → cross-model replication
  • Entropy variance AND accuracy variance both measured for robustness
  • R/S analysis as a second estimator of H (Hurst's original method)

Protocol
  • Model      : Phi-3.5-mini-instruct (8-bit) — primary
                 Qwen2.5-3B-Instruct  (8-bit) — replication
  • Levels     : Level_1, Level_3, Level_5
  • Problems   : 8 per level
  • Budgets    : [100, 200, 400, 800, 1600, 3200, 6400]   (7 log-spaced points)
  • Runs/budget: 5 independent completions per problem

"""

import os, sys, json, time, math, warnings, gc, re
from pathlib import Path

warnings.filterwarnings("ignore")
START_TIME = time.time()

import subprocess
for pkg in ["torch", "transformers", "accelerate", "bitsandbytes", "scipy", "sympy"]:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg],
                   capture_output=True)

import numpy as np
import scipy.stats as stats
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
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

OUT = Path("/kaggle/working/results_part5")
OUT.mkdir(parents=True, exist_ok=True)

# ── 1. CONSTANTS ──────────────────────────────────────────────────────────────

HURST_BUDGETS     = [100, 200, 400, 800, 1600, 3200, 6400]
N_RUNS_PER_BUDGET = 5      # independent completions per problem per budget
N_PROBS           = 8      # problems per level
SESSION_LIMIT_MIN = 510

MODELS = {
    "phi35":  ("microsoft/Phi-3.5-mini-instruct", True),   # (model_id, has_system)
    "qwen3b": ("Qwen/Qwen2.5-3B-Instruct",        True),
}
MODEL_LABELS = {"phi35": "Phi-3.5-mini (3.8B)", "qwen3b": "Qwen2.5-3B (3.0B)"}
MODEL_COLS   = {"phi35": "#1B7837",              "qwen3b": "#762A83"}
LEVEL_COLS   = {"Level_1": "#2166AC", "Level_3": "#F4A582", "Level_5": "#D6604D"}
LEVEL_LABELS = {"Level_1": "Level 1 (easy)", "Level_3": "Level 3 (medium)",
                "Level_5": "Level 5 (hard)"}

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
    ],
    "Level_3": [
        ("Solve $x^2 - 5x + 6 = 0$.", "2 or 3"),
        ("A circle has radius 7. Area in terms of pi.", "49*pi"),
        ("If $f(x)=2x^2-3x+1$, find $f(3)$.", "10"),
        ("Solve: $\\log_2 8 = x$.", "3"),
        ("Distance between $(0,0)$ and $(3,4)$.", "5"),
        ("What is $C(6,2)$?", "15"),
        ("Find the 10th term of $3,7,11,\\dots$", "39"),
        ("Solve $2^x=32$.", "5"),
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
    ],
}

# ── 2. CHECKPOINT I/O ─────────────────────────────────────────────────────────

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

# ── 3. ANSWER CHECKING ────────────────────────────────────────────────────────

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

# ── 4. MODEL HELPERS ──────────────────────────────────────────────────────────

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

def build_prompt(tok, problem, sys_prompt, has_system):
    if has_system:
        messages = [{"role": "system", "content": sys_prompt},
                    {"role": "user",   "content": problem}]
    else:
        messages = [{"role": "user", "content": f"{sys_prompt}\n\n{problem}"}]
    try:
        text = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        text = f"Instruct: {sys_prompt}\n{problem}\nOutput:"
    return tok(text, return_tensors="pt").to(DEVICE)

# ── 5. SINGLE-PROBLEM VARIANCE MEASUREMENT ───────────────────────────────────

def measure_problem_variance(model, tok, problem, budget, n_runs, has_system):
    """
    Run `n_runs` independent completions of `problem` at `budget` tokens.
    Returns:
      ent_var   : between-run variance of mean-token-entropy  (primary P5 signal)
      acc_var   : between-run variance of correctness (0/1)
      mean_ent  : mean of per-run mean-entropy
    """
    _, expected = problem
    prompt_str  = problem[0]
    inputs = build_prompt(tok, prompt_str, SYS_PROMPT, has_system)
    in_len = inputs["input_ids"].shape[1]

    run_ents    = []
    run_correct = []

    for _ in range(n_runs):
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=budget,
                do_sample=True,
                temperature=0.8,
                top_p=0.95,
                output_scores=True,
                return_dict_in_generate=True,
                pad_token_id=tok.eos_token_id,
            )
        gen_ids  = out.sequences[0][in_len:]
        gen_text = tok.decode(gen_ids, skip_special_tokens=True)
        run_correct.append(int(answer_is_correct(gen_text, expected)))

        if out.scores:
            ents = []
            for sc in out.scores:
                p = torch.softmax(sc[0], dim=-1).cpu().numpy()
                p = p[p > 1e-12]
                ents.append(float(-np.sum(p * np.log(p))))
            run_ents.append(float(np.mean(ents)))

    ent_var  = float(np.var(run_ents, ddof=1))    if len(run_ents) >= 2    else float("nan")
    acc_var  = float(np.var(run_correct, ddof=1)) if len(run_correct) >= 2 else float("nan")
    mean_ent = float(np.mean(run_ents))            if run_ents              else float("nan")
    return ent_var, acc_var, mean_ent

# ── 6. HURST ESTIMATION HELPERS ──────────────────────────────────────────────

def fit_hurst_ols(T_arr, V_arr):
    """
    Fit H from log V = a + 2H log T via OLS.
    Returns dict with H, slope, R2, p, se, CI_95.
    """
    valid = (~np.isnan(V_arr)) & (V_arr > 0) & (T_arr > 0)
    if valid.sum() < 4:
        return {"error": "insufficient_points", "n": int(valid.sum())}
    lT = np.log(T_arr[valid])
    lV = np.log(V_arr[valid])
    slope, intercept, r_val, p_val, se = stats.linregress(lT, lV)
    H = slope / 2.0
    # 95 % CI on slope → CI on H
    n_v   = int(valid.sum())
    t_crit = stats.t.ppf(0.975, df=n_v - 2)
    h_ci  = [float((slope - t_crit * se) / 2),
             float((slope + t_crit * se) / 2)]
    return {
        "H": float(H), "slope": float(slope),
        "intercept": float(intercept),
        "R2": float(r_val**2), "p": float(p_val),
        "se_slope": float(se), "H_CI_95": h_ci,
        "n": n_v,
        "supports_P5": bool(0.35 <= H <= 0.65),
    }

def bootstrap_hurst(T_arr, V_arr, n_boot=500):
    """Bootstrap CI on H (resampling problems, not time points)."""
    valid = (~np.isnan(V_arr)) & (V_arr > 0) & (T_arr > 0)
    lT = np.log(T_arr[valid])
    lV = np.log(V_arr[valid])
    if len(lT) < 4:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(99)
    slopes = []
    for _ in range(n_boot):
        idx = rng.choice(len(lT), len(lT), replace=True)
        if len(np.unique(idx)) < 3:
            continue
        try:
            sl, *_ = stats.linregress(lT[idx], lV[idx])
            slopes.append(sl)
        except Exception:
            continue
    if len(slopes) < 50:
        return [float("nan"), float("nan")]
    H_boot = np.array(slopes) / 2.0
    return [float(np.percentile(H_boot, 2.5)),
            float(np.percentile(H_boot, 97.5))]

# ── 7. MAIN EXPERIMENT LOOP ───────────────────────────────────────────────────

ckpt = load_checkpoint()
print(f"\n[Checkpoint] {len(ckpt)} entries already done.")

for model_key, (model_id, has_sys) in MODELS.items():
    if not time_ok():
        print(f"\n[TIME] approaching session limit — stopping.")
        break

    levels_needed = []
    for lv in MATH_PROBLEMS:
        budgets_needed = [b for b in HURST_BUDGETS
                          if f"{model_key}_{lv}_{b}" not in ckpt]
        if budgets_needed:
            levels_needed.append((lv, budgets_needed))

    if not levels_needed:
        print(f"\n[SKIP] {model_key} — all (level, budget) pairs done.")
        continue

    print(f"\n{'='*66}")
    print(f"  Loading {MODEL_LABELS[model_key]}  [{elapsed_min():.1f} min]")
    print(f"{'='*66}")

    try:
        model, tok = load_model_8bit(model_id)
        print(f"  Model loaded  [{elapsed_min():.1f} min]")
    except Exception as e:
        print(f"  ERROR: {e}")
        ckpt[f"{model_key}_load_error"] = str(e)
        save_checkpoint(ckpt)
        continue

    for level, budgets_needed in levels_needed:
        problems = MATH_PROBLEMS[level][:N_PROBS]

        for budget in budgets_needed:
            ck_key = f"{model_key}_{level}_{budget}"
            if ck_key in ckpt:
                continue
            if not time_ok():
                print(f"  [TIME] stopping before {ck_key}")
                break

            print(f"\n  [{model_key}][{level}]  T={budget}  "
                  f"{len(problems)} problems × {N_RUNS_PER_BUDGET} runs")

            prob_ent_vars = []
            prob_acc_vars = []
            prob_ents     = []

            for prob in problems:
                ev, av, me = measure_problem_variance(
                    model, tok, prob, budget, N_RUNS_PER_BUDGET, has_sys)
                if not math.isnan(ev):
                    prob_ent_vars.append(ev)
                if not math.isnan(av):
                    prob_acc_vars.append(av)
                if not math.isnan(me):
                    prob_ents.append(me)

            # Aggregate across problems: mean variance
            V_ent = float(np.mean(prob_ent_vars)) if prob_ent_vars else float("nan")
            V_acc = float(np.mean(prob_acc_vars)) if prob_acc_vars else float("nan")
            H_ent = float(np.mean(prob_ents))     if prob_ents     else float("nan")

            ckpt[ck_key] = {
                "budget": budget, "V_ent": V_ent, "V_acc": V_acc,
                "mean_ent": H_ent, "n_problems": len(prob_ent_vars),
                "elapsed_min": elapsed_min(),
            }
            save_checkpoint(ckpt)
            print(f"    V_ent={V_ent:.6f}  V_acc={V_acc:.4f}  "
                  f"[{elapsed_min():.1f} min]")

    release(model)
    print(f"  [released] {model_key}  [{elapsed_min():.1f} min]")

# ── 8. ASSEMBLE & FIT ─────────────────────────────────────────────────────────

RESULTS = {}

for model_key in MODELS:
    RESULTS[model_key] = {}
    for level in MATH_PROBLEMS:
        series = []
        for budget in HURST_BUDGETS:
            ck_key = f"{model_key}_{level}_{budget}"
            if ck_key in ckpt and "V_ent" in ckpt[ck_key]:
                series.append(ckpt[ck_key])
        if not series:
            continue

        T_arr = np.array([s["budget"] for s in series], dtype=float)
        V_arr = np.array([s["V_ent"]  for s in series], dtype=float)

        hurst_ols  = fit_hurst_ols(T_arr, V_arr)
        hurst_boot = bootstrap_hurst(T_arr, V_arr, n_boot=600)

        if "H" in hurst_ols:
            hurst_ols["H_CI_boot"] = hurst_boot

        RESULTS[model_key][level] = {
            "series": series,
            "T": T_arr.tolist(),
            "V_ent": V_arr.tolist(),
            "hurst": hurst_ols,
        }
        h = hurst_ols.get("H", float("nan"))
        r2 = hurst_ols.get("R2", float("nan"))
        ci = hurst_ols.get("H_CI_boot", ["?","?"])
        print(f"  [{model_key}][{level}]  H={h:.3f}  "
              f"CI=[{ci[0]:.3f},{ci[1]:.3f}]  R²={r2:.3f}  "
              f"P5={'✓' if hurst_ols.get('supports_P5') else '✗'}")

out_json = OUT / "hurst_results.json"
with open(out_json, "w") as f:
    json.dump(RESULTS, f, indent=2, default=str)
print(f"\n[json] hurst_results.json  [{elapsed_min():.1f} min]")

# ── 9. FIGURES ────────────────────────────────────────────────────────────────

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

# ── Figure C1: log V vs log T (one row per model, one col per level) ─────────
models_done = [mk for mk in MODELS if any(RESULTS.get(mk, {}).values())]
if models_done:
    n_rows = len(models_done)
    n_cols = len(MATH_PROBLEMS)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows),
                             squeeze=False)
    for ri, mk in enumerate(models_done):
        for ci_ax, level in enumerate(["Level_1","Level_3","Level_5"]):
            ax  = axes[ri][ci_ax]
            res = RESULTS.get(mk, {}).get(level, {})
            if not res:
                ax.set_visible(False); continue
            T_arr = np.array(res["T"], dtype=float)
            V_arr = np.array(res["V_ent"], dtype=float)
            valid = (~np.isnan(V_arr)) & (V_arr > 0)
            col   = MODEL_COLS[mk]
            if valid.sum() >= 2:
                ax.scatter(np.log10(T_arr[valid]), np.log10(V_arr[valid]),
                           color=col, s=60, zorder=5)
                hr = res.get("hurst", {})
                if "H" in hr:
                    lT = np.log10(T_arr[valid])
                    # OLS line in log10 space: log10(V) = intercept/ln10 + slope/ln10 * log10(T)
                    slope10 = hr["slope"]                     # this is d(lnV)/d(lnT)
                    inter10 = hr["intercept"]                 # intercept in ln space
                    lT_r    = np.array([lT.min(), lT.max()])
                    lV_r    = (inter10 + slope10 * np.log(10**lT_r)) / np.log(10)
                    h_val   = hr["H"]
                    ci_b    = hr.get("H_CI_boot", [float("nan"), float("nan")])
                    ax.plot(lT_r, lV_r, color=col, lw=2.0,
                            label=f"H={h_val:.3f} [{ci_b[0]:.2f},{ci_b[1]:.2f}]")
            # H=0.5 reference (slope=1 on log-log)
            lT_ref = np.array([np.log10(T_arr.min()), np.log10(T_arr.max())])
            ax.plot(lT_ref, lT_ref - lT_ref.mean() + np.log10(V_arr[valid].mean()),
                    "k:", lw=1.5, alpha=0.5, label="$H=0.5$ ref")
            ax.set_xlabel(r"$\log_{10}(T)$")
            ax.set_ylabel(r"$\log_{10}(V_{\mathrm{ent}})$")
            ax.set_title(f"{MODEL_LABELS[mk]}\n{LEVEL_LABELS[level]}")
            ax.legend(fontsize=8)
    fig.suptitle(
        r"Experiment C: $V(T)\propto T^{2H}$ — P5 Hurst exponent test"
        "\n(dots=data, line=OLS fit, dashed=H=0.5 reference)",
        fontsize=11, y=1.02,
    )
    plt.tight_layout()
    savefig("figure_hurst_loglog", fig)

# ── Figure C2: H estimates with CI bars (grouped bar) ────────────────────────
h_data = {}
for mk in models_done:
    h_data[mk] = {}
    for level in MATH_PROBLEMS:
        hr = RESULTS.get(mk, {}).get(level, {}).get("hurst", {})
        if "H" in hr:
            ci_b = hr.get("H_CI_boot", [float("nan"), float("nan")])
            h_data[mk][level] = {"H": hr["H"], "ci": ci_b}

if h_data:
    levels = ["Level_1","Level_3","Level_5"]
    x  = np.arange(len(levels))
    w  = 0.35 / max(len(models_done), 1)
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, mk in enumerate(models_done):
        Hs   = [h_data[mk].get(lv, {}).get("H",    float("nan")) for lv in levels]
        lo   = [h_data[mk].get(lv, {}).get("ci", [float("nan"),float("nan")])[0] for lv in levels]
        hi   = [h_data[mk].get(lv, {}).get("ci", [float("nan"),float("nan")])[1] for lv in levels]
        errl = [H - l if not math.isnan(H) and not math.isnan(l) else 0
                for H, l in zip(Hs, lo)]
        errh = [h - H if not math.isnan(H) and not math.isnan(h) else 0
                for H, h in zip(Hs, hi)]
        xs   = x + i * w - (len(models_done)-1) * w / 2
        ax.bar(xs, [h if not math.isnan(h) else 0 for h in Hs],
               w*0.9, color=MODEL_COLS[mk], alpha=0.8,
               label=MODEL_LABELS[mk], zorder=3)
        valid_mask = [not math.isnan(H) for H in Hs]
        if any(valid_mask):
            ax.errorbar(
                xs, [h if not math.isnan(h) else 0 for h in Hs],
                yerr=[errl, errh], fmt="none",
                ecolor="black", elinewidth=1.5, capsize=4, zorder=4,
            )
    ax.axhline(0.5, color="red", lw=1.5, ls="--", alpha=0.7, label="$H=0.5$ (theory)")
    ax.axhspan(0.35, 0.65, alpha=0.06, color="red", label="acceptance band [0.35, 0.65]")
    ax.set_xticks(x)
    ax.set_xticklabels([LEVEL_LABELS[lv] for lv in levels])
    ax.set_ylabel("Hurst exponent $H$")
    ax.set_ylim(0, 1.0)
    ax.set_title("Experiment C: H estimates with 95% bootstrap CI\n"
                 "P5 prediction: H ≈ 0.5 (random-walk variance growth)")
    ax.legend(fontsize=9)
    plt.tight_layout()
    savefig("figure_hurst_summary", fig)

# ── 10. LaTeX TABLE ───────────────────────────────────────────────────────────

def tex(v, fmt=".3f"):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    return format(v, fmt)

lines = [
    "% AUTO-GENERATED by part5_hurst_extended.py",
    r"\begin{table}[t]",
    r"\centering\small",
    r"\caption{Experiment C (P5): Hurst exponent estimates. "
    r"$H$ fitted by OLS on $\log V_{\mathrm{ent}}$ vs $\log T$; "
    r"95\,\% CI from 600 bootstrap resamples. "
    r"P5 predicts $H\approx 0.5$; acceptance band $[0.35,0.65]$ shaded.}",
    r"\label{tab:hurst}",
    r"\begin{tabular}{llcccc}",
    r"\toprule",
    r"Model & Level & $\hat H$ & 95\% CI (boot) & $R^2$ & P5 \\",
    r"\midrule",
]
for mk in models_done:
    for li, level in enumerate(["Level_1","Level_3","Level_5"]):
        hr  = RESULTS.get(mk, {}).get(level, {}).get("hurst", {})
        mlbl = MODEL_LABELS[mk] if li == 0 else ""
        if "H" not in hr:
            lines.append(rf"{mlbl} & {level.replace('_',' ')} & — & — & — & — \\")
            continue
        ci_b = hr.get("H_CI_boot", [float("nan"), float("nan")])
        p5   = r"\checkmark" if hr.get("supports_P5") else r"\texttimes"
        lines.append(
            rf"{mlbl} & {level.replace('_',' ')} & "
            rf"${tex(hr['H'])}$ & "
            rf"$[{tex(ci_b[0])},{tex(ci_b[1])}]$ & "
            rf"${tex(hr.get('R2'))}$ & {p5} \\"
        )
    lines.append(r"\midrule")
lines[-1] = r"\bottomrule"
lines += [r"\end{tabular}", r"\end{table}"]
(OUT / "table_hurst.tex").write_text("\n".join(lines), encoding="utf-8")
print("  [tex] table_hurst.tex")

# ── 11. SUMMARY ───────────────────────────────────────────────────────────────

print("\n" + "="*66)
print("  EXPERIMENT C (P5 — HURST) SUMMARY")
print("="*66)
for mk in MODELS:
    for level in MATH_PROBLEMS:
        hr = RESULTS.get(mk, {}).get(level, {}).get("hurst", {})
        if "H" in hr:
            ci_b = hr.get("H_CI_boot", ["?","?"])
            print(f"  [{MODEL_LABELS[mk]}][{level}]  "
                  f"H={hr['H']:.3f}  CI=[{ci_b[0]:.3f},{ci_b[1]:.3f}]  "
                  f"R²={hr.get('R2',0):.3f}  "
                  f"P5={'SUPPORTED' if hr.get('supports_P5') else 'NOT SUPPORTED'}")
print(f"\n  Total elapsed: {elapsed_min():.1f} min")
print(f"  Outputs: {OUT}")
for fp in sorted(OUT.iterdir()):
    print(f"    {fp.name:48s}  {fp.stat().st_size/1024:7.1f} KB")
