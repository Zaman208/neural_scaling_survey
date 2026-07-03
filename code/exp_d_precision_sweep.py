#!/usr/bin/env python3
"""
EXPERIMENT D: Precision Sweep Extension (P3 — improved)
=========================================================
Tests Proposition P3: T*(fp16) > T*(bf16) > T*(int8) > T*(int4)
(higher precision → later overthinking threshold)

Improvements over paper2's precision section
  • Adds BF16 between FP16 and INT8 — four-point trend instead of three
  • Runs all three difficulty levels, not just Level_3
  • 3 independent samples per run (was 2) → tighter accuracy estimates
  • Checkpointed per (precision, level) — fine-grained crash recovery

Protocol
  • Model  : microsoft/Phi-3.5-mini-instruct
  • Levels : Level_1, Level_3, Level_5
  • Probs  : 10 per level
  • Budgets: [300, 800, 1600, 3200, 6400]
  • Precs  : fp16, bf16, int8, int4  (model reloaded at each precision)
  • Samples: 3 per (problem, budget)

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

OUT = Path("/kaggle/working/results_part6")
OUT.mkdir(parents=True, exist_ok=True)

#  1. CONSTANTS 

MODEL_ID  = "microsoft/Phi-3.5-mini-instruct"
BUDGETS   = [300, 800, 1600, 3200, 6400]
N_SAMPLES = 3
N_PROBS   = 10
SESSION_LIMIT_MIN = 510

# Ordered from highest to lowest precision (P3 expects T* in same order)
PRECISIONS = ["fp16", "bf16", "int8", "int4"]
PREC_LABELS = {
    "fp16": "FP16 (full)",
    "bf16": "BF16 (brain float)",
    "int8": "INT8 (8-bit)",
    "int4": "INT4 (4-bit)",
}
PREC_COLS = {
    "fp16": "#2166AC",
    "bf16": "#4DAC26",
    "int8": "#F4A582",
    "int4": "#D6604D",
}
LEVEL_LABELS = {
    "Level_1": "Level 1 (easy)",
    "Level_3": "Level 3 (medium)",
    "Level_5": "Level 5 (hard)",
}
LEVEL_COLS = {"Level_1":"#2166AC","Level_3":"#F4A582","Level_5":"#D6604D"}

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

#  4. g-MODEL FIT 

def g_raw(T, Delta, sigma_r2, k):
    V = np.maximum(k * np.asarray(T, dtype=float), 0.0)
    s = np.maximum(sigma_r2 + V, 1e-12)
    return (1.0 / np.sqrt(2.0 * np.pi * s)) * np.exp(-Delta**2 / (2.0 * s))

def g_scaled(T, Delta, sigma_r2, k, scale, offset):
    return scale * g_raw(T, Delta, sigma_r2, k) + offset

def fit_g(T_arr, A_arr, n_boot=400):
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
    n   = len(T)
    rss = float(np.sum((A - g_scaled(T, *best_p))**2))
    kp  = len(best_p)
    aic = n * np.log(rss / n + 1e-15) + 2 * kp
    bic = n * np.log(rss / n + 1e-15) + kp * np.log(n)
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
        "AIC":float(aic),"BIC":float(bic),
        "CI":ci,"n_data":int(n),
    }

#  5. MODEL LOADING 

def load_model_at_precision(model_id, prec):
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    if prec == "fp16":
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.float16,
            device_map="auto", trust_remote_code=False,
        )
    elif prec == "bf16":
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.bfloat16,
            device_map="auto", trust_remote_code=False,
        )
    elif prec == "int8":
        bnb = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, quantization_config=bnb,
            device_map="auto", trust_remote_code=False,
        )
    elif prec == "int4":
        bnb = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id, quantization_config=bnb,
            device_map="auto", trust_remote_code=False,
        )
    else:
        raise ValueError(f"Unknown precision: {prec}")

    model.eval()
    return model, tok

def release(model):
    del model; gc.collect(); torch.cuda.empty_cache()

def build_prompt(tok, problem):
    messages = [
        {"role": "system", "content": SYS_PROMPT},
        {"role": "user",   "content": problem},
    ]
    try:
        text = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        text = f"<|system|>\n{SYS_PROMPT}<|end|>\n<|user|>\n{problem}<|end|>\n<|assistant|>\n"
    return tok(text, return_tensors="pt").to(DEVICE)

#  6. SWEEP 

def run_sweep(model, tok, problems, budgets, n_samples=N_SAMPLES):
    results = []
    for budget in budgets:
        all_correct = []
        for problem, expected in problems:
            inputs = build_prompt(tok, problem)
            in_len = inputs["input_ids"].shape[1]
            run_correct = []
            for _ in range(n_samples):
                with torch.no_grad():
                    out = model.generate(
                        **inputs,
                        max_new_tokens=budget,
                        do_sample=True, temperature=0.7, top_p=0.9,
                        pad_token_id=tok.eos_token_id,
                        return_dict_in_generate=True,
                    )
                # out is GenerateOutput if return_dict_in_generate=True, else raw tensor
                seq = out.sequences if hasattr(out, "sequences") else out
                gen_ids  = seq[0][in_len:]
                gen_text = tok.decode(gen_ids, skip_special_tokens=True)
                run_correct.append(int(answer_is_correct(gen_text, expected)))
            all_correct.append(float(np.mean(run_correct)))
        acc = float(np.mean(all_correct))
        results.append({"budget": budget, "accuracy": acc})
        print(f"      T={budget:5d}  acc={acc:.3f}")
    return results

#  7. MAIN EXPERIMENT LOOP 

ckpt = load_checkpoint()
print(f"\n[Checkpoint] {len(ckpt)} entries already done.")

for prec in PRECISIONS:
    if not time_ok():
        print(f"\n[TIME] approaching session limit — stopping.")
        break

    levels_needed = [lv for lv in MATH_PROBLEMS
                     if f"{prec}_{lv}" not in ckpt]
    if not levels_needed:
        print(f"\n[SKIP] {prec} — all levels done.")
        continue

    print(f"\n{'='*66}")
    print(f"  Loading {MODEL_ID} at {prec.upper()}  [{elapsed_min():.1f} min]")
    print(f"  Levels remaining: {levels_needed}")
    print(f"{'='*66}")

    try:
        model, tok = load_model_at_precision(MODEL_ID, prec)
        print(f"  Loaded  [{elapsed_min():.1f} min]")
    except Exception as e:
        print(f"  ERROR loading {prec}: {e}")
        ckpt[f"{prec}_load_error"] = str(e)
        save_checkpoint(ckpt)
        continue

    for level in levels_needed:
        ck_key = f"{prec}_{level}"
        if ck_key in ckpt:
            continue
        if not time_ok():
            print(f"  [TIME] stopping before {ck_key}")
            break

        problems = MATH_PROBLEMS[level][:N_PROBS]
        print(f"\n  [{prec}][{level}]  "
              f"{len(problems)} probs × {len(BUDGETS)} budgets × {N_SAMPLES} runs")

        sweep = run_sweep(model, tok, problems, BUDGETS)
        T_arr = np.array([r["budget"]       for r in sweep])
        A_arr = np.array([r["accuracy"]*100 for r in sweep])
        fit   = fit_g(T_arr, A_arr)

        print(f"  -> R²={fit.get('R2',0):.4f}  T*={fit.get('T_star',0):.0f}")

        ckpt[ck_key] = {"sweep": sweep, "fit": fit,
                        "prec": prec, "level": level,
                        "elapsed_min": elapsed_min()}
        save_checkpoint(ckpt)
        print(f"  [ckpt] {ck_key}  [{elapsed_min():.1f} min]")

    release(model)
    print(f"  [released] {prec}  [{elapsed_min():.1f} min]")

#  8. ASSEMBLE RESULTS 

RESULTS = {"sweeps": {}, "fits": {}, "P3_tests": {}}

for prec in PRECISIONS:
    RESULTS["sweeps"][prec] = {}
    RESULTS["fits"][prec]   = {}
    for level in MATH_PROBLEMS:
        ck_key = f"{prec}_{level}"
        if ck_key in ckpt and "sweep" in ckpt[ck_key]:
            RESULTS["sweeps"][prec][level] = ckpt[ck_key]["sweep"]
            RESULTS["fits"][prec][level]   = ckpt[ck_key]["fit"]

# P3 test per level: T* monotone-decreasing with quantisation?
for level in MATH_PROBLEMS:
    t_stars = {}
    for prec in PRECISIONS:
        fit = RESULTS["fits"].get(prec, {}).get(level, {})
        if "T_star" in fit:
            t_stars[prec] = fit["T_star"]
    if len(t_stars) >= 2:
        avail = [p for p in PRECISIONS if p in t_stars]
        mono  = all(t_stars[avail[i]] >= t_stars[avail[i+1]]
                    for i in range(len(avail)-1))
        RESULTS["P3_tests"][level] = {
            "T_stars": t_stars,
            "precisions_available": avail,
            "monotone_decreasing": mono,
        }
        print(f"  [{level}] P3: "
              + "  ".join(f"{p}={t_stars[p]:.0f}" for p in avail)
              + f"  supported={'✓' if mono else '✗'}")

out_json = OUT / "precision_results.json"
with open(out_json, "w") as f:
    json.dump(RESULTS, f, indent=2, default=str)
print(f"\n[json] precision_results.json  [{elapsed_min():.1f} min]")

#  9. FIGURES 

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

#  Figure D1: Level_3 curves for all 4 precisions 
level_main = "Level_3"
has_any = any(level_main in RESULTS["sweeps"].get(p, {}) for p in PRECISIONS)
if has_any:
    fig, ax = plt.subplots(figsize=(8, 4.8))
    T_dense = np.linspace(BUDGETS[0], BUDGETS[-1]*1.4, 2000)
    for prec in PRECISIONS:
        data = RESULTS["sweeps"].get(prec, {}).get(level_main)
        if not data:
            continue
        col = PREC_COLS[prec]
        T_d = np.array([r["budget"]       for r in data])
        A_d = np.array([r["accuracy"]*100 for r in data])
        ax.scatter(T_d/1000, A_d, color=col, s=50, zorder=6)
        ax.plot(T_d/1000, A_d, "o-", color=col, lw=1.3, alpha=0.4)
        fit = RESULTS["fits"].get(prec, {}).get(level_main, {})
        if "Delta" in fit:
            popt = [fit["Delta"],fit["sigma_r2"],fit["k"],fit["scale"],fit["offset"]]
            Tf   = T_dense[T_dense <= T_d.max()*1.3]
            ax.plot(Tf/1000, g_scaled(Tf, *popt), color=col, lw=2.2,
                    label=f"{PREC_LABELS[prec]}  T*={fit['T_star']:.0f}")
            ax.axvline(fit["T_star"]/1000, color=col, lw=0.9, ls=":", alpha=0.7)
    ax.set_xlabel("Budget $T$ (thousands of tokens)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Experiment D: Precision sweep — Level 3 (medium)\n"
                 r"P3 prediction: $T^*(\mathrm{fp16})>T^*(\mathrm{bf16})>"
                 r"T^*(\mathrm{int8})>T^*(\mathrm{int4})$")
    ax.legend(fontsize=9)
    plt.tight_layout()
    savefig("figure_precision_curves", fig)

#  Figure D2: T* bar chart across all levels 
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
for ax_i, level in enumerate(["Level_1","Level_3","Level_5"]):
    ax  = axes[ax_i]
    p3t = RESULTS["P3_tests"].get(level, {})
    ts  = p3t.get("T_stars", {})
    avail = [p for p in PRECISIONS if p in ts]
    if not avail:
        ax.set_visible(False); continue
    vals = [ts[p] for p in avail]
    cols = [PREC_COLS[p] for p in avail]
    lbls = [PREC_LABELS[p] for p in avail]
    bars = ax.bar(range(len(avail)), vals, color=cols, alpha=0.82,
                  width=0.6, zorder=3)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, v+max(vals)*0.02,
                f"{v:.0f}", ha="center", fontsize=9, fontweight="bold")
    ax.set_xticks(range(len(avail)))
    ax.set_xticklabels(lbls, fontsize=8, rotation=15)
    ax.set_ylabel("$T^*$ (tokens)")
    mono = p3t.get("monotone_decreasing", False)
    ax.set_title(f"{LEVEL_LABELS[level]}\nP3 {'✓ supported' if mono else '✗ not supported'}")
    ax.grid(axis="x", alpha=0)
plt.suptitle("Experiment D: T* vs precision (Phi-3.5-mini)\n"
             "P3 predicts monotone decrease from FP16 → INT4",
             fontsize=11, y=1.02)
plt.tight_layout()
savefig("figure_precision_tstar", fig)

#  10. LaTeX TABLE 

def tex(v, fmt=".3f"):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    return format(v, fmt)

lines = [
    "% AUTO-GENERATED by part6_precision_extended.py",
    r"\begin{table}[t]",
    r"\centering\small",
    r"\caption{Experiment D (P3): overthinking threshold $T^*$ by weight "
    r"precision and difficulty level (Phi-3.5-mini, 10 problems/level, "
    r"3 runs/budget). P3 predicts $T^*$ monotone-decreasing from FP16 to INT4. "
    r"\checkmark = supported.}",
    r"\label{tab:precision-extended}",
    r"\begin{tabular}{lcccc}",
    r"\toprule",
    r"Level & FP16 & BF16 & INT8 & INT4 \\",
    r"\midrule",
]
for level in ["Level_1","Level_3","Level_5"]:
    vals = []
    for prec in PRECISIONS:
        fit = RESULTS["fits"].get(prec, {}).get(level, {})
        vals.append(f"${tex(fit.get('T_star'),'.0f')}$" if "T_star" in fit else "—")
    p3t  = RESULTS["P3_tests"].get(level, {})
    mono = r"\checkmark" if p3t.get("monotone_decreasing") else r"\texttimes"
    lines.append(
        rf"{LEVEL_LABELS[level]} & {' & '.join(vals)} \\ "
        rf"\quad[P3: {mono}]"
    )
lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
(OUT / "table_precision_extended.tex").write_text("\n".join(lines), encoding="utf-8")
print("  [tex] table_precision_extended.tex")

#  11. SUMMARY 

print("\n" + "="*66)
print("  EXPERIMENT D (P3 — PRECISION SWEEP) SUMMARY")
print("="*66)
for level in ["Level_1","Level_3","Level_5"]:
    p3t = RESULTS["P3_tests"].get(level, {})
    ts  = p3t.get("T_stars", {})
    mono = p3t.get("monotone_decreasing", "?")
    row  = "  ".join(f"{p}={ts[p]:.0f}" for p in PRECISIONS if p in ts)
    print(f"  {level}: {row}  P3={'SUPPORTED' if mono else 'NOT SUPPORTED'}")
print(f"\n  Total elapsed: {elapsed_min():.1f} min")
print(f"  Outputs: {OUT}")
for fp in sorted(OUT.iterdir()):
    print(f"    {fp.name:48s}  {fp.stat().st_size/1024:7.1f} KB")
