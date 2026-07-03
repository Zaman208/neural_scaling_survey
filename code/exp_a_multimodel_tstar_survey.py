#!/usr/bin/env python3
"""
EXPERIMENT A: Multi-Model Validation

Models tested
  Phi-3.5-mini-instruct   3.8 B   (microsoft)
  Qwen2.5-3B-Instruct     3.0 B   (Qwen)
  Gemma-2-2B-it           2.0 B   (Google)

Protocol (per model)
  • 3 difficulty levels  : Level_1 (easy), Level_3 (medium), Level_5 (hard)
  • 8 problems / level   : subset of the paper's MATH problem bank
  • Budgets              : [150, 300, 600, 1200, 2400, 4800]
  • 2 independent samples per (problem, budget)
  • 8-bit quantisation for all models

"""

import os, sys, json, time, math, warnings, gc, re
from pathlib import Path
from collections import Counter

warnings.filterwarnings("ignore")
START_TIME = time.time()

#  0. INSTALL 
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

OUT = Path("/kaggle/working/results_part3")
OUT.mkdir(parents=True, exist_ok=True)

#  1. CONSTANTS 

BUDGETS   = [150, 300, 600, 1200, 2400, 4800]
N_SAMPLES = 2       # independent completions per (problem, budget)
N_PROBS   = 8       # problems per level

SESSION_LIMIT_MIN = 510   # warn / soft-stop at 8.5 h

MODELS = {
    "phi35":  "microsoft/Phi-3.5-mini-instruct",
    "qwen3b": "Qwen/Qwen2.5-3B-Instruct",
    "gemma2": "google/gemma-2-2b-it",
}

MODEL_BITS = {"phi35": 8, "qwen3b": 8, "gemma2": 8}

# Which models support a "system" turn in their chat template
MODEL_HAS_SYSTEM = {"phi35": True, "qwen3b": True, "gemma2": False}

SYS_PROMPT = (
    "You are a precise math assistant. "
    "Think step by step. "
    "Write your final numerical answer after the word ANSWER: on its own line."
)

MATH_PROBLEMS = {
    "Level_1": [
        ("Compute $2^3 + 3^2$.", "17"),
        ("What is $15\\% $ of $200$?", "30"),
        ("If $x + 5 = 12$, find $x$.", "7"),
        ("What is $\\sqrt{49}$?", "7"),
        ("Calculate $3!$.", "6"),
        ("What is $4^2 - 3^2$?", "7"),
        ("Find $x$: $2x = 18$.", "9"),
        ("Compute $7 \\times 8$.", "56"),
        ("What is $2^5$?", "32"),
        ("Find the mean of $4, 6, 8, 10$.", "7"),
        ("What is $(-3)^2$?", "9"),
        ("If $y - 4 = 10$, find $y$.", "14"),
        ("Simplify $12/18$.", "2/3"),
        ("Perimeter of a square with side 5.", "20"),
        ("What is $10\\%$ of $350$?", "35"),
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
        ("Find the vertex of $y=x^2-4x+3$.", "2"),
        ("Arrangements: 4 books from 6?", "360"),
        ("Solve $2^x=32$.", "5"),
        ("Find $\\gcd(48,18)$.", "6"),
        ("P(A)=0.3, P(B)=0.5, independent. Find P(A and B).", "0.15"),
        ("Derivative of $x^3$.", "3*x**2"),
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
        ("Critical points of f(x)=x^3-3*x.", "1 or -1"),
        ("Rank of [[1,2,3],[4,5,6],[7,8,9]].", "2"),
        ("How many primes between 1 and 50?", "15"),
        ("P(exactly 5 heads in 10 coin flips).", "63/256"),
        ("Find integer solutions to x^2+y^2=25.", "3 and 4"),
    ],
}

LEVEL_LABELS = {
    "Level_1": "Level 1 (easy)",
    "Level_3": "Level 3 (medium)",
    "Level_5": "Level 5 (hard)",
}
LEVEL_COLS = {"Level_1": "#2166AC", "Level_3": "#F4A582", "Level_5": "#D6604D"}
MODEL_COLS = {"phi35": "#1B7837", "qwen3b": "#762A83", "gemma2": "#E08214"}
MODEL_LABELS = {
    "phi35":  "Phi-3.5-mini (3.8B)",
    "qwen3b": "Qwen2.5-3B (3.0B)",
    "gemma2": "Gemma-2-2B (2.0B)",
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
        (r"\\pi","pi"), (r"\\ln","log"), (r"\\log","log"),
        (r"\\sin","sin"), (r"\\cos","cos"), (r"\\cdot","*"),
        (r"\\times","*"), (r"\\infty","oo"),
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
        for i, nm in enumerate(["Delta","sigma_r2","k","scale","offset"]):
            ci[nm] = [float(np.percentile(ba[:,i], 2.5)),
                      float(np.percentile(ba[:,i], 97.5))]
        bt_s = []
        for pb in ba:
            Tf2 = np.linspace(T.min(), T.max() * 2.5, 2000)
            bt_s.append(float(Tf2[np.argmax(g_scaled(Tf2, *pb))]))
        ci["T_star"] = [float(np.percentile(bt_s, 2.5)),
                        float(np.percentile(bt_s, 97.5))]
    return {
        "Delta": float(Delta), "sigma_r2": float(sr2),
        "k": float(k), "scale": float(sc), "offset": float(off),
        "R2": float(best_r2), "T_star": T_star, "peak_acc": peak,
        "AIC": float(aic), "BIC": float(bic),
        "CI": ci, "n_data": int(n),
    }

#  5. MODEL HELPERS 

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
    """Return a tokenised prompt using the model's chat template."""
    if has_system:
        messages = [
            {"role": "system",    "content": sys_prompt},
            {"role": "user",      "content": problem},
        ]
    else:
        # Gemma-2 and similar: fold system prompt into user turn
        messages = [{"role": "user", "content": f"{sys_prompt}\n\n{problem}"}]
    try:
        text = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        # Fallback for models without a registered chat template
        if has_system:
            text = f"System: {sys_prompt}\nUser: {problem}\nAssistant:"
        else:
            text = f"{sys_prompt}\n{problem}\nAnswer:"
    return tok(text, return_tensors="pt").to(DEVICE)

#  6. SWEEP FUNCTION 

def run_sweep(model, tok, problems, budgets, model_key, n_samples=N_SAMPLES):
    """
    Returns list[dict] with one entry per budget:
      {"budget": int, "accuracy": float, "mean_ent": float, "V_between": float}
    """
    has_sys = MODEL_HAS_SYSTEM.get(model_key, True)
    results = []
    for budget in budgets:
        all_correct  = []
        all_ent      = []
        all_V_between = []
        for problem, expected in problems:
            inputs = build_prompt(tok, problem, SYS_PROMPT, has_sys)
            in_len = inputs["input_ids"].shape[1]
            run_correct = []
            run_ents    = []
            for _ in range(n_samples):
                with torch.no_grad():
                    out = model.generate(
                        **inputs,
                        max_new_tokens=budget,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9,
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
            all_correct.append(float(np.mean(run_correct)))
            if len(run_ents) >= 2:
                all_V_between.append(float(np.var(run_ents, ddof=1)))
            if run_ents:
                all_ent.append(float(np.mean(run_ents)))
        acc       = float(np.mean(all_correct))
        mean_ent  = float(np.mean(all_ent))      if all_ent      else float("nan")
        V_between = float(np.mean(all_V_between)) if all_V_between else float("nan")
        results.append({"budget": budget, "accuracy": acc,
                        "mean_ent": mean_ent, "V_between": V_between})
        print(f"      T={budget:5d}  acc={acc:.3f}  H={mean_ent:.4f}")
    return results

#  7. MAIN EXPERIMENT LOOP 

ckpt = load_checkpoint()
print(f"\n[Checkpoint] loaded {len(ckpt)} completed (model, level) pairs.")

for model_key, model_id in MODELS.items():
    if not time_ok():
        print(f"\n[WARNING] Approaching session limit at {elapsed_min():.1f} min — stopping.")
        break

    # check if ALL levels done for this model
    levels_needed = [lv for lv in MATH_PROBLEMS
                     if f"{model_key}_{lv}" not in ckpt]
    if not levels_needed:
        print(f"\n[SKIP] {model_key} — all levels already in checkpoint.")
        continue

    print(f"\n{'='*66}")
    print(f"  Loading {MODEL_LABELS[model_key]}  ({model_id})  [{elapsed_min():.1f} min]")
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
        if ck_key in ckpt:
            print(f"  [skip] {ck_key} already done.")
            continue
        if not time_ok():
            print(f"  [TIME] stopping before {ck_key}")
            break

        problems = MATH_PROBLEMS[level][:N_PROBS]
        print(f"\n  [{model_key}][{level}]  "
              f"{len(problems)} problems × {len(BUDGETS)} budgets × {N_SAMPLES} samples")

        sweep = run_sweep(model, tok, problems, BUDGETS, model_key)

        T_arr = np.array([r["budget"]       for r in sweep])
        A_arr = np.array([r["accuracy"]*100 for r in sweep])
        fit   = fit_g(T_arr, A_arr, n_boot=400)

        print(f"  -> R²={fit.get('R2',0):.4f}  "
              f"T*={fit.get('T_star',0):.0f}  "
              f"peak={fit.get('peak_acc',0):.1f}%")

        ckpt[ck_key] = {"sweep": sweep, "fit": fit,
                        "model_id": model_id, "level": level,
                        "elapsed_min": elapsed_min()}
        save_checkpoint(ckpt)
        print(f"  [ckpt] {ck_key} saved  [{elapsed_min():.1f} min]")

    release(model)
    print(f"  [released] {model_key}  [{elapsed_min():.1f} min]")

#  8. ASSEMBLE FINAL RESULTS 

RESULTS = {"models": {}, "fits": {}, "P1_per_model": {}}

for model_key in MODELS:
    RESULTS["models"][model_key] = {}
    RESULTS["fits"][model_key]   = {}
    for level in MATH_PROBLEMS:
        ck_key = f"{model_key}_{level}"
        if ck_key in ckpt and "sweep" in ckpt[ck_key]:
            RESULTS["models"][model_key][level] = ckpt[ck_key]["sweep"]
            RESULTS["fits"][model_key][level]   = ckpt[ck_key]["fit"]

# P1 test per model: T*(L1) < T*(L3) < T*(L5)?
for model_key in MODELS:
    fits = RESULTS["fits"].get(model_key, {})
    t_stars = {lv: fits[lv]["T_star"] for lv in ["Level_1","Level_3","Level_5"]
               if lv in fits and "T_star" in fits[lv]}
    if len(t_stars) == 3:
        ok = (t_stars["Level_1"] < t_stars["Level_3"] < t_stars["Level_5"])
        RESULTS["P1_per_model"][model_key] = {
            **t_stars, "monotonic": ok,
            "ratio": t_stars["Level_5"] / max(t_stars["Level_1"], 1),
        }

out_json = OUT / "multimodel_results.json"
with open(out_json, "w") as f:
    json.dump(RESULTS, f, indent=2, default=str)
print(f"\n[json] multimodel_results.json  [{elapsed_min():.1f} min]")

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

#  Figure A1: one panel per model, 3 curves per level 
n_models_done = sum(1 for mk in MODELS if any(
    f"{mk}_{lv}" in ckpt and "sweep" in ckpt[f"{mk}_{lv}"]
    for lv in MATH_PROBLEMS))

if n_models_done > 0:
    fig, axes = plt.subplots(1, max(n_models_done, 1),
                             figsize=(5.5 * n_models_done, 4.8),
                             squeeze=False)
    ax_idx = 0
    for model_key, model_id in MODELS.items():
        model_data = RESULTS["models"].get(model_key, {})
        model_fits = RESULTS["fits"].get(model_key, {})
        if not model_data:
            continue
        ax = axes[0][ax_idx]; ax_idx += 1
        for level in ["Level_1", "Level_3", "Level_5"]:
            if level not in model_data:
                continue
            col  = LEVEL_COLS[level]
            data = model_data[level]
            T_d  = np.array([r["budget"]       for r in data])
            A_d  = np.array([r["accuracy"]*100 for r in data])
            ax.scatter(T_d/1000, A_d, color=col, s=40, zorder=6)
            ax.plot(T_d/1000, A_d, "o-", color=col, lw=1.3, alpha=0.4)
            fit = model_fits.get(level, {})
            if "Delta" in fit:
                popt = [fit["Delta"], fit["sigma_r2"], fit["k"],
                        fit["scale"], fit["offset"]]
                Tf = np.linspace(T_d.min(), T_d.max()*1.4, 2000)
                ax.plot(Tf/1000, g_scaled(Tf, *popt), color=col, lw=2.0,
                        label=f"{LEVEL_LABELS[level]}  T*={fit['T_star']:.0f}")
                ax.axvline(fit["T_star"]/1000, color=col, lw=0.8, ls=":", alpha=0.7)
        ax.set_xlabel("Budget $T$ (k tokens)")
        ax.set_ylabel("Accuracy (%)")
        ax.set_title(MODEL_LABELS[model_key])
        ax.legend(fontsize=8, loc="upper left")
    fig.suptitle("Experiment A: Overthinking curves — multi-model validation\n"
                 "(8 problems/level, 2 runs, 8-bit quantisation)",
                 fontsize=11, y=1.02)
    plt.tight_layout()
    savefig("figure_multimodel", fig)

#  Figure A2: T* comparison table bar chart 
p1_data = RESULTS.get("P1_per_model", {})
done_models = [mk for mk in MODELS if mk in p1_data]
if done_models:
    levels_plot = ["Level_1","Level_3","Level_5"]
    x = np.arange(len(levels_plot))
    width = 0.25
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for i, mk in enumerate(done_models):
        t_vals = [p1_data[mk].get(lv, 0) for lv in levels_plot]
        bars   = ax.bar(x + i*width, t_vals, width,
                        color=MODEL_COLS[mk], label=MODEL_LABELS[mk],
                        alpha=0.82, zorder=3)
        for bar, v in zip(bars, t_vals):
            if v:
                ax.text(bar.get_x()+bar.get_width()/2, v + 30,
                        f"{v:.0f}", ha="center", fontsize=7, rotation=45)
    ax.set_xticks(x + width)
    ax.set_xticklabels([LEVEL_LABELS[lv] for lv in levels_plot])
    ax.set_ylabel("Optimal budget $T^*$ (tokens)")
    ax.set_title("Experiment A: T* by model and difficulty level\n"
                 "P1 prediction: T* increases with difficulty (for all models)")
    ax.legend(fontsize=9)
    plt.tight_layout()
    savefig("figure_multimodel_tstar", fig)

# 10. LaTeX TABLE 

def tex(v, fmt=".3f"):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    return format(v, fmt)

lines = [
    "% AUTO-GENERATED by part3_multimodel.py",
    r"\begin{table}[t]",
    r"\centering\small",
    r"\caption{Experiment A: g-model fits per model and difficulty level. "
    r"$T^*$ = fitted overthinking threshold (tokens); "
    r"Peak = fitted peak accuracy (\%); "
    r"P1 = T* monotone-increasing with difficulty.}",
    r"\label{tab:multimodel}",
    r"\begin{tabular}{llccccc}",
    r"\toprule",
    r"Model & Level & $T^*$ & 95\% CI & Peak (\%) & $R^2$ & P1 \\",
    r"\midrule",
]
for model_key in MODELS:
    model_fits = RESULTS["fits"].get(model_key, {})
    p1_info    = RESULTS["P1_per_model"].get(model_key, {})
    p1_str     = "✓" if p1_info.get("monotonic") else "✗"
    for li, level in enumerate(["Level_1","Level_3","Level_5"]):
        fit  = model_fits.get(level, {})
        ci   = fit.get("CI", {})
        t_ci = ci.get("T_star", [float("nan"), float("nan")])
        mlbl = MODEL_LABELS[model_key] if li == 0 else ""
        p1c  = p1_str if li == 2 else ""
        if "T_star" not in fit:
            lines.append(rf"{mlbl} & {level.replace('_',' ')} & \multicolumn{{4}}{{c}}{{not completed}} & {p1c} \\")
        else:
            lines.append(
                rf"{mlbl} & {level.replace('_',' ')} & "
                rf"${tex(fit.get('T_star'),'.0f')}$ & "
                rf"$[{tex(t_ci[0],'.0f')},{tex(t_ci[1],'.0f')}]$ & "
                rf"${tex(fit.get('peak_acc'),'.1f')}$ & "
                rf"${tex(fit.get('R2'))}$ & {p1c} \\\\"
            )
    lines.append(r"\midrule")
lines[-1] = r"\bottomrule"
lines += [r"\end{tabular}", r"\end{table}"]
tex_path = OUT / "table_multimodel.tex"
tex_path.write_text("\n".join(lines), encoding="utf-8")
print(f"  [tex] table_multimodel.tex")

# 11. SUMMARY

print("\n" + "="*66)
print("  EXPERIMENT A SUMMARY")
print("="*66)
for model_key in MODELS:
    p1 = RESULTS["P1_per_model"].get(model_key, {})
    if p1:
        print(f"  {MODEL_LABELS[model_key]}")
        for lv in ["Level_1","Level_3","Level_5"]:
            print(f"    {lv}: T*={p1.get(lv,0):.0f}")
        print(f"    P1 supported: {p1.get('monotonic','?')}  "
              f"ratio={p1.get('ratio',0):.1f}x")
    else:
        print(f"  {MODEL_LABELS[model_key]}: incomplete")
print(f"\n  Total elapsed: {elapsed_min():.1f} min")
print(f"  Outputs: {OUT}")
for fp in sorted(OUT.iterdir()):
    print(f"    {fp.name:48s}  {fp.stat().st_size/1024:7.1f} KB")
