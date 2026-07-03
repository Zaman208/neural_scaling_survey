#!/usr/bin/env python3
"""
EXPERIMENT E: Prospective Prediction 
Protocol
  Train budgets  : [150, 300, 600, 1200]
  Test  budgets  : [2400, 4800, 9600]          ← never used during fitting
  Model          : Phi-3.5-mini-instruct (8-bit) — primary
                   Qwen2.5-3B-Instruct  (8-bit) — replication
  Levels         : Level_1, Level_3, Level_5
  Problems       : 10 per level
  Samples        : 3 per (problem, budget)

Evaluation
  For each (model, level):
    1. Run ALL 7 budgets (train+test) to get ground-truth accuracy
    2. Fit g-model on train budgets only
    3. Predict accuracy at test budgets using fitted model
    4. Compute prediction error: MAE, RMSE, R² on test set
  Strong result: R²_test > 0.8 at test budgets (model predicts unseen data)

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

OUT = Path("/kaggle/working/results_part7")
OUT.mkdir(parents=True, exist_ok=True)

#  1. CONSTANTS 

TRAIN_BUDGETS  = [150, 300, 600, 1200]
TEST_BUDGETS   = [2400, 4800, 9600]
ALL_BUDGETS    = TRAIN_BUDGETS + TEST_BUDGETS

N_SAMPLES = 3
N_PROBS   = 10
SESSION_LIMIT_MIN = 510

MODELS = {
    "phi35":  ("microsoft/Phi-3.5-mini-instruct", True),
    "qwen3b": ("Qwen/Qwen2.5-3B-Instruct",        True),
}
MODEL_LABELS = {"phi35": "Phi-3.5-mini (3.8B)", "qwen3b": "Qwen2.5-3B (3.0B)"}
MODEL_COLS   = {"phi35": "#1B7837",              "qwen3b": "#762A83"}
LEVEL_COLS   = {"Level_1":"#2166AC","Level_3":"#F4A582","Level_5":"#D6604D"}
LEVEL_LABELS = {"Level_1":"Level 1 (easy)","Level_3":"Level 3 (medium)",
                "Level_5":"Level 5 (hard)"}

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

#  4. g-MODEL FIT (train-only) 

def g_raw(T, Delta, sigma_r2, k):
    V = np.maximum(k * np.asarray(T, dtype=float), 0.0)
    s = np.maximum(sigma_r2 + V, 1e-12)
    return (1.0 / np.sqrt(2.0 * np.pi * s)) * np.exp(-Delta**2 / (2.0 * s))

def g_scaled(T, Delta, sigma_r2, k, scale, offset):
    return scale * g_raw(T, Delta, sigma_r2, k) + offset

def fit_g_train(T_train, A_train, n_boot=300):
    """
    Fit g-model on training budgets only.
    Returns params, T*, and prediction function.
    """
    T = np.asarray(T_train, dtype=float)
    A = np.asarray(A_train, dtype=float)
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
        return None

    T_fine = np.linspace(T.min(), max(T.max(), max(TEST_BUDGETS)) * 1.1, 10000)
    A_fine = g_scaled(T_fine, *best_p)
    T_star = float(T_fine[np.argmax(A_fine)])

    # bootstrap CI on params using training data only
    rng  = np.random.default_rng(42)
    bts  = []
    n    = len(T)
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

    # Predict at TEST budgets using each bootstrap sample → prediction CI
    T_test = np.array(TEST_BUDGETS, dtype=float)
    pred_boot = []
    for pb in bts:
        pred_boot.append(g_scaled(T_test, *pb).tolist())

    pred_ci = {}
    if pred_boot:
        pb_arr = np.array(pred_boot)
        for i, b in enumerate(TEST_BUDGETS):
            pred_ci[b] = [float(np.percentile(pb_arr[:, i], 2.5)),
                          float(np.percentile(pb_arr[:, i], 97.5))]

    return {
        "params":     best_p.tolist(),
        "R2_train":   float(best_r2),
        "T_star":     T_star,
        "pred_ci":    pred_ci,
        "n_boot_ok":  len(bts),
    }

def predict_at(params, budgets):
    """Apply fitted params to arbitrary budgets."""
    return g_scaled(np.array(budgets, dtype=float), *params).tolist()

def eval_prediction(pred, actual):
    """MAE, RMSE, R² for test-set prediction."""
    p = np.array(pred)
    a = np.array(actual)
    mae  = float(np.mean(np.abs(p - a)))
    rmse = float(np.sqrt(np.mean((p - a)**2)))
    ss_r = float(np.sum((p - a)**2))
    ss_t = float(np.sum((a - a.mean())**2))
    r2   = float(1.0 - ss_r / ss_t) if ss_t > 0 else float("nan")
    return {"MAE": mae, "RMSE": rmse, "R2_test": r2}

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

def build_prompt(tok, problem, has_system):
    if has_system:
        messages = [{"role":"system","content":SYS_PROMPT},
                    {"role":"user",  "content":problem}]
    else:
        messages = [{"role":"user","content":f"{SYS_PROMPT}\n\n{problem}"}]
    try:
        text = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        text = f"Instruct: {SYS_PROMPT}\n{problem}\nOutput:"
    return tok(text, return_tensors="pt").to(DEVICE)

def run_one_budget(model, tok, problems, budget, has_sys, n_samples=N_SAMPLES):
    all_correct = []
    for problem, expected in problems:
        inputs = build_prompt(tok, problem, has_sys)
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
            seq = out.sequences if hasattr(out, "sequences") else out
            gen_ids  = seq[0][in_len:]
            gen_text = tok.decode(gen_ids, skip_special_tokens=True)
            run_correct.append(int(answer_is_correct(gen_text, expected)))
        all_correct.append(float(np.mean(run_correct)))
    return float(np.mean(all_correct))

#  6. MAIN EXPERIMENT LOOP 

ckpt = load_checkpoint()
print(f"\n[Checkpoint] {len(ckpt)} (model,level,budget) entries already done.")

for model_key, (model_id, has_sys) in MODELS.items():
    if not time_ok():
        print(f"\n[TIME] approaching session limit — stopping.")
        break

    # Find which (level, budget) pairs are still needed
    todo = []
    for lv in MATH_PROBLEMS:
        for b in ALL_BUDGETS:
            if f"{model_key}_{lv}_{b}" not in ckpt:
                todo.append((lv, b))

    if not todo:
        print(f"\n[SKIP] {model_key} — all done.")
        continue

    print(f"\n{'='*66}")
    print(f"  Loading {MODEL_LABELS[model_key]}  [{elapsed_min():.1f} min]")
    print(f"  Remaining: {len(todo)} (level,budget) pairs")
    print(f"{'='*66}")

    try:
        model, tok = load_model_8bit(model_id)
        print(f"  Model loaded  [{elapsed_min():.1f} min]")
    except Exception as e:
        print(f"  ERROR: {e}")
        ckpt[f"{model_key}_load_error"] = str(e)
        save_checkpoint(ckpt)
        continue

    for level, budget in todo:
        ck_key = f"{model_key}_{level}_{budget}"
        if ck_key in ckpt:
            continue
        if not time_ok():
            print(f"  [TIME] stopping before {ck_key}")
            break

        tag = "TRAIN" if budget in TRAIN_BUDGETS else "TEST "
        print(f"  [{model_key}][{level}][{tag} T={budget}]  "
              f"{N_PROBS} probs × {N_SAMPLES} runs")

        problems = MATH_PROBLEMS[level][:N_PROBS]
        acc = run_one_budget(model, tok, problems, budget, has_sys)

        ckpt[ck_key] = {
            "accuracy": acc, "budget": budget,
            "split": "train" if budget in TRAIN_BUDGETS else "test",
            "elapsed_min": elapsed_min(),
        }
        save_checkpoint(ckpt)
        print(f"    acc={acc:.3f}  [{elapsed_min():.1f} min]")

    release(model)
    print(f"  [released] {model_key}  [{elapsed_min():.1f} min]")

#  7. ASSEMBLE & PREDICT 

RESULTS = {}

for model_key in MODELS:
    RESULTS[model_key] = {}
    for level in MATH_PROBLEMS:
        # Collect measured accuracies
        measured = {}
        for b in ALL_BUDGETS:
            ck_key = f"{model_key}_{level}_{b}"
            if ck_key in ckpt and "accuracy" in ckpt[ck_key]:
                measured[b] = ckpt[ck_key]["accuracy"] * 100  # as %

        if len(measured) < len(TRAIN_BUDGETS):
            print(f"  [{model_key}][{level}] insufficient train data — skipping fit")
            RESULTS[model_key][level] = {"measured": measured, "error": "insufficient_train"}
            continue

        # Fit on train budgets only
        T_train = np.array([b for b in TRAIN_BUDGETS if b in measured])
        A_train = np.array([measured[b] for b in TRAIN_BUDGETS if b in measured])

        fit_res = fit_g_train(T_train, A_train)
        if fit_res is None:
            print(f"  [{model_key}][{level}] fit failed")
            RESULTS[model_key][level] = {"measured": measured, "error": "fit_failed"}
            continue

        # Predict at test budgets using train-fitted params
        T_test_available = [b for b in TEST_BUDGETS if b in measured]
        predicted_test   = predict_at(fit_res["params"], T_test_available)
        actual_test      = [measured[b] for b in T_test_available]

        eval_res = eval_prediction(predicted_test, actual_test) \
                   if len(actual_test) >= 2 else {}

        RESULTS[model_key][level] = {
            "measured":     {str(k): v for k, v in measured.items()},
            "fit":          fit_res,
            "T_test_avail": T_test_available,
            "predicted":    {b: p for b, p in zip(T_test_available, predicted_test)},
            "actual_test":  {b: a for b, a in zip(T_test_available, actual_test)},
            "eval":         eval_res,
        }

        r2t = eval_res.get("R2_test", float("nan"))
        mae = eval_res.get("MAE", float("nan"))
        print(f"  [{MODEL_LABELS[model_key]}][{level}]  "
              f"R²_train={fit_res['R2_train']:.3f}  "
              f"R²_test={r2t:.3f}  MAE={mae:.1f}%  "
              f"T*={fit_res['T_star']:.0f}")

out_json = OUT / "prospective_results.json"
with open(out_json, "w") as f:
    json.dump(RESULTS, f, indent=2, default=str)
print(f"\n[json] prospective_results.json  [{elapsed_min():.1f} min]")

#  8. FIGURES 

plt.rcParams.update({
    "font.family":"serif","font.size":11,"axes.titlesize":11,
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

models_done = [mk for mk in MODELS
               if any(RESULTS.get(mk,{}).get(lv,{}).get("fit") for lv in MATH_PROBLEMS)]

# ── Figure E1: train fit + test prediction (one row per model, one col per level) ──
if models_done:
    n_rows = len(models_done)
    n_cols = 3
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.5*n_cols, 4.8*n_rows),
                             squeeze=False)
    for ri, mk in enumerate(models_done):
        for ci_ax, level in enumerate(["Level_1","Level_3","Level_5"]):
            ax  = axes[ri][ci_ax]
            res = RESULTS.get(mk, {}).get(level, {})
            if not res or "fit" not in res or not res["fit"]:
                ax.set_visible(False); continue

            measured  = {int(k): v for k, v in res["measured"].items()}
            fit_res   = res["fit"]
            predicted = {int(k): v for k, v in res.get("predicted", {}).items()}
            actual_t  = {int(k): v for k, v in res.get("actual_test", {}).items()}

            col = MODEL_COLS[mk]
            # All measured data
            T_all = np.array(sorted(measured.keys()), dtype=float)
            A_all = np.array([measured[int(b)] for b in T_all])

            # Train points
            T_tr = np.array([b for b in T_all if int(b) in TRAIN_BUDGETS])
            A_tr = np.array([measured[int(b)] for b in T_tr])

            # Test actual points
            T_te = np.array(sorted(actual_t.keys()), dtype=float)
            A_te = np.array([actual_t[int(b)] for b in T_te])

            # Fitted curve (extrapolated to test range)
            T_dense = np.linspace(min(T_all)*0.8, max(T_all)*1.05, 3000)
            A_curve = g_scaled(T_dense, *fit_res["params"])

            ax.scatter(T_tr/1000, A_tr, color=col, s=60, zorder=7,
                       marker="o", label="Train data")
            ax.scatter(T_te/1000, A_te, color="black", s=80, zorder=8,
                       marker="*", label="Test actual")
            ax.plot(T_dense/1000, A_curve, color=col, lw=2.0, zorder=5,
                    label=f"g-fit (train only)\nT*={fit_res['T_star']:.0f}")

            # Predicted test points
            for b, p_val in predicted.items():
                ax.scatter(b/1000, p_val, color="red", s=80, marker="^",
                           zorder=9)
                if int(b) in actual_t:
                    ax.plot([b/1000, b/1000], [p_val, actual_t[int(b)]],
                            "r-", lw=1.2, alpha=0.7)

            # Prediction CI bands
            ci = fit_res.get("pred_ci", {})
            for b_str, ci_vals in ci.items():
                b_int = int(b_str)
                if ci_vals[0] is not None:
                    ax.axvspan(b_int/1000 - 0.05, b_int/1000 + 0.05,
                               ymin=(ci_vals[0] - ax.get_ylim()[0]) /
                                    max(ax.get_ylim()[1] - ax.get_ylim()[0], 1),
                               ymax=(ci_vals[1] - ax.get_ylim()[0]) /
                                    max(ax.get_ylim()[1] - ax.get_ylim()[0], 1),
                               alpha=0.12, color="red")

            # Vertical split line
            ax.axvline(max(TRAIN_BUDGETS)/1000, color="gray", lw=1.2,
                       ls="--", alpha=0.6, label="train|test split")

            eval_r = res.get("eval", {})
            r2t    = eval_r.get("R2_test", float("nan"))
            mae    = eval_r.get("MAE",     float("nan"))
            ax.set_xlabel("Budget $T$ (k tokens)")
            ax.set_ylabel("Accuracy (%)")
            ax.set_title(
                f"{MODEL_LABELS[mk]}\n{LEVEL_LABELS[level]}\n"
                f"R²_test={r2t:.2f}  MAE={mae:.1f}%"
            )
            ax.legend(fontsize=7, loc="upper right")

    fig.suptitle(
        "Experiment E: Prospective prediction\n"
        "Model fitted on T∈{150,300,600,1200} — predicts T∈{2400,4800,9600}\n"
        "● train  ★ test-actual  ▲ test-predicted   red line = error",
        fontsize=11, y=1.03,
    )
    plt.tight_layout()
    savefig("figure_prospective_fit", fig)

#  Figure E2: prediction error summary 
error_data = []
for mk in models_done:
    for level in MATH_PROBLEMS:
        res  = RESULTS.get(mk, {}).get(level, {})
        eval_r = res.get("eval", {})
        if "R2_test" in eval_r:
            error_data.append({
                "model": mk, "level": level,
                "R2_test": eval_r["R2_test"],
                "MAE":     eval_r["MAE"],
                "RMSE":    eval_r["RMSE"],
            })

if error_data:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    levels    = ["Level_1","Level_3","Level_5"]
    x         = np.arange(len(levels))
    w         = 0.35
    for ax_idx, (metric, label, thresh) in enumerate([
        ("R2_test", "$R^2$ on test budgets",     0.8),
        ("MAE",     "MAE (pp) on test budgets",  None),
    ]):
        ax = axes[ax_idx]
        for i, mk in enumerate(models_done):
            vals = []
            for lv in levels:
                d = next((e for e in error_data if e["model"]==mk and e["level"]==lv), {})
                vals.append(d.get(metric, float("nan")))
            xs = x + (i - (len(models_done)-1)/2) * w
            ax.bar(xs, [v if not math.isnan(v) else 0 for v in vals],
                   w*0.9, color=MODEL_COLS[mk], alpha=0.82,
                   label=MODEL_LABELS[mk], zorder=3)
        if thresh is not None:
            ax.axhline(thresh, color="red", lw=1.5, ls="--",
                       alpha=0.7, label=f"threshold={thresh}")
        ax.set_xticks(x)
        ax.set_xticklabels([LEVEL_LABELS[lv] for lv in levels], fontsize=9)
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.legend(fontsize=8)
    plt.suptitle("Experiment E: Prospective prediction quality\n"
                 "R² > 0.8 → model captures real phenomenon (not just overfitting)",
                 fontsize=11, y=1.02)
    plt.tight_layout()
    savefig("figure_prospective_error", fig)

#  9. LaTeX TABLE 

def tex(v, fmt=".3f"):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    return format(v, fmt)

lines = [
    "% AUTO-GENERATED by part7_prospective.py",
    r"\begin{table}[t]",
    r"\centering\small",
    r"\caption{Experiment E: prospective prediction quality. "
    r"g-model fitted on $T\in\{150,300,600,1200\}$ tokens only, "
    r"then evaluated at unseen $T\in\{2400,4800,9600\}$. "
    r"$R^2_{\mathrm{test}}>0.8$ indicates the theory generalises beyond "
    r"the training window.}",
    r"\label{tab:prospective}",
    r"\begin{tabular}{llcccc}",
    r"\toprule",
    r"Model & Level & $R^2_{\mathrm{train}}$ & $R^2_{\mathrm{test}}$ "
    r"& MAE (pp) & $T^*$ \\",
    r"\midrule",
]
for mk in MODELS:
    for li, level in enumerate(["Level_1","Level_3","Level_5"]):
        res    = RESULTS.get(mk, {}).get(level, {})
        fit_r  = res.get("fit", {}) or {}
        eval_r = res.get("eval", {}) or {}
        mlbl   = MODEL_LABELS[mk] if li == 0 else ""
        if not fit_r:
            lines.append(rf"{mlbl} & {level.replace('_',' ')} & \multicolumn{{4}}{{c}}{{incomplete}} \\")
        else:
            lines.append(
                rf"{mlbl} & {level.replace('_',' ')} & "
                rf"${tex(fit_r.get('R2_train'))}$ & "
                rf"${tex(eval_r.get('R2_test'))}$ & "
                rf"${tex(eval_r.get('MAE'),'.1f')}$ & "
                rf"${tex(fit_r.get('T_star'),'.0f')}$ \\"
            )
    lines.append(r"\midrule")
lines[-1] = r"\bottomrule"
lines += [r"\end{tabular}", r"\end{table}"]
(OUT / "table_prospective.tex").write_text("\n".join(lines), encoding="utf-8")
print("  [tex] table_prospective.tex")

#  10. SUMMARY 

print("\n" + "="*66)
print("  EXPERIMENT E (PROSPECTIVE PREDICTION) SUMMARY")
print("="*66)
for mk in MODELS:
    for level in MATH_PROBLEMS:
        res    = RESULTS.get(mk, {}).get(level, {})
        fit_r  = res.get("fit", {}) or {}
        eval_r = res.get("eval", {}) or {}
        if fit_r:
            r2tr = fit_r.get("R2_train", float("nan"))
            r2te = eval_r.get("R2_test",  float("nan"))
            mae  = eval_r.get("MAE",      float("nan"))
            tstar= fit_r.get("T_star",    float("nan"))
            strong = "STRONG" if r2te > 0.8 else ("MODERATE" if r2te > 0.5 else "WEAK")
            print(f"  [{MODEL_LABELS[mk]}][{level}]  "
                  f"R²_tr={r2tr:.3f}  R²_te={r2te:.3f}  "
                  f"MAE={mae:.1f}pp  T*={tstar:.0f}  → {strong}")
print(f"\n  Total elapsed: {elapsed_min():.1f} min")
print(f"  Outputs: {OUT}")
for fp in sorted(OUT.iterdir()):
    print(f"    {fp.name:48s}  {fp.stat().st_size/1024:7.1f} KB")
