# Neural Scaling Laws: A Survey, Conceptual Synthesis, and Multi-Axis Framework

## Overview

This paper surveys the neural scaling law literature (2017-2026) across **eight resource axes** - parameters, training data, compute, data quality/repetition, architecture (MoE), numerical precision, multimodal/domain, RL post-training, and inference-time compute - and organises them into a single conceptual framework.

The core contribution is a **master loss equation** that nests Chinchilla, SoftQ, and precision-aware laws as special cases and derives a *non-monotone* accuracy-vs-budget curve for inference-time compute from first principles (Gaussian mechanism + random-walk variance model).

Key results:

- **Three formal Synthesis Observations** with closed-form proofs (non-monotone expected reward, Bernoulli robustness, random-walk variance growth)
- **Six testable predictions** (P1-P6) about overthinking behaviour across difficulty and model size
- **Quantitative fits** to published accuracy-vs-budget data (R² ≥ 0.99, bootstrap 95% CIs)
- **Twelve publication-quality figures** generated from `code/generate_figures.py`
- **Five illustrative Kaggle experiments** (Phi-3.5-mini, Qwen2.5-1.5B/3B on MATH-500)

---

## Repository Layout

```
FINAL_SCALING_REVIEW_PAPER/
├── paper/
│   ├─- neural_scaling_survey_tmlr.tex   # Main TMLR-format LaTeX paper
│   └── references.bib                   # ~45 verified BibTeX entries
├── code/
│   └── generate_figures.py              # Reproduces all 12 figures
├── figures/
│   ├── fig1_axes_overview.png           # Eight scaling axes schematic
│   ├── fig2_kaplan_vs_chinchilla.png    # Compute-optimal allocation comparison
│   ├── fig3_nonmonotone_inference.png   # Non-monotone inference curves + g(V)
│   ├── fig4_quantitative_fits.png       # Gaussian-mechanism fits to Zhou et al. 2025
│   ├── fig5_tstar_difficulty.png        # T*(d) vs Δ; peak accuracy vs |Δ| (P4)
│   ├── fig6_chinchilla_surface.png      # L(N,D) heat-map + optimal frontier
│   ├── fig7_exponent_comparison.png     # Exponent bar charts across studies
│   ├── fig8_precision_scaling.png       # Precision law and compute-optimal bits
│   ├── fig9_master_equation.png         # Master equation component panels
│   ├── fig10_p6_model_size.png          # P6: larger model → earlier T*
│   ├── fig11_zipf_coverage.png          # Zipfian coverage → β = (ζ−1)/ζ
│   └── fig12_aic_bic.png               # AIC/BIC model comparison
├── results/                             # Experiment outputs (JSON checkpoints)
├── data/                                # Benchmark data references (MATH-500 etc.)
└── README.md                            # This file
```

---

## Reproducing the Figures

**Requirements:** Python ≥ 3.9, numpy, scipy, matplotlib.

```bash
pip install numpy scipy matplotlib
python code/generate_figures.py
```

All 12 figures are written to `figures/`. The script has no external data dependencies — all curves use closed-form equations or the digitised data in Table 2 of the paper.

---


## Core Framework

The **master loss equation** (Section 6):

```
L(N, D, T; d) = [E + A·N^{−α} + B·D^{−β}]   ← base (Chinchilla)
              + ΔP(P)                           ← precision correction
              − R(T; d) + O(T; d)               ← inference-time gain/penalty
```

where:
- `R(T; d) = ρ(d)·(1 − exp(−T/τ(d)))` — saturating reasoning gain
- `O(T; d) = ω(d)·max(0, T − T*(d))²` — quadratic overthinking penalty
- `T*(d) = (Δ(d)² − σ_r²) / k` — closed-form overthinking threshold (Eq. 12)

The non-monotonicity of `g(V)` (Synthesis Observation 1) is derived analytically and verified numerically to 5×10⁻⁵ absolute error across six tested Δ values.

---

## Quantitative Fits

Fits to accuracy-vs-budget data from Zhou et al. (2025) — *Overthinking in Reasoning LLMs*:

| Series | Δ̂ | σ̂_r² | k̂ ×10⁻⁵ | R² | T*_fit |
|---|---|---|---|---|---|
| AIME, R1-32B | 0.436 | 0.047 | 1.08 | 0.998 | 13,325 |
| AIME, s1-32B | 0.512 | 0.064 | 1.80 | 0.996 | 10,994 |
| GPQA-D, R1-32B | 0.440 | 0.051 | 1.36 | 0.990 | 10,555 |

Bootstrap 95% CIs computed with B=1,000 resamples (see Table 4 of paper).

---

## Testable Predictions

| ID | Prediction | Status |
|---|---|---|
| P1 | T*(d) increases monotonically with difficulty | Mixed (literature: ✓; small-model experiment: partial) |
| P3 | Precision × reasoning interaction: lower P → earlier T* | Not supported at small scale |
| P4 | Peak accuracy ∝ 1/\|Δ\| | Not confirmed at small scale |
| P5 | Variance drift: V(T) = kT (Hurst H≈0.5) | Not supported (H≈0 in experiments) |
| P6 | Larger models reach T* earlier | Partial (1.5B vs 3B at L1: ✓) |
| P2 | Compute reallocation toward T as d grows | Conjecture (no direct test) |

Scope note: experiments ran on 1.5–3.8B models on Kaggle T4×2. Scope-boundary findings should not be read as falsifications at frontier scale.

---

## Five Illustrative Experiments (Kaggle T4×2)

All experiments in `code/` (part3–part7) use:
- **Models:** Phi-3.5-mini-instruct (3.8B), Qwen2.5-1.5B/3B-Instruct, Gemma-2-2b-it
- **Benchmark:** MATH-500, difficulty levels L1/L3/L5
- **Budgets:** T ∈ {150, 300, 600, 1200, 2400, 4800, 9600} tokens
- **Fitting:** scipy.optimize.curve_fit, 600-resample bootstrap 95% CI

Full experiment descriptions in Appendix C of the paper.

---

## Citation

```bibtex
@article{[author]2026scaling,
  title   = {Neural Scaling Laws: A Survey, Conceptual Synthesis,
             and Multi-Axis Framework},
  author  = {[Author Name]},
  journal = {Transactions on Machine Learning Research},
  year    = {2026},
  url     = {https://github.com/zaman208/neural-scaling-survey}
}
```

---

## Key References

- Kaplan et al. (2020). *Scaling Laws for Neural Language Models.* arXiv:2001.08361
- Hoffmann et al. (2022). *Training Compute-Optimal LLMs (Chinchilla).* arXiv:2203.15556
- Kumar et al. (2024). *Scaling Laws for Precision.* arXiv:2411.04330
- Snell et al. (2024). *Scaling LLM Test-Time Compute.* arXiv:2408.03314
- Zhou et al. (2025). *Overthinking in Reasoning LLMs.* arXiv:2502.01999
- Ghosal et al. (2025). *The Mirage of Thinking.* arXiv:2506.09250

See `paper/references.bib` for the full bibliography (~45 entries).

---

## License

MIT License — see `LICENSE` for details.
