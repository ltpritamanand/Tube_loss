# TubeNet: Quantifying Uncertainty in LLM-as-a-Judge

This folder contains the source code and experimental setup for constructing calibrated prediction intervals for **LLM-as-a-Judge** scores using the **Tube Loss** function, as presented in:

> *Chovatiya, P., Patel, N., & Anand, P. "Quantifying LLM-as-a-Judge Uncertainty via Prediction Intervals: TubeNet, Tube Loss, and Conformal Prediction."*

## Overview

LLM-as-a-Judge is the dominant paradigm for scalable evaluation of natural language generation, but a single score (e.g., "4") hides the uncertainty of a stochastic judge. This project converts point-estimate LLM scores into prediction intervals with a target **≥90% marginal coverage** guarantee.

The pipeline has three stages:

1. **Feature extraction** — SentenceTransformer embeddings of the article and summary, augmented with LLM judge uncertainty statistics (`mean`, `std`, `q05`, `q95`) elicited through repeated black-box sampling.
2. **Interval learning** — **TubeNet**, a dual-head MLP that outputs an interval midpoint and a Softplus half-width, guaranteeing lower < upper by construction. Trained with the Tube Loss, which jointly penalises miscoverage and excessive width.
3. **Conformal calibration** — Split conformal prediction on a held-out calibration set expands the bounds to provide distribution-free marginal coverage.

This is the first application of the Tube Loss function to LLM evaluation, extending it beyond probabilistic forecasting.

## Experimental Conditions

| Condition | Features | Dim | LLM Calls |
|---|---|---|---|
| A | BERT(article) ⊕ BERT(summary) | 768 | 0 |
| C1 | BERT ⊕ 10 raw repeated scores | 778 | 10 |
| C2 | BERT ⊕ [mean, std, q05, q95] | 772 | 5 |

Condition **C2** is the best configuration: adding only four distributional statistics reduces interval width by up to **87%** (consistency) and **51%** (coherence) relative to the BERT-only baseline.

## Key Results

* Best configuration (C2 + Boosted CQR): ≥90% coverage on all four SummEval dimensions.
* Outperforms the prior state of the art (Sheng et al., EMNLP 2025) in **27/28** method–dimension comparisons.
* Diversity beats size: Gemma2-9b produces narrower intervals than GPT-OSS-120b because more varied scores give a richer uncertainty signal.
* TubeNet's interval midpoint beats the raw LLM score on MAE against human ground truth across all 9 experiments and 4 dimensions.
* Five judges evaluated: Llama-3.3-70b, Mixtral-8x7b, Qwen3-32b, Gemma2-9b-it, and GPT-OSS-120b (all via the Groq API).

## File Descriptions

### `experiments/`

* **`main_v2_a.py`**: Condition A — BERT-only baseline (no LLM calls).
* **`main_v2_c1.py`**: Condition C1 — BERT + raw repeated LLM scores.
* **`main_v2_c2.py`**: Condition C2 — BERT + distributional statistics (recommended).
* **`main_v2_c2_cqr.py` / `_asym_cqr.py` / `_chr.py` / `_lvd.py` / `_boosted_lcp.py` / `_r2ccp.py`**: C2 with the respective conformal wrapper.
* **`main_v2_c2_boosted_cqr.py`**: C2 + Boosted CQR — best overall configuration.
* **`main2.py`**: shared utilities / early prototype.

### Other folders

* **`data/`**: paired SummEval annotations with CNN/DailyMail source articles.
* **`data_processing/`**: script for re-pairing summaries with source articles (`pair_data.py`).
* **`evaluation/`**: SummEval evaluation toolkit (Fabbri et al., 2021).
* **`bmp_results_A/`**: figures and JSON results for Condition A.
* **`Report.pdf`**, **`Presentation.pdf`**: project report and presentation slides.

## How to Run

1. **Dependencies**:

```bash
pip install torch sentence-transformers groq scikit-learn numpy matplotlib tqdm
```

2. **API key**: set `GROQ_API_KEY` in your environment.

3. **Data**: the C2-family scripts expect a file named `clean_single_annotation.jsonl` in this folder. The paired SummEval file is bundled under `data/`.

4. **Run experiments** (from this folder):

```bash
python experiments/main_v2_a.py            # Condition A baseline
python experiments/main_v2_c2.py           # Condition C2 (recommended)
python experiments/main_v2_c2_boosted_cqr.py   # best overall
```

## Citation

If you use this codebase, please cite the accepted paper:

> Chovatiya, P., Patel, N., & Anand, P. "Quantifying LLM-as-a-Judge Uncertainty via Prediction Intervals: TubeNet, Tube Loss, and Conformal Prediction."

Please also cite the Tube Loss paper:

> Anand, P., Bandyopadhyay, T., & Chandra, S. "Tube Loss: A Novel Approach for Prediction Interval Estimation and Probabilistic Forecasting." Transactions on Machine Learning and Research, 2026. (arXiv:2412.06853)
