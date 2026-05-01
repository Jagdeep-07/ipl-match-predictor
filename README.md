# IPL Match Winner Prediction with Hyperparameter Optimization

Predicts whether Team 1 wins an IPL cricket match using only pre-match and toss-time information. The core focus is comparing **Random Search** vs **Bayesian Optimization (Optuna)** for tuning a TensorFlow MLP classifier.

Built as a machine learning optimization project at Fanshawe College, Winter 2026.

---

## Problem

Binary classification: did Team 1 win? Features are restricted to information available before the first ball — team identities, toss result, venue, and rolling historical win rates computed from earlier matches only. No in-match or post-match data leaks in.

---

## Results

| Model | Accuracy | F1 | ROC-AUC |
|---|---|---|---|
| Dummy baseline (most frequent) | 51.83% | 0.683 | 0.500 |
| Simple MLP baseline (no tuning) | 52.44% | 0.361 | **0.566** |
| Best Random Search MLP | **54.27%** | 0.540 | 0.513 |
| Best Bayesian Optuna MLP | 51.83% | **0.578** | 0.528 |

Test set is the newest 15% of matches by date (chronological split, not random).

**Best hyperparameters found:**

| | Random Search | Optuna (TPE) |
|---|---|---|
| Learning rate | 0.000456 | 0.000197 |
| Batch size | 128 | 128 |
| Dropout | 0.5 | 0.2 |
| Hidden units | 64 | 256 |
| Optimizer | RMSProp | SGD |
| Runtime (50 trials) | 173.2 s | 188.1 s |
| Best val ROC-AUC | 0.579 | 0.569 |

---

## Key findings

- Hyperparameter tuning did not consistently beat the untuned baseline across all metrics. Random Search gained ~1.8 pp accuracy; Optuna gained ~22 pp F1 but not accuracy. Neither exceeded the baseline's ROC-AUC.
- The ceiling is likely genuine: pre-match tabular features carry limited signal for IPL outcomes. Injuries, pitch conditions, and day-of form are unobserved.
- Adam was the most stable optimizer in ablation (56.44% val accuracy). SGD had higher recall but noisy convergence.
- Dropout-only regularization gave the best accuracy/F1 balance. Adding weight decay and a LR schedule pushed recall at the cost of precision.
- SHAP analysis identified `team1_prior_win_rate` and `team2_prior_win_rate` as the dominant global predictors. Venue and toss decision contributed less than expected.

---

## Dataset

[IPL Dataset on Kaggle](https://www.kaggle.com/datasets/patrickb1912/ipl-complete-dataset-20082020) — matches from 2008–2020.

- `matches.csv` — 956 rows, one per match
- `deliveries.csv` — 179 078 rows, ball-by-ball (used for context, not modeling)

---

## Setup

```bash
pip install tensorflow optuna shap scikit-learn pandas numpy matplotlib seaborn
```

Open `ipl_optimization_project.ipynb` and run all cells in order. The final cell exports all figures and tables to `ipl_project_outputs/`.

---

## Outputs

All generated artifacts are in `ipl_project_outputs/`:

- 4 confusion matrices (one per model)
- Loss curves, convergence plots, ROC and precision-recall curves
- Ablation comparison charts (optimizer, regularization)
- SHAP global summary and local waterfall plots
- CSV tables for every metric reported above

---

## Project structure

```
.
├── ipl_optimization_project.ipynb   # full pipeline
├── ipl_dataset/
│   ├── matches.csv
│   └── deliveries.csv
├── ipl_project_outputs/             # generated figures and tables
│   ├── *.png  (17 charts)
│   └── *.csv  (9 result tables)
├── IPL_Optimization_Final_Report.pdf
└── ipl_presentation.pdf
```
