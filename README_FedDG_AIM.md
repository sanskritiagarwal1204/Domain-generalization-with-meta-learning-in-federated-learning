# FedMedICL — Domain Generalization with Meta‑Learning in Federated Learning (Tabular Health)


This repository implements **FedMedICL** for tabular health data using a **TabTransformer** backbone, **dual BatchNorm** (local/global), and a compact **BN Adapter** that mixes instance and global statistics to improve cross‑site generalization.

---

## Repository Structure

```
FedMedICL_COVID/
├── configs/
│   ├── __init__.py
│   └── covid_tabular.yaml
├── dataset/
│   ├── CovidSurveillanceDataset.py
│   └── __init__.py
├── model/
│   ├── BNAdapter.py
│   ├── TabTransformer.py
│   └── __init__.py
├── reporting/
│   ├── __init__.py
│   ├── metrics.py
│   └── plots.py
├── scripts/
│   ├── __init__.py
│   └── train_covid_federated.py   # legacy, references non‑existent modules
├── training/
│   ├── __init__.py
│   ├── algorithms.py
│   ├── client_update.py
│   └── federated_trainer.py
├── utils/
│   ├── logger.py
│   └── saver.py
├── wandb/                          # local/offline run artifacts (safe to delete)
├── __init__.py
├── evaluation.py
├── requirements.txt                # heavy, hardware‑specific (see below)
└── train_fedmedicl.py
```

Top level:
```
FEDMEDICL_IMPLEMENTATION.ipynb
NOVELTY_FEDMEDICL.ipynb
Report.pdf
```

---

## Installation

```bash
# Python 3.10 or 3.11 recommended
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Minimal dependencies (CPU-friendly). Adjust torch/torchvision for your platform if using GPU.
pip install "torch>=2.1,<2.6" "torchvision>=0.16,<0.21" \
            pandas numpy scikit-learn matplotlib tqdm PyYAML wandb
```

---

## Dataset

Default target: **COVID‑19 Case Surveillance Public Use Data (CSV)**.

- **Categorical features (expected):** `age_group`, `sex`, `Race and ethnicity (combined)`, `medcond_yn`.
- **Label column:** `death_yn` with mapping `"No" → 0`, `"Yes" → 1`.
- **Client splitting:**
  - If the CSV contains `HospitalID`, rows are split by that column.
  - If not, a `simulated_client_id` is generated and written to `./data/simulated_clients.csv` for reproducibility.

---

## Quick Start

### 1) Train

```bash
python FedMedICL_COVID/train_fedmedicl.py \
  --dataset /path/to/COVID-19_Case_Surveillance_Public_Use_Data.csv \
  --algorithm FedDG \
  --num_clients 5 \
  --rounds 50 \
  --adapter_rounds 5 \
  --emb_dim 64 --hidden_dim 512 --n_layers 4 --transformer_heads 8 --dropout 0.1 \
  --batch_size 4096 --lr 0.01 --weight_decay 1e-4 \
  --num_workers 4 --amp \
  --output_dir ./runs
```

Artifacts (logs, `final_model.pth`, per‑client ROC curves, `metrics.json`) are saved under `FedMedICL_COVID/experiments/<algo>_<rounds>R/<timestamp>/`.

### 2) Evaluate

```bash
python FedMedICL_COVID/evaluation.py \
  --dataset /path/to/COVID-19_Case_Surveillance_Public_Use_Data.csv \
  --checkpoint FedMedICL_COVID/experiments/feddg_50R/<run>/final_model.pth \
  --batch_size 4096 --num_workers 4 --use_adapter \
  --output_dir ./eval_results
```

Outputs: `eval_results/eval_metrics.json`, `eval_results/roc_curve.png`.

---

## Models & Training Algorithms

- **Backbone:** `model/TabTransformer.py` (categorical embeddings → Transformer encoder).
- **Dual BatchNorm:** `bn_local` (affine) + `bn_global` (running stats only); forward can switch between them.
- **BN Adapter:** `model/BNAdapter.py` learns an α∈[0,1] to mix instance/global stats (`adapt_forward`).
- **Classifier:** single FC layer.
- **Algorithms (see `training/`):** `FedDG` (default), `FedAvg`, `ERM`, `FedCB`, `FedDGHybrid`.

---

## Command‑Line Arguments (Training)

| Argument | Type | Default | Notes |
| --- | ---: | ---: | --- |
| `--dataset` | str | `/content/COVID-19_Case_Surveillance_Public_Use_Data.csv` | Path to CSV |
| `--model` | str | `tabtransformer` | Currently single model path is implemented |
| `--algorithm` | str | `FedDG` | Options listed above |
| `--num_clients` | int | `5` | Client count (or simulated if `HospitalID` missing) |
| `--num_tasks` | int | `4` | Reserved for future use |
| `--rounds` | int | `50` | Communication rounds |
| `--adapter_rounds` | int | `5` | BN‑adapter fine‑tuning rounds |
| `--emb_dim` | int | `64` | Categorical embedding size |
| `--hidden_dim` | int | `512` | Hidden dimension |
| `--n_layers` | int | `4` | Transformer encoder layers |
| `--transformer_heads` | int | `8` | Attention heads |
| `--dropout` | float | `0.1` | Dropout |
| `--batch_size` | int | `4096` | Batch size |
| `--num_workers` | int | `4` | DataLoader workers |
| `--lr` | float | `0.01` | Learning rate |
| `--weight_decay` | float | `1e-4` | L2 regularization |
| `--amp` | flag | off | Mixed precision if CUDA available |
| `--output_dir` | str | `./runs` | Base directory for experiment folders |

---


## Weights & Biases (W&B)

- The repository contains `wandb/` folders with offline runs. You can safely delete these from version control.
- To run offline: `export WANDB_MODE=offline`. To run online: `wandb login` then set `WANDB_PROJECT`/`WANDB_ENTITY`.

---

