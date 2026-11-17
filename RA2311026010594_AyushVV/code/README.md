# OSAP-Net — Few-Shot Detection of Rare and Previously Unseen Bearing Faults

PyTorch implementation for the SRM IST Deep Learning project *"Development of a Few-Shot Deep Learning Framework for Detection of Rare and Previously Unseen Failure Conditions"* (FJ3).

The project compares five methods on identical few-shot open-set episodes from the CWRU bearing dataset:

| Key | Method | Role |
|---|---|---|
| `cnn_scratch` | WDCNN + softmax, trained from scratch on each episode's K-shot support | Conventional supervised classifier |
| `transfer` | WDCNN pre-trained on the base classes + logistic-regression head | Transfer-learning baseline |
| `protonet` | Prototypical Network with a WDCNN backbone | Established few-shot baseline |
| `siamese` | Siamese WDCNN (Zhang et al., IEEE Access 2019) | **Base paper** |
| `osapnet` | Open-Set Attentive Prototypical Network | **Proposed** |
| `osap_no_se`, `osap_no_adaptive`, `osap_no_openloss` | OSAP-Net with one component removed | Ablations |

## Status

- **All experiments were executed:** 7 trainable models × 3 seeds trained; 8 methods × 3 seeds × 10 settings evaluated (129,000 test episodes).
- Trained checkpoints are in `outputs/checkpoints/`, seed-0 thresholds in `outputs/results/` (re-run `evaluate` to regenerate per-episode results), tables and charts in `outputs/report/`, and training logs in `outputs/logs/`.
- Seed 0 was run with `scripts/run_all.sh`. Seeds 1 and 2 were run in parallel with `scripts/run_one_seed.sh` (identical commands).
- Verified: 8/8 unit tests, the leakage audit (52/52 evaluation recordings disjoint), and the browser re-implementation (`scripts/export_demo.py`), which matches PyTorch to ≤ 1.5 × 10⁻⁶.

## 1. Install

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt                         # CPU is enough; a GPU is used automatically if present
export PYTHONPATH=src                                   # Windows PowerShell: $env:PYTHONPATH="src"
```

Google Colab: upload this folder, run `!pip install -r requirements.txt`, then prefix commands with `!PYTHONPATH=src`.

## 2. Dataset download

```bash
bash scripts/download_data.sh      # clones a public mirror of the CWRU data (~890 MB) into data/CWRU-dataset
```

Alternatively, download the *12k Drive End*, *12k Fan End* and *Normal Baseline* `.mat` files from the official CWRU Bearing Data Center (https://engineering.case.edu/bearingdatacenter/download-data-file) into `data/CWRU-dataset/`. Any sub-folder layout works. Files are found by their official number, e.g. `105.mat` or `105_0.mat`.

Note: files 3001–3008 (0.028" faults) store their signal under different variable names (`X056_DE_time` … `X051_DE_time`); the loader handles this.

## 3. Verify (seconds to a minute)

```bash
python scripts/check_leakage.py    # proves train/val/test classes and support/query windows are disjoint
bash scripts/smoke_test.sh         # runs every code path for a few steps (numbers are NOT results)
python -m pytest -q tests
```

## 4. Train

```bash
python -m osapnet.train --model osapnet --seed 0        # proposed
python -m osapnet.train --model siamese --seed 0        # base paper
python -m osapnet.train --model protonet --seed 0
python -m osapnet.train --model supervised --seed 0     # pre-training for the "transfer" baseline
```

Best checkpoints (selected on **validation classes only**) are saved to `outputs/checkpoints/`, and learning curves to `outputs/logs/`.

## 5. Evaluate

```bash
python -m osapnet.evaluate --model osapnet --seed 0 --protocol all
```

The protocols are:

- `same`: K = 1, 3, 5, 10
- `cross_load`: query recordings from a different motor load
- `cross_domain`: fan-end bearing and sensor
- `noise`: SNR −4, 0, 4, 8 dB
- `confusion`: a fixed known/unknown split

The threshold τ is calibrated on validation classes using only known-class scores (target FRR 5%).

## 6. Everything + report

```bash
bash scripts/run_all.sh            # all models × 3 seeds × all protocols (estimate: 6–10 h CPU, 1–2 h GPU)
python -m osapnet.report           # outputs/report/FJ3_Experimental_Results.xlsx + charts/*.png
```

## 7. Inference demo

```bash
python -m osapnet.infer --model osapnet --seed 0 --known IR014 B014 OR014@6 --query-class IR028 --k 5
```

This prints, for each incoming window, either a known fault label or `UNKNOWN FAILURE - inspect`, along with the open-set score and the threshold.

## 8. Project structure

```
configs/default.yaml        all hyper-parameters, class splits and protocols
src/osapnet/data/cwru.py    class registry, .mat loading, region A/B windowing, noise
src/osapnet/data/sampler.py few-shot open-set episode sampler
src/osapnet/models/         backbone.py (WDCNN + SE), fewshot.py (Siamese, OSAP-Net/ProtoNet, CNN)
src/osapnet/losses.py       CE + entropy + margin open-set loss
src/osapnet/metrics.py      accuracy, macro P/R/F1, AUROC, AUPRC, UDR, FAR, FRR, threshold
src/osapnet/engine.py       one-episode inference for every method
src/osapnet/train.py        episodic / Siamese / supervised training with validation
src/osapnet/evaluate.py     all protocols, identical episodes for all methods
src/osapnet/report.py       tables, Wilcoxon tests and charts from results
src/osapnet/infer.py        inference demonstration
scripts/                    run_all.sh, smoke_test.sh, check_leakage.py, download_data.sh, make_diagrams.py
tests/test_core.py          unit tests
```

## 9. Reproducibility

- Fixed seeds (0, 1, 2) and deterministic PyTorch algorithms.
- Episodes are generated from `(seed, protocol, K)`, so every method is scored on exactly the same episodes, which enables paired Wilcoxon tests.
- Class splits and every hyper-parameter live in `configs/default.yaml`.
- Verified with Python 3.13, PyTorch 2.x (CPU), NumPy, SciPy, scikit-learn 1.x.

## License and data

The code is for academic use. The CWRU data are provided by the Case Western Reserve University Bearing Data Center; cite their website and Smith & Randall (2015) when using them.
