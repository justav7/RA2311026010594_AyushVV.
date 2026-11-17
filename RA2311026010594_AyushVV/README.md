# Few-Shot Deep Learning Framework for Detection of Rare and Previously Unseen Failure Conditions

**OSAP-Net: Open-Set Attentive Prototypical Network for bearing fault diagnosis**

21CSC406J – Neural Networks and Deep Learning · Case Study
Department of Computational Intelligence, SRM Institute of Science and Technology, Kattankulathur

| Name | Register number |
|---|---|
| Ayush VV | RA2311026010594 |
| Afnaan Shafir | RA2311026010606 |
| Monish | RA2311026010596 |

## What this project does
Machines run on bearings, and a damaged bearing changes the machine's vibration. Deep-learning models can recognise bearing faults from vibration, but normally only when they have hundreds of labelled examples of every fault, and they always force a never-seen fault into a known name.

**OSAP-Net** learns a fault from only **1–10 labelled examples**. When it meets a fault it has **never seen**, it answers **UNKNOWN** instead of guessing. We compare it with the base paper's model (Siamese WDCNN, Zhang et al., *IEEE Access*, 2019) on the CWRU bearing dataset.

## Folder contents
| Folder | Contents |
|---|---|
| `Presentation/` | Presentation comparing the old and new models (`.pptx` and `.pdf`) |
| `Report/` | NNDL case-study report (`.docx` and `.pdf`) and the full research report (`.pdf`) |
| `Demo/` | `FJ3_Live_Demo.html`: the trained model running in the browser. Download it and open it in Chrome or Edge (it works offline) |
| `code/` | Complete PyTorch source code, the 21 trained models and the measured results |

## Results (mean of 3 runs, same motor load)
| Method | Accuracy (K=1) | AUROC (K=1) | Accuracy (K=5) | AUROC (K=5) |
|---|---|---|---|---|
| Old model — Siamese WDCNN (base paper) | 78.3% | 0.767 | 83.4% | 0.810 |
| ProtoNet (WDCNN) | 93.0% | 0.905 | 97.7% | 0.952 |
| Transfer CNN | 95.4% | 0.710 | 98.9% | 0.768 |
| CNN retrained every episode | 98.2% | 0.915 | 99.9% | 0.964 |
| **OSAP-Net (ours)** | 95.0% | 0.936 | 97.9% | 0.966 |

- **Accuracy:** known faults named correctly.
- **AUROC:** how well unseen faults are separated from known ones (1.0 = perfect).
- **K:** the number of labelled examples per fault.

Key findings:
- **Against the base paper:** OSAP-Net is better in all 10 test conditions (paired Wilcoxon p ≤ 5.7×10⁻⁵). Unseen faults are wrongly accepted 16.1% of the time, versus 54.2% for the old model (K=1).
- **What helped:** the gain comes from open-set training with pretend-unknown faults. SE attention and the adaptive prototype showed no measurable benefit.
- **Speed trade-off:** a CNN retrained on every episode is more accurate on known faults, but it takes 0.70 s per episode versus 5.7 ms for OSAP-Net, and it must be retrained whenever a fault is added.

## How to run the code
```
cd code
pip install -r requirements.txt
git clone --depth 1 https://github.com/s-whynot/CWRU-dataset.git data/CWRU-dataset
```
Then point Python at the code:

| System | Command |
|---|---|
| Windows (cmd) | `set PYTHONPATH=src` |
| Windows (PowerShell) | `$env:PYTHONPATH="src"` |
| Mac / Linux | `export PYTHONPATH=src` |

Run these in order:
```
python scripts/check_leakage.py      # data-leakage audit
python -m pytest -q tests            # unit tests
python -m osapnet.infer --model osapnet --seed 0 --known IR014 B014 OR014@6 --query-class IR028
```
The last command uses the trained model (no training needed). IR028 was never seen, so every window should be reported as **UNKNOWN FAILURE**.

Optional steps:
- `python -m osapnet.evaluate --model osapnet --seed 0 --protocol same` re-tests the saved model.
- `bash scripts/run_all.sh` retrains everything (about 2–3 hours on a laptop CPU).

More detail is in [`code/README.md`](code/README.md).

## Dataset
CWRU Bearing Data Center (Case Western Reserve University), 12 kHz drive-end and fan-end data. The dataset is not stored in this repository; download it with the command above.
