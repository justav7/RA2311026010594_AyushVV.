"""
Aggregate per-episode results into tables and figures.

python -m osapnet.report
Reads  outputs/results/episodes_*.csv, scores_*.npz, confusion_*.json, outputs/logs/*.csv
Writes outputs/report/FJ3_Experimental_Results.xlsx  (+ CSV copies)
       outputs/report/charts/*.png
Nothing is written for experiments that have not been run.
"""
from __future__ import annotations

import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

METRICS = ["acc", "precision", "recall", "macro_f1", "auroc", "auprc", "udr", "far", "frr",
           "open_acc", "open_macro_f1"]
NAMES = {"cnn_scratch": "Supervised CNN (scratch)", "transfer": "Transfer CNN (pretrain+LR)",
         "protonet": "ProtoNet (WDCNN)", "siamese": "Siamese WDCNN [base]",
         "osapnet": "OSAP-Net (proposed)", "osap_no_se": "OSAP-Net w/o SE",
         "osap_no_adaptive": "OSAP-Net w/o adaptive proto", "osap_no_openloss": "OSAP-Net w/o open-set loss"}
MAIN = ["cnn_scratch", "transfer", "protonet", "siamese", "osapnet"]
ABL = ["osapnet", "osap_no_se", "osap_no_adaptive", "osap_no_openloss", "protonet"]
COL = {"cnn_scratch": "#9AA5B1", "transfer": "#6B7C93", "protonet": "#2E86AB",
       "siamese": "#F18F01", "osapnet": "#0B3C5D"}


def load_cfg_out():
    import yaml
    with open("configs/default.yaml") as f:
        return yaml.safe_load(f)["paths"]["outputs"]


def main():
    out = load_cfg_out()
    res, rep = os.path.join(out, "results"), os.path.join(out, "report")
    ch = os.path.join(rep, "charts")
    os.makedirs(ch, exist_ok=True)
    files = glob.glob(os.path.join(res, "episodes_*.csv"))
    if not files:
        print("No results found - run experiments first (scripts/run_all.sh).")
        return
    ep = pd.concat([pd.read_csv(f) for f in files])
    per_seed = ep.groupby(["model", "tag", "seed"])[[m for m in METRICS if m in ep]].mean().reset_index()
    agg = per_seed.groupby(["model", "tag"]).agg(["mean", "std"])
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    agg = agg.drop(columns=[c for c in agg.columns if c.startswith("seed")]).reset_index()
    agg.insert(1, "method", agg.model.map(NAMES))

    # paired Wilcoxon signed-rank test: proposed vs base on identical episodes
    tests = []
    for tag in ep.tag.unique():
        a = ep[(ep.model == "osapnet") & (ep.tag == tag)].sort_values(["seed", "episode"])
        b = ep[(ep.model == "siamese") & (ep.tag == tag)].sort_values(["seed", "episode"])
        if len(a) and len(a) == len(b):
            for m in ["acc", "auroc"]:
                d = a[m].values - b[m].values
                p = wilcoxon(d).pvalue if np.any(d != 0) else 1.0
                tests.append({"tag": tag, "metric": m, "mean_diff": d.mean(), "p_value": p, "n_episodes": len(d)})
    with pd.ExcelWriter(os.path.join(rep, "FJ3_Experimental_Results.xlsx")) as xw:
        agg.to_excel(xw, "summary_mean_std", index=False)
        per_seed.to_excel(xw, "per_seed", index=False)
        pd.DataFrame(tests).to_excel(xw, "wilcoxon_osap_vs_base", index=False)
    agg.to_csv(os.path.join(rep, "summary_mean_std.csv"), index=False)
    pd.DataFrame(tests).to_csv(os.path.join(rep, "wilcoxon_osap_vs_base.csv"), index=False)

    def get(model, tag, m):
        r = agg[(agg.model == model) & (agg.tag == tag)]
        return (r[f"{m}_mean"].item(), r[f"{m}_std"].fillna(0).item()) if len(r) else (np.nan, 0)

    # --- K-shot curves
    shots = sorted({int(t.split("k")[-1]) for t in agg.tag if t.startswith("same-k")})
    for m, lab in [("acc", "Known-class accuracy"), ("auroc", "Unknown-detection AUROC"),
                   ("macro_f1", "Macro F1 (known)"), ("udr", "Unknown detection rate")]:
        plt.figure(figsize=(6, 4))
        for mod in MAIN:
            v = [get(mod, f"same-k{k}", m) for k in shots]
            if not np.all(np.isnan([x[0] for x in v])):
                plt.errorbar(shots, [x[0] for x in v], [x[1] for x in v], marker="o", capsize=3,
                             label=NAMES[mod], color=COL[mod])
        plt.xlabel("K (labelled samples per known class)"); plt.ylabel(lab); plt.xticks(shots)
        plt.grid(alpha=.3); plt.legend(fontsize=8); plt.tight_layout()
        plt.savefig(os.path.join(ch, f"shots_{m}.png"), dpi=200); plt.close()
    # --- noise
    snr_tags = sorted({t for t in agg.tag if t.startswith("noise")}, key=lambda t: int(t[5:].split("dB")[0]))
    if snr_tags:
        snrs = [int(t[5:].split("dB")[0]) for t in snr_tags]
        for m in ["acc", "auroc"]:
            plt.figure(figsize=(6, 4))
            for mod in MAIN:
                v = [get(mod, t, m) for t in snr_tags]
                plt.errorbar(snrs, [x[0] for x in v], [x[1] for x in v], marker="s", capsize=3,
                             label=NAMES[mod], color=COL[mod])
            plt.xlabel("SNR (dB)"); plt.ylabel(m.upper()); plt.grid(alpha=.3); plt.legend(fontsize=8)
            plt.tight_layout(); plt.savefig(os.path.join(ch, f"noise_{m}.png"), dpi=200); plt.close()
    # --- condition shift bars
    for m in ["acc", "auroc"]:
        tags = ["same-k5", "cross_load-k5", "cross_domain-k5"]
        x = np.arange(len(tags)); w = 0.16
        plt.figure(figsize=(7, 4))
        for i, mod in enumerate(MAIN):
            v = [get(mod, t, m) for t in tags]
            plt.bar(x + (i - 2) * w, [a[0] for a in v], w, yerr=[a[1] for a in v], label=NAMES[mod], color=COL[mod])
        plt.xticks(x, ["Same load", "Cross load", "Cross domain (FE)"]); plt.ylabel(m.upper())
        plt.legend(fontsize=7, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.01), frameon=False)
        plt.grid(axis="y", alpha=.3); plt.tight_layout()
        plt.savefig(os.path.join(ch, f"shift_{m}.png"), dpi=200); plt.close()
    # --- ROC curves (seed 0, K=5)
    plt.figure(figsize=(5, 5))
    for mod in MAIN:
        f = os.path.join(res, f"scores_{mod}_seed0.npz")
        if os.path.exists(f):
            z = np.load(f)
            if "same_k5_score" in z:
                fpr, tpr, _ = roc_curve(z["same_k5_unknown"], z["same_k5_score"])
                plt.plot(fpr, tpr, label=NAMES[mod], color=COL[mod])
    plt.plot([0, 1], [0, 1], "k--", lw=.8); plt.xlabel("False positive rate (known flagged unknown)")
    plt.ylabel("True positive rate (unknown detected)"); plt.legend(fontsize=7); plt.tight_layout()
    plt.savefig(os.path.join(ch, "roc_k5.png"), dpi=200); plt.close()
    # --- confusion matrices
    for f in glob.glob(os.path.join(res, "confusion_*_seed0.json")):
        d = json.load(open(f)); cm = np.array(d["matrix"]); mod = os.path.basename(f)[10:-11]
        cmn = cm / cm.sum(1, keepdims=True)
        plt.figure(figsize=(4.6, 4))
        plt.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
        for i in range(4):
            for j in range(4):
                plt.text(j, i, f"{cmn[i, j]:.2f}", ha="center", va="center",
                         color="white" if cmn[i, j] > .5 else "black", fontsize=9)
        plt.xticks(range(4), d["labels"], rotation=30); plt.yticks(range(4), d["labels"])
        plt.xlabel("Predicted"); plt.ylabel("True"); plt.title(NAMES.get(mod, mod), fontsize=9)
        plt.tight_layout(); plt.savefig(os.path.join(ch, f"confusion_{mod}.png"), dpi=200); plt.close()
    # --- training curves
    for f in glob.glob(os.path.join(out, "logs", "*_seed0_curve.csv")):
        d = pd.read_csv(f); mod = os.path.basename(f).replace("_seed0_curve.csv", "")
        fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
        ax[0].plot(d.step, d.total.rolling(50, min_periods=1).mean()); ax[0].set_title("Training loss (moving avg)")
        v = d.dropna(subset=["val_acc"])
        ax[1].plot(v.step, v.val_acc, "o-", label="val accuracy"); ax[1].plot(v.step, v.val_auroc, "s-", label="val AUROC")
        ax[1].legend(); ax[1].set_title("Validation (val classes only)")
        for a_ in ax: a_.set_xlabel("step"); a_.grid(alpha=.3)
        fig.suptitle(NAMES.get(mod, mod)); fig.tight_layout()
        fig.savefig(os.path.join(ch, f"curve_{mod}.png"), dpi=200); plt.close(fig)
    print(f"Report written to {rep}")


if __name__ == "__main__":
    main()
