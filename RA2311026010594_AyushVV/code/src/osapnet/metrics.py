"""
Evaluation metrics.

KNOWN-CLASS (closed-set) metrics use only queries from known classes and the
argmax over the N known classes:
    accuracy, macro precision, macro recall, macro F1

UNKNOWN-CLASS (open-set) metrics treat "unknown" as the POSITIVE class and use
the open-set score s(x) (larger = more unknown):
    AUROC  - threshold-free ranking quality of known vs unknown
    AUPRC  - area under precision-recall curve for detecting unknowns
With a threshold tau (calibrated on VALIDATION classes only):
    UDR  (unknown detection rate) = fraction of unknown queries rejected
    FAR  (false acceptance rate)  = fraction of unknown queries accepted as known = 1 - UDR
    FRR  (false rejection rate)   = fraction of known queries wrongly rejected
    open-set accuracy / macro-F1 over N known classes + 1 "unknown" class
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             precision_recall_fscore_support, roc_auc_score)


def episode_metrics(pred: np.ndarray, score: np.ndarray, qy: np.ndarray, n_way: int,
                    tau: float | None) -> dict:
    known = qy >= 0
    m = {}
    yk, pk = qy[known], pred[known]
    m["acc"] = accuracy_score(yk, pk)
    p, r, f, _ = precision_recall_fscore_support(yk, pk, labels=list(range(n_way)),
                                                 average="macro", zero_division=0)
    m.update(precision=p, recall=r, macro_f1=f)
    if (~known).any():
        is_unk = (~known).astype(int)
        m["auroc"] = roc_auc_score(is_unk, score)
        m["auprc"] = average_precision_score(is_unk, score)
        if tau is not None:
            rej = score > tau
            m["udr"] = rej[~known].mean()
            m["far"] = 1.0 - m["udr"]
            m["frr"] = rej[known].mean()
            open_pred = np.where(rej, -1, pred)                # -1 = "unknown"
            m["open_acc"] = (open_pred == qy).mean()
            m["open_macro_f1"] = f1_score(qy, open_pred, labels=list(range(-1, n_way)),
                                          average="macro", zero_division=0)
    return m


def calibrate_threshold(known_scores: np.ndarray, target_frr: float = 0.05) -> float:
    """tau = (1 - target_frr) quantile of KNOWN-class scores from validation episodes.
    No unknown sample is needed, so the threshold cannot leak test information."""
    return float(np.quantile(known_scores, 1.0 - target_frr))
