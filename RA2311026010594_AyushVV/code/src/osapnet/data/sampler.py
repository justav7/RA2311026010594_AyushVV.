"""
Few-shot OPEN-SET episode sampler.

An *episode* imitates one deployment situation:
  * N "known" fault classes, each with only K labelled windows (the support set)
  * query windows from those N classes (must be classified)
  * query windows from U "unknown" classes that have NO support examples
    (must be rejected as "unknown failure")

Training episodes are drawn from training (base) classes; the U unknown classes
there are "pseudo-unknowns" used by the open-set loss. Evaluation episodes are
drawn from validation or test classes that the network has never seen.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .cwru import CWRUBank, add_noise, zscore


@dataclass
class Episode:
    support: np.ndarray        # (N*K, L)
    support_y: np.ndarray      # (N*K,) values 0..N-1
    query: np.ndarray          # (Q_total, L)
    query_y: np.ndarray        # (Q_total,) 0..N-1 for known, -1 for unknown
    known_names: List[str]
    unknown_names: List[str]
    query_names: List[str]     # true class name of each query window


class EpisodeSampler:
    def __init__(self, bank: CWRUBank, seed: int = 0):
        self.bank = bank
        self.rng = np.random.default_rng(seed)

    # ---------------------------------------------------------- training
    def train_episode(self, classes: List[str], n_known: int, n_unknown: int,
                      k_shot: int, q_known: int, q_unknown: int) -> Episode:
        """Episode from TRAINING classes. Support/query indices are disjoint."""
        rng = self.rng
        chosen = list(rng.choice(classes, n_known + n_unknown, replace=False))
        known, unknown = chosen[:n_known], chosen[n_known:]
        S, Sy, Q, Qy, Qn = [], [], [], [], []
        for i, c in enumerate(known):
            pool = np.concatenate([self.bank.train[c][l] for l in self.bank.loads])
            idx = rng.choice(len(pool), k_shot + q_known, replace=False)
            S.append(pool[idx[:k_shot]]); Sy += [i] * k_shot
            Q.append(pool[idx[k_shot:]]); Qy += [i] * q_known; Qn += [c] * q_known
        for c in unknown:
            pool = np.concatenate([self.bank.train[c][l] for l in self.bank.loads])
            idx = rng.choice(len(pool), q_unknown, replace=False)
            Q.append(pool[idx]); Qy += [-1] * q_unknown; Qn += [c] * q_unknown
        return Episode(zscore(np.concatenate(S)), np.array(Sy), zscore(np.concatenate(Q)),
                       np.array(Qy), known, unknown, Qn)

    # -------------------------------------------------------- evaluation
    def eval_episode(self, classes: List[str], n_known: int, k_shot: int, q_per_class: int,
                     protocol: str = "same", snr_db: Optional[float] = None,
                     shuffle: bool = True) -> Episode:
        """Episode from VALIDATION/TEST classes.

        protocol = "same"       support and query from the same motor load
                   "cross_load" query recordings come from a different load
        Support windows always come from region A, queries from region B.
        All classes in `classes` that are not chosen as known become unknown.
        """
        rng = self.rng
        perm = list(rng.permutation(classes)) if shuffle else list(classes)
        known, unknown = perm[:n_known], perm[n_known:]
        l_s = int(rng.choice(self.bank.loads))
        if protocol == "same":
            l_q = l_s
        elif protocol == "cross_load":
            l_q = int(rng.choice([l for l in self.bank.loads if l != l_s]))
        else:
            raise ValueError(protocol)
        S, Sy, Q, Qy, Qn = [], [], [], [], []
        for i, c in enumerate(known):
            a = self.bank.eval[c][l_s].support
            S.append(a[rng.choice(len(a), k_shot, replace=False)]); Sy += [i] * k_shot
            b = self.bank.eval[c][l_q].query
            Q.append(b[rng.choice(len(b), q_per_class, replace=False)]); Qy += [i] * q_per_class
            Qn += [c] * q_per_class
        for c in unknown:
            b = self.bank.eval[c][l_q].query
            Q.append(b[rng.choice(len(b), q_per_class, replace=False)]); Qy += [-1] * q_per_class
            Qn += [c] * q_per_class
        S, Q = np.concatenate(S), np.concatenate(Q)
        if snr_db is not None:                      # noise added to the RAW signal
            S, Q = add_noise(S, snr_db, rng), add_noise(Q, snr_db, rng)
        return Episode(zscore(S), np.array(Sy), zscore(Q), np.array(Qy), known, unknown, Qn)
