"""
Training losses.

L_total = L_CE                       (classify known queries)
        + lambda_ent * L_ent         (pseudo-unknown queries -> uniform prediction)
        + lambda_m   * L_margin      (known close to own prototype, unknown far from all)

L_ent    = 1 - H(p(.|x_u)) / log N            (0 when prediction is perfectly uniform)
L_margin = mean_k relu(d_y(x_k) - m_in) + mean_u relu(m_out - min_c d_c(x_u))
The entropy term follows the open-set meta-learning idea of Liu et al. (CVPR 2020);
the margin term acts directly on the rejection score used at test time.
"""
import math

import torch
import torch.nn.functional as F


def episode_loss(out: dict, qy: torch.Tensor, n_way: int, cfg: dict, open_loss: bool):
    known = qy >= 0
    logits = out["logits"]
    l_ce = F.cross_entropy(logits[known], qy[known])
    logs = {"ce": l_ce.item()}
    loss = l_ce
    if open_loss and (~known).any():
        p_u = torch.softmax(logits[~known], 1)
        H = -(p_u * torch.log(p_u.clamp_min(1e-8))).sum(1)
        l_ent = (1.0 - H / math.log(n_way)).mean()
        d = out["dist"]
        l_m = F.relu(d[known].gather(1, qy[known][:, None]).squeeze(1) - cfg["m_in"]).mean() \
            + F.relu(cfg["m_out"] - d[~known].min(1).values).mean()
        loss = loss + cfg["lambda_ent"] * l_ent + cfg["lambda_margin"] * l_m
        logs.update(ent=l_ent.item(), margin=l_m.item())
    logs["total"] = loss.item()
    return loss, logs
