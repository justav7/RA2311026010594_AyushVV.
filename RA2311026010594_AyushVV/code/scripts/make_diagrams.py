"""Generate the methodology figures used in FJ1 and FJ3 (python scripts/make_diagrams.py OUTDIR).
All figures are schematic drawings of the implemented method, except fig_signals,
which plots real CWRU windows. None of them contains experimental results."""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch  # noqa: E402

OUT = sys.argv[1] if len(sys.argv) > 1 else "figures"
os.makedirs(OUT, exist_ok=True)
NAVY, NAVY2, CYAN, BLUE, GREY, RED, GREEN, AMBER = "#0B1F3A", "#16335C", "#0EA5C6", "#2563EB", "#64748B", "#DC2626", "#059669", "#D97706"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})


def box(ax, x, y, w, h, text, fc=NAVY, tc="white", fs=10, ec=None, bold=False, r=0.08):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.02,rounding_size={r}",
                                fc=fc, ec=ec or fc, lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", color=tc, fontsize=fs,
            fontweight="bold" if bold else "normal", linespacing=1.3)


def arrow(ax, x1, y1, x2, y2, c=NAVY, lw=1.6, style="-|>"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=14, color=c, lw=lw))


def canvas(w, h, xl, yl):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, xl); ax.set_ylim(0, yl); ax.axis("off")
    return fig, ax


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------- 1 architecture
fig, ax = canvas(14, 6.2, 14, 6.2)
ax.text(7, 5.95, "OSAP-Net: Open-Set Attentive Prototypical Network", ha="center", fontsize=15, fontweight="bold", color=NAVY)
# input
t = np.linspace(0, 1, 300); sig = np.sin(2 * np.pi * 9 * t) * 0.3 + 0.35 * (np.random.default_rng(1).standard_normal(300)) * np.exp(-((t * 10) % 1) * 4)
ax.plot(0.25 + t * 1.5, 4.1 + sig * 0.45, color=NAVY, lw=0.8)
ax.text(1.0, 3.35, "Raw vibration window\n2048 samples, z-scored", ha="center", fontsize=9, color=NAVY)
blocks = [("Conv k64 s16\n16 ch\nBN·ReLU\nMaxPool", BLUE), ("Conv k3 · 32 ch\nBN·ReLU\nSE attention\nMaxPool", BLUE),
          ("Conv k2 · 64 ch\nBN·ReLU\nSE attention\nMaxPool", BLUE), ("Conv k3 · 64 ch\nBN·ReLU\nSE attention\nMaxPool", BLUE),
          ("Conv k3 · 64 ch\nBN·ReLU\nSE attention\nMaxPool", BLUE)]
x0 = 2.1
for i, (txt, c) in enumerate(blocks):
    box(ax, x0 + i * 1.32, 3.35, 1.18, 1.55, txt, fc=c, fs=7.6)
    if i:
        arrow(ax, x0 + i * 1.32 - 0.13, 4.12, x0 + i * 1.32 - 0.01, 4.12)
arrow(ax, 1.8, 4.12, 2.08, 4.12)
ax.text(x0 + 2.6, 5.15, "WDCNN backbone with Squeeze-and-Excitation channel attention", ha="center", fontsize=10, color=BLUE, fontweight="bold")
box(ax, 8.85, 3.6, 1.05, 1.05, "Flatten 192\n→ FC\nz ∈ ℝ¹⁰⁰", fc=NAVY2, fs=8.5)
arrow(ax, 8.6, 4.12, 8.83, 4.12)
# prototype head
box(ax, 10.3, 3.25, 3.4, 1.75, "Adaptive prototype head\n• reliability weights aᵢ\n• prototype p_c = Σ aᵢ zᵢ\n• shrinkage variance σ_c²\n• normalised distance d̃_c(x)", fc=CYAN, fs=8.8, tc="white")
arrow(ax, 9.92, 4.12, 10.28, 4.12)
# support
box(ax, 10.3, 5.25, 3.4, 0.55, "Support set: K labelled windows × N known faults", fc="white", tc=NAVY, ec=NAVY, fs=7.2)
arrow(ax, 12.0, 5.24, 12.0, 5.02)
# outputs
box(ax, 8.4, 1.25, 2.5, 1.2, "Softmax over known faults\np(c|x) ∝ exp(−γ(d̃_c + log σ_c²))", fc="white", tc=NAVY, ec=BLUE, fs=8.4)
box(ax, 11.2, 1.25, 2.5, 1.2, "Open-set score\ns(x) = min_c d̃_c(x)", fc="white", tc=NAVY, ec=CYAN, fs=9)
arrow(ax, 11.3, 3.23, 9.8, 2.47); arrow(ax, 12.45, 3.23, 12.45, 2.47)
box(ax, 8.4, 0.05, 5.3, 0.8, "Decision:  s(x) ≤ τ → known fault = argmax p(c|x)\ns(x) > τ → UNKNOWN FAILURE", fc=NAVY, fs=9, bold=True)
arrow(ax, 9.65, 1.23, 9.9, 0.87); arrow(ax, 12.45, 1.23, 12.2, 0.87)
# training box
box(ax, 0.2, 0.05, 7.7, 2.6, "", fc="#EEF6FB", ec=CYAN, r=0.1)
ax.text(4.05, 2.35, "Episodic open-set training (base classes only)", ha="center", fontsize=10.5, fontweight="bold", color=NAVY)
ax.text(0.45, 1.25, "• 5 known + 2 pseudo-unknown classes per episode, K ~ U{1..10}\n"
                    "• L = L_CE  (classify known queries)\n"
                    "   + λ_ent · L_ent  (pseudo-unknowns → uniform prediction)\n"
                    "   + λ_m · L_margin (known near own prototype, unknown far)\n"
                    "• τ calibrated on validation classes (known-only, FRR = 5 %)",
        fontsize=9, color=NAVY, va="center", linespacing=1.45)
save(fig, "fig_architecture.png")

# ---------------------------------------------------------------- 2 workflow
fig, ax = canvas(14, 4.6, 14, 4.6)
steps = [("1  Data", "CWRU .mat files\n16 DE + 5 FE classes\n4 motor loads"),
         ("2  Preprocess", "2048-pt windows\nz-score per window\nregion A / B split"),
         ("3  Class split", "8 train · 3 val\n5 test (unseen)\n5 cross-domain"),
         ("4  Episodes", "N-way K-shot\n+ unknown queries\nK = 1, 3, 5, 10"),
         ("5  Train", "Siamese (base)\nProtoNet · OSAP-Net\nablations · CNNs"),
         ("6  Calibrate τ", "validation classes\nknown scores only\nFRR target 5 %"),
         ("7  Evaluate", "same / cross-load\ncross-domain · noise\n3 seeds × 600 eps"),
         ("8  Analyse", "mean ± std\nWilcoxon test\nablation · ROC")]
w = 1.55
for i, (h, b) in enumerate(steps):
    x = 0.15 + i * 1.73
    c = CYAN if i in (2, 5) else NAVY
    box(ax, x, 2.35, w, 0.65, h, fc=c, fs=10.5, bold=True)
    box(ax, x, 0.75, w, 1.45, b, fc="#EEF3F9", tc=NAVY, ec="#C9D6E6", fs=8.0)
    if i < len(steps) - 1:
        arrow(ax, x + w + 0.02, 2.67, x + 1.72, 2.67)
ax.text(7, 4.15, "End-to-end research methodology", ha="center", fontsize=15, fontweight="bold", color=NAVY)
ax.text(7, 3.55, "Leakage controls: class-disjoint splits  ·  support only from region A, queries only from region B  ·  τ never sees test or unknown data",
        ha="center", fontsize=9.5, color=CYAN, fontweight="bold")
ax.text(7, 0.3, "Identical episodes (same random seeds) are used for every compared method → paired statistical comparison", ha="center", fontsize=9, color=GREY)
save(fig, "fig_workflow.png")

# ---------------------------------------------------------------- 3 dataset split
fig, ax = canvas(14, 5.6, 14, 5.6)
ax.text(7, 5.3, "Fault-condition-disjoint class split and leakage-controlled windowing", ha="center", fontsize=14, fontweight="bold", color=NAVY)
groups = [("TRAIN (base) – 8 classes", ["Normal", "IR007", "IR021", "B007", "OR007@6", "OR021@6", "OR007@3", "OR021@12"], NAVY),
          ("VALIDATION – 3", ["B021", "OR007@12", "OR021@3"], AMBER),
          ("TEST (never seen) – 5", ["IR014", "B014", "OR014@6", "IR028", "B028"], CYAN),
          ("CROSS-DOMAIN (fan end) – 5", ["FE-IR014", "FE-B014", "FE-IR021", "FE-B021", "FE-OR007@3"], GREEN)]
xs = [0.2, 4.4, 7.05, 10.35]; ws = [4.0, 2.45, 3.1, 3.45]
for (title, cl, c), x, w_ in zip(groups, xs, ws):
    box(ax, x, 4.35, w_, 0.55, title, fc=c, fs=9.5, bold=True)
    cols = 1 if "VALID" in title else 2
    for j, n in enumerate(cl):
        cx = x + 0.08 + (j % cols) * (w_ - 0.16) / cols
        cy = 3.75 - (j // cols) * 0.5
        box(ax, cx, cy, (w_ - 0.16) / cols - 0.08, 0.4, n, fc="white", tc=c, ec=c, fs=8.6)
ax.text(0.2, 1.75, "One evaluation recording (≈121 k samples, ≈10 s):", fontsize=10, color=NAVY, fontweight="bold")
ax.add_patch(FancyBboxPatch((0.2, 0.55), 6.3, 0.75, boxstyle="round,pad=0.01", fc="#DBEAFE", ec=BLUE))
ax.add_patch(FancyBboxPatch((6.85, 0.55), 6.6, 0.75, boxstyle="round,pad=0.01", fc="#CFFAFE", ec=CYAN))
for i in range(14):
    ax.add_patch(plt.Rectangle((0.28 + i * 0.44, 0.65), 0.4, 0.55, fc=BLUE, alpha=.55, ec="white"))
for i in range(15):
    ax.add_patch(plt.Rectangle((6.93 + i * 0.43, 0.65), 0.39, 0.55, fc=CYAN, alpha=.65, ec="white"))
ax.text(3.35, 0.25, "Region A (first half): SUPPORT windows only — 28 non-overlapping", ha="center", fontsize=9, color=BLUE)
ax.text(10.15, 0.25, "Region B (second half): QUERY windows only — 29 non-overlapping", ha="center", fontsize=9, color="#0E7490")
ax.text(6.67, 0.92, "guard\ngap", ha="center", fontsize=7.5, color=RED)
save(fig, "fig_dataset_split.png")

# ---------------------------------------------------------------- 4 few-shot episode
fig, ax = canvas(13, 5, 13, 5)
ax.text(6.5, 4.7, "One few-shot open-set test episode (3-way, K-shot, 2 unknown classes)", ha="center", fontsize=13.5, fontweight="bold", color=NAVY)
box(ax, 0.2, 0.3, 5.2, 3.95, "", fc="#EEF6FB", ec=BLUE)
ax.text(2.8, 3.95, "SUPPORT SET (labelled, K per class)", ha="center", fontsize=10, fontweight="bold", color=BLUE)
rng = np.random.default_rng(3)
for r, (n, c) in enumerate([("IR014", BLUE), ("B014", GREEN), ("OR014@6", AMBER)]):
    ax.text(0.45, 3.25 - r * 1.05, n, fontsize=9.5, color=c, fontweight="bold", va="center")
    for k in range(5):
        x = 1.6 + k * 0.72; y = 3.25 - r * 1.05
        tt = np.linspace(0, 1, 80)
        ax.plot(x + tt * 0.6, y + 0.18 * rng.standard_normal(80) * np.exp(-((tt * (4 + r)) % 1) * 3), color=c, lw=.7)
ax.text(2.8, 0.45, "K = 1, 3, 5 or 10 windows per known fault", ha="center", fontsize=8.5, color=GREY)
box(ax, 6.0, 0.3, 6.8, 3.95, "", fc="#F8FAFC", ec=NAVY)
ax.text(9.4, 3.95, "QUERY SET (to be diagnosed)", ha="center", fontsize=10, fontweight="bold", color=NAVY)
qs = [("IR014", BLUE, "→ IR014"), ("B014", GREEN, "→ B014"), ("OR014@6", AMBER, "→ OR014@6"), ("IR028 (unseen)", RED, "→ UNKNOWN"), ("B028 (unseen)", RED, "→ UNKNOWN")]
for r, (n, c, out) in enumerate(qs):
    y = 3.45 - r * 0.66
    ax.text(6.25, y, n, fontsize=9, color=c, fontweight="bold", va="center")
    tt = np.linspace(0, 1, 80)
    ax.plot(8.0 + tt * 1.5, y + 0.15 * rng.standard_normal(80), color=c, lw=.7)
    ax.text(10.0, y, out, fontsize=9.5, color=c, va="center", fontweight="bold")
ax.text(9.4, 0.45, "15 query windows per class; unknown classes have NO support examples", ha="center", fontsize=8.5, color=GREY)
arrow(ax, 5.45, 2.3, 5.95, 2.3)
save(fig, "fig_fewshot_episode.png")

# ---------------------------------------------------------------- 5 known vs unknown
fig, ax = plt.subplots(figsize=(11, 5.4))
ax.set_xlim(-6, 8); ax.set_ylim(-4.2, 4.6); ax.set_aspect("equal"); ax.axis("off")
ax.set_title("Known vs. unseen failure recognition in the embedding space", fontsize=13.5, fontweight="bold", color=NAVY)
rng = np.random.default_rng(7)
for (cx, cy, s, c, n) in [(-3.5, 1.6, .55, BLUE, "IR014"), (-1.2, -2.1, .8, GREEN, "B014"), (1.6, 1.9, .45, AMBER, "OR014@6")]:
    pts = rng.normal([cx, cy], s * 0.7, (25, 2)); ax.scatter(pts[:, 0], pts[:, 1], s=12, color=c, alpha=.55)
    ax.add_patch(Circle((cx, cy), s * 2.1, fill=False, ls="--", color=c, lw=1.4))
    ax.scatter([cx], [cy], marker="*", s=320, color=c, edgecolor=NAVY, zorder=5)
    ax.text(cx, cy + s * 2.1 + .2, n, ha="center", fontsize=10, color=c, fontweight="bold")
u = rng.normal([4.6, -1.8], .5, (18, 2)); ax.scatter(u[:, 0], u[:, 1], s=24, marker="x", color=RED)
ax.text(4.6, -3.25, "Unseen fault (IR028)\ns(x) > τ  →  UNKNOWN", ha="center", fontsize=9.5, color=RED, fontweight="bold")
q = np.array([-3.1, 1.2]); ax.scatter(*q, s=80, color=NAVY, zorder=6); ax.annotate("query inside IR014 boundary\n→ classified IR014", q, (-5.9, -0.6), fontsize=8.5, color=NAVY, arrowprops=dict(arrowstyle="->", color=NAVY))
ax.text(5.3, 3.4, "Normalised distance\n d̃_c(x) = ‖z − p_c‖² / (D σ_c²)\n s(x) = min_c d̃_c(x)\n dashed boundary: d̃_c = τ\n ★ prototype p_c, radius ∝ σ_c", fontsize=9, color=NAVY, va="top",
        bbox=dict(boxstyle="round", fc="#EEF6FB", ec=CYAN))
save(fig, "fig_known_unknown.png")

# ---------------------------------------------------------------- 6 base vs proposed
fig, ax = canvas(14, 6, 14, 6)
ax.text(3.4, 5.65, "Base paper: Siamese WDCNN [Zhang et al. 2019]", ha="center", fontsize=12, fontweight="bold", color=AMBER)
ax.text(10.5, 5.65, "Proposed: OSAP-Net", ha="center", fontsize=12, fontweight="bold", color=CYAN)
box(ax, 0.3, 4.4, 2.6, 0.75, "Window A", fc="white", tc=NAVY, ec=NAVY, fs=9.5)
box(ax, 3.9, 4.4, 2.6, 0.75, "Window B", fc="white", tc=NAVY, ec=NAVY, fs=9.5)
box(ax, 0.3, 3.15, 2.6, 0.9, "WDCNN (shared)\nsigmoid emb. 100", fc=AMBER, fs=9)
box(ax, 3.9, 3.15, 2.6, 0.9, "WDCNN (shared)\nsigmoid emb. 100", fc=AMBER, fs=9)
arrow(ax, 1.6, 4.38, 1.6, 4.07); arrow(ax, 5.2, 4.38, 5.2, 4.07)
box(ax, 1.9, 1.95, 3.0, 0.8, "|z_A − z_B| → Dropout → FC\nsigmoid  P(same class)", fc=NAVY2, fs=9)
arrow(ax, 1.6, 3.13, 2.6, 2.77); arrow(ax, 5.2, 3.13, 4.2, 2.77)
box(ax, 0.6, 0.25, 5.6, 1.35, "Pairwise binary cross-entropy\nTest: argmax similarity over support classes\nNo class spread · no rejection (closed set)", fc="#FEF3C7", tc=NAVY, ec=AMBER, fs=9)
arrow(ax, 3.4, 1.93, 3.4, 1.62)
box(ax, 7.6, 4.4, 5.8, 0.75, "Support set (N × K)  +  query windows", fc="white", tc=NAVY, ec=NAVY, fs=9.5)
box(ax, 7.6, 3.15, 5.8, 0.9, "WDCNN + SE channel attention → embedding 100", fc=BLUE, fs=9.5)
arrow(ax, 10.5, 4.38, 10.5, 4.07)
box(ax, 7.6, 1.95, 5.8, 0.8, "Adaptive prototypes + shrinkage variance → d̃_c(x)", fc=CYAN, fs=9.5)
arrow(ax, 10.5, 3.13, 10.5, 2.77)
box(ax, 7.6, 0.25, 5.8, 1.35, "Episodic CE + entropy + margin loss (pseudo-unknowns)\nTest: argmax p(c|x)  if s(x) ≤ τ,  else UNKNOWN\nClass spread modelled · calibrated rejection (open set)", fc="#CFFAFE", tc=NAVY, ec=CYAN, fs=9)
arrow(ax, 10.5, 1.93, 10.5, 1.62)
ax.plot([7.0, 7.0], [0.2, 5.9], color="#CBD5E1", lw=1.5)
save(fig, "fig_base_vs_proposed.png")

# ---------------------------------------------------------------- 7 base architecture
fig, ax = canvas(13, 4.2, 13, 4.2)
ax.text(6.5, 3.95, "Base paper architecture (re-implemented from official code)", ha="center", fontsize=13, fontweight="bold", color=NAVY)
layers = ["Input\n2048 × 1", "Conv k64 s16\n16 ch · pool", "Conv k3\n32 ch · pool", "Conv k2\n64 ch · pool", "Conv k3\n64 ch · pool", "Conv k3 (valid)\n64 ch · pool", "FC 100\nsigmoid"]
for row, y in enumerate([2.6, 0.9]):
    for i, l in enumerate(layers):
        box(ax, 0.2 + i * 1.33, y, 1.2, 0.95, l, fc=NAVY if i else "white", tc="white" if i else NAVY, ec=NAVY, fs=8)
        if i:
            arrow(ax, 0.2 + i * 1.33 - 0.12, y + 0.47, 0.2 + i * 1.33 - 0.01, y + 0.47)
    ax.text(0.2, y + 1.07, ["Twin branch A", "Twin branch B (shared weights)"][row], fontsize=8.5, color=GREY)
box(ax, 9.75, 1.55, 1.45, 1.35, "|z_A − z_B|\nL1 distance\nDropout 0.5", fc=AMBER, fs=8.5)
box(ax, 11.45, 1.55, 1.4, 1.35, "FC 1\nsigmoid\nP(same)", fc=CYAN, fs=9)
arrow(ax, 9.4, 3.05, 9.75, 2.6); arrow(ax, 9.4, 1.35, 9.75, 1.8); arrow(ax, 11.2, 2.22, 11.43, 2.22)
save(fig, "fig_base_architecture.png")

# ---------------------------------------------------------------- 8 leakage comparison
fig, ax = canvas(13, 3.6, 13, 3.6)
ax.text(6.5, 3.35, "Why the evaluation protocol matters (window-level vs. region-level splitting)", ha="center", fontsize=12.5, fontweight="bold", color=NAVY)
ax.text(0.2, 2.65, "Base paper", fontsize=10, fontweight="bold", color=AMBER)
for i in range(40):
    ax.add_patch(plt.Rectangle((1.6 + i * 0.11, 2.45 + (i % 4) * 0.08), 1.0, 0.07, fc=AMBER if i % 5 else RED, alpha=.6))
ax.text(4.0, 2.05, "660 train windows, stride 80 (96 % overlap); validation = random subset of the SAME pool", fontsize=8.5, color=NAVY, ha="center")
for i in range(6):
    ax.add_patch(plt.Rectangle((7.4 + i * 0.95, 2.5), 0.9, 0.25, fc=NAVY, alpha=.7))
ax.text(10.2, 2.05, "25 test windows / class / load (same classes as training)", fontsize=8.5, color=NAVY, ha="center")
ax.text(0.2, 1.05, "This work", fontsize=10, fontweight="bold", color=CYAN)
for i in range(6):
    ax.add_patch(plt.Rectangle((1.6 + i * 0.95, 0.85), 0.9, 0.3, fc=BLUE, alpha=.7))
for i in range(6):
    ax.add_patch(plt.Rectangle((7.4 + i * 0.95, 0.85), 0.9, 0.3, fc=CYAN, alpha=.8))
ax.text(4.3, 0.45, "Region A: support only (unseen test classes)", fontsize=8.5, color=BLUE, ha="center")
ax.text(10.2, 0.45, "Region B: queries only — zero shared samples", fontsize=8.5, color="#0E7490", ha="center")
ax.plot([7.1, 7.1], [0.75, 1.25], color=RED, lw=2)
save(fig, "fig_protocol_comparison.png")

# ---------------------------------------------------------------- 9 real signals
try:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
    from osapnet.data.cwru import CLASS_FILES, load_signal
    import yaml
    root = yaml.safe_load(open("configs/default.yaml"))["paths"]["data_root"]
    show = ["Normal", "IR007", "B007", "OR007@6", "IR028", "FE-IR014"]
    fig, axs = plt.subplots(3, 2, figsize=(12, 6.4), sharex=True)
    for a_, c in zip(axs.ravel(), show):
        ch, nums = CLASS_FILES[c]
        s = load_signal(root, nums[0], ch)[20000:22048]
        a_.plot(np.arange(2048) / 12.0, s, color=NAVY if c != "IR028" else RED, lw=.6)
        a_.set_title(f"{c}  (file {nums[0]}, load 0 HP, {ch} sensor)", fontsize=9.5, color=NAVY)
        a_.grid(alpha=.25); a_.set_ylabel("acc. (g)", fontsize=8)
    for a_ in axs[-1]:
        a_.set_xlabel("time (ms) — one 2048-sample window")
    fig.suptitle("Real CWRU vibration windows used as model input", fontsize=13, fontweight="bold", color=NAVY)
    fig.tight_layout(); save(fig, "fig_signals.png")
except Exception as e:  # dataset not present
    print("fig_signals skipped:", e)
print("figures written to", OUT)
