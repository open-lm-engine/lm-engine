# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import matplotlib.pyplot as plt
import seaborn as sns


raw_texts = [
    """RNN;			0.0000	0.0000	0.0000	0.0000	0.0000
GRU;			71.4000	26.6000	13.0000	7.6000	1.8000
Mamba-2;			100.0000	100.0000	100.0000	97.8000	10.8000
GDN;			100.0000	100.0000	100.0000	100.0000	88.8000
GDN[-1,1];			100.0000	100.0000	100.0000	99.4000	99.6000
M$^2$RNN;			100.0000	100.0000	100.0000	98.4000	91.8000
Transformer++;			100.0000	100.0000	100.0000	0.0000	0.0000
Hybrid Mamba-2;			100.0000	100.0000	100.0000	0.0000	0.0000
Hybrid GDN;			100.0000	100.0000	100.0000	100.0000	100.0000
Hybrid M$^2$RNN;			100.0000	100.0000	100.0000	96.2000	92.0000
Hybrid Mamba-2 M$^2$RNN-1;			100.0000	100.0000	100.0000	6.2000	0.0000
Hybrid Mamba-2 M$^2$RNN-4;			100.0000	100.0000	100.0000	21.6000	1.0000
Hybrid GDN M$^2$RNN-1;			100.0000	100.0000	100.0000	100.0000	100.0000
Hybrid GDN M$^2$RNN-4;			100.0000	100.0000	100.0000	92.8000	55.6000""",
    """RNN;			3.0000	1.0000	1.2000	1.2000	0.6000
GRU;			49.8000	13.6000	9.8000	2.4000	3.0000
Mamba-2;			100.0000	99.2000	80.4000	7.6000	8.0000
GDN;			100.0000	99.8000	73.2000	2.4000	3.0000
GDN[-1,1];			100.0000	98.0000	75.2000	9.6000	10.8000
M$^2$RNN;			100.0000	54.4000	49.0000	4.0000	4.6000
Transformer++;			100.0000	100.0000	100.0000	0.0000	0.0000
Hybrid Mamba-2;			100.0000	100.0000	100.0000	0.0000	0.0000
Hybrid GDN;			100.0000	100.0000	100.0000	86.8000	0.0000
Hybrid M$^2$RNN;			99.8000	100.0000	100.0000	50.4000	4.6000
Hybrid Mamba-2 M$^2$RNN-1;			100.0000	100.0000	100.0000	0.0000	0.0000
Hybrid Mamba-2 M$^2$RNN-4;			100.0000	100.0000	100.0000	41.0000	0.6000
Hybrid GDN M$^2$RNN-1;			100.0000	99.8000	99.8000	54.6000	3.0000
Hybrid GDN M$^2$RNN-4;			100.0000	100.0000	100.0000	98.8000	64.0000""",
    """RNN;			0.0000	0.0000	0.0000	0.0000	0.0000
GRU;			2.8000	1.4000	0.0000	0.0000	0.2000
Mamba-2;			54.6000	67.0000	31.6000	7.4000	2.2000
GDN;			88.2000	69.2000	42.4000	6.6000	2.0000
GDN[-1,1];			99.2000	96.8000	51.8000	21.4000	8.4000
M$^2$RNN;			95.4000	66.6000	16.0000	8.0000	2.6000
Transformer++;			93.4000	86.6000	80.0000	0.0000	0.0000
Hybrid Mamba-2;			99.6000	99.2000	96.2000	0.0000	0.0000
Hybrid GDN;			78.2000	98.0000	85.4000	79.8000	0.0000
Hybrid M$^2$RNN;			95.6000	90.0000	87.4000	62.2000	0.4000
Hybrid Mamba-2 M$^2$RNN-1;			98.8000	94.2000	95.6000	0.0000	0.0000
Hybrid Mamba-2 M$^2$RNN-4;			89.6000	98.2000	92.2000	57.2000	7.2000
Hybrid GDN M$^2$RNN-1;			97.2000	78.0000	74.0000	81.2000	2.4000
Hybrid GDN M$^2$RNN-4;			99.8000	94.8000	92.6000	71.8000	27.4000""",
]

sequence_lengths = [1024, 2048, 4096, 8192, 16384]


def plot(raw_text, ax, title):
    results = {}
    for line in raw_text.strip().split("\n"):
        model, line = line.split(";")
        parts = line.split()
        scores = list(map(float, parts))
        results[model] = scores

    for model, scores in results.items():
        # if model in [
        #     "RNN",
        #     "GRU",
        #     "Hybrid Mamba-2",
        #     "Mamba-2",
        #     "Hybrid Mamba-2 M$^2$RNN-1",
        #     "Hybrid Mamba-2 M$^2$RNN-4",
        #     # "GDN[-1,1]",
        # ]:
        if model in ["RNN", "GRU", "Hybrid GDN", "GDN", "GDN[-1,1]", "Hybrid GDN M$^2$RNN-1", "Hybrid GDN M$^2$RNN-4"]:
            continue

        ax.plot(sequence_lengths, scores, marker="o", linewidth=2, label=model)

    ax.set_xticks(sequence_lengths)
    ax.set_xlabel("Sequence Length")
    ax.set_title(title)
    ax.set_xticks(sequence_lengths)
    ax.set_xticklabels(sequence_lengths, rotation=45, ha="right")
    ax.grid(True)

    ax.axvline(
        x=4096,
        color="black",
        linestyle="--",
        linewidth=1.5,
        alpha=0.8,
    )


sns.set(style="whitegrid")

fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

titles = ["S-NIAH-1", "S-NIAH-2", "S-NIAH-3"]

for ax, raw_text, title in zip(axes, raw_texts, titles):
    plot(raw_text, ax, title)

axes[0].set_ylabel("Accuracy (%)")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=14)  # adjust depending on #models


plt.tight_layout(rect=[0, 0.1, 1, 1])  # leave space for legend
plt.savefig("niah.svg", format="svg")
