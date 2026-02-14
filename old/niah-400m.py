# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import matplotlib.pyplot as plt
import seaborn as sns


raw_texts = [
    """RNN;			0.0000	0.0000	0.0000	0.0000	0.0000
GRU;			8.6000	2.4000	0.2000	0.6000	0.0000
Mamba-2;			100.0000	99.8000	99.2000	91.2000	28.0000
GDN;			100.0000	99.8000	59.6000	26.2000	11.6000
GDN[-1,1];			100.0000	100.0000	73.8000	33.0000	15.2000
M$^2$RNN;			100.0000	97.2000	55.4000	15.6000	4.0000
Transformer++;			100.0000	100.0000	100.0000	0.0000	0.0000
Hybrid Mamba-2;			100.0000	100.0000	100.0000	0.0000	0.0000
Hybrid GDN;			100.0000	100.0000	67.0000	29.6000	12.6000
Hybrid M$^2$RNN;			100.0000	100.0000	100.0000	93.4000	67.6000
Hybrid Mamba-2 M$^2$RNN-1;			100.0000	100.0000	100.0000	0.0000	0.0000
Hybrid Mamba-2 M$^2$RNN-3;			100.0000	100.0000	100.0000	78.6000	60.8000
Hybrid GDN M$^2$RNN-1;			100.0000	100.0000	100.0000	100.0000	99.8000
Hybrid GDN M$^2$RNN-3;			100.0000	99.8000	98.2000	35.0000	12.6000""",
    """RNN;			0.0000	0.0000	0.0000	0.0000	0.0000
GRU;			4.2000	0.8000	0.0000	1.4000	0.0000
Mamba-2;			100.0000	15.6000	11.0000	2.4000	3.2000
GDN;			100.0000	18.0000	26.4000	3.2000	5.8000
GDN[-1,1];			99.0000	95.6000	51.8000	4.6000	8.0000
M$^2$RNN;			100.0000	30.6000	32.8000	2.4000	4.8000
Transformer++;			100.0000	100.0000	76.8000	0.0000	0.0000
Hybrid Mamba-2;			100.0000	100.0000	100.0000	0.0000	0.0000
Hybrid GDN;			100.0000	100.0000	100.0000	5.8000	0.0000
Hybrid M$^2$RNN;			100.0000	100.0000	100.0000	59.4000	0.0000
Hybrid Mamba-2 M$^2$RNN-1;			100.0000	100.0000	100.0000	0.0000	0.0000
Hybrid Mamba-2 M$^2$RNN-3;			100.0000	100.0000	100.0000	49.8000	2.6000
Hybrid GDN M$^2$RNN-1;			100.0000	99.8000	100.0000	52.0000	16.8000
Hybrid GDN M$^2$RNN-3;			100.0000	100.0000	100.0000	58.4000	18.4000""",
    """RNN;			0.0000	0.0000	0.0000	0.0000	0.0000
GRU;			0.0000	0.0000	0.0000	0.0000	0.0000
Mamba-2;			69.4000	53.6000	21.6000	4.6000	5.0000
GDN;			93.4000	71.4000	36.6000	17.6000	6.4000
GDN[-1,1];			96.8000	73.6000	43.6000	23.4000	10.2000
M$^2$RNN;			65.2000	27.6000	4.6000	1.8000	1.2000
Transformer++;			92.4000	93.6000	50.0000	0.0000	0.0000
Hybrid Mamba-2;			97.2000	97.8000	97.8000	0.0000	0.0000
Hybrid GDN;			91.0000	88.2000	86.2000	34.2000	0.0000
Hybrid M$^2$RNN;			99.6000	99.8000	100.0000	68.4000	0.0000
Hybrid Mamba-2 M$^2$RNN-1;			70.2000	62.4000	70.0000	0.0000	0.0000
Hybrid Mamba-2 M$^2$RNN-3;			97.2000	87.0000	94.4000	77.0000	0.8000
Hybrid GDN M$^2$RNN-1;			83.4000	94.6000	92.0000	82.0000	1.8000
Hybrid GDN M$^2$RNN-3;			99.2000	97.0000	94.2000	56.4000	19.0000""",
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
        if model in [
            "RNN",
            "GRU",
            "Hybrid Mamba-2",
            "Mamba-2",
            "Hybrid Mamba-2 M$^2$RNN-1",
            "Hybrid Mamba-2 M$^2$RNN-3",
        ]:
            # if model in ["RNN", "GRU", "Hybrid GDN", "GDN", "GDN[-1,1]", "Hybrid GDN M$^2$RNN-1", "Hybrid GDN M$^2$RNN-3"]:
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
fig.legend(handles, labels, loc="lower center", ncol=8, frameon=False, fontsize=14)  # adjust depending on #models


plt.tight_layout(rect=[0, 0.1, 1, 1])  # leave space for legend
plt.savefig("niah.svg", format="svg")
