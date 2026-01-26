import matplotlib.pyplot as plt
import seaborn as sns


raw_texts = [
    """Transformer++;100.0000	100.0000	100.0000	0.0000	0.0000
Mamba2;100.0000	91.2000	58.6000	18.6000	3.2000
Gated DeltaNet;100.0000	100.0000	100.0000	100.0000	100.0000
Gated DeltaNet [-1,1];100.0000	100.0000	100.0000	84.2000	35.8000
M$^2$RNN;99.6000	80.0000	29.4000	8.2000	1.8000
Hybrid Mamba2;100.0000	100.0000	100.0000	30.6000	0.6000
Hybrid Gated DeltaNet;100.0000	99.8000	70.2000	33.2000	15.2000
Hybrid M$^2$RNN;100.0000	100.0000	100.0000	100.0000	100.0000
Hybrid Gated DeltaNet + M$^2$RNN-1;100.0000	100.0000	97.6000	76.4000	69.6000
Hybrid Gated DeltaNet + M$^2$RNN-3;100.0000	93.8000	48.6000	18.6000	7.0000""",
    """Transformer++;100.0000	100.0000	89.8000	0.0000	0.0000
Mamba2;95.4000	14.4000	9.2000	2.4000	3.0000
Gated DeltaNet;100.0000	56.8000	40.4000	4.0000	5.0000
Gated DeltaNet [-1,1];100.0000	91.0000	37.8000	6.0000	9.8000
M$^2$RNN;99.6000	48.2000	35.8000	2.4000	7.0000
Hybrid Mamba2;100.0000	100.0000	100.0000	0.6000	0.0000
Hybrid Gated DeltaNet;100.0000	99.6000	85.4000	10.6000	0.8000
Hybrid M$^2$RNN;100.0000	100.0000	100.0000	85.2000	23.4000
Hybrid Gated DeltaNet + M$^2$RNN-1;100.0000	100.0000	100.0000	82.6000	13.8000
Hybrid Gated DeltaNet + M$^2$RNN-3;100.0000	100.0000	100.0000	76.8000	27.6000""",
    """Transformer++;98.8000	94.2000	56.4000	0.0000	0.0000
Mamba2;42.0000	13.8000	4.4000	1.8000	2.4000
Gated DeltaNet;95.8000	86.2000	43.4000	14.6000	3.6000
Gated DeltaNet [-1,1];95.8000	75.8000	36.2000	15.6000	7.8000
M$^2$RNN;67.2000	35.4000	17.4000	3.8000	2.0000
Hybrid Mamba2;98.0000	97.0000	96.8000	4.6000	0.0000
Hybrid Gated DeltaNet;94.2000	83.8000	72.2000	25.6000	0.0000
Hybrid M$^2$RNN;97.0000	99.8000	95.0000	82.8000	0.0000
Hybrid Gated DeltaNet + M$^2$RNN-1;98.6000	99.4000	98.0000	65.8000	13.8000
Hybrid Gated DeltaNet + M$^2$RNN-3;98.6000	96.6000	91.4000	72.6000	11.6000""",
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
        ax.plot(sequence_lengths, scores, marker="o", linewidth=2, label=model)

    ax.set_xticks(sequence_lengths)
    ax.set_xlabel("Sequence Length")
    ax.set_title(title)
    ax.set_xticks(sequence_lengths)
    ax.set_xticklabels(sequence_lengths, rotation=45, ha="right")
    ax.grid(True)


sns.set(style="whitegrid")

fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

titles = ["S-NIAH-1", "S-NIAH-2", "S-NIAH-3"]

for ax, raw_text, title in zip(axes, raw_texts, titles):
    plot(raw_text, ax, title)

axes[0].set_ylabel("Accuracy (%)")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False, fontsize=14)  # adjust depending on #models


plt.tight_layout(rect=[0, 0.1, 1, 1])  # leave space for legend
plt.savefig("niah.svg", format="svg")
