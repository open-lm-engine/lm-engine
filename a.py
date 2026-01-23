import matplotlib.pyplot as plt
import seaborn as sns


raw_texts = [
    """softmax-attention			100.0000	100.0000	100.0000	0.0000	0.0000
mamba2			100.0000	91.2000	58.6000	18.6000	3.2000
gated-deltanet			100.0000	99.8000	70.2000	33.2000	15.2000
RSA			99.6000	80.0000	29.4000	8.2000	1.8000
hybrid-mamba2			100.0000	100.0000	100.0000	30.6000	0.6000
hybrid-gated-deltanet			100.0000	99.8000	70.2000	33.2000	15.2000
hybrid-RSA			100.0000	100.0000	100.0000	100.0000	100.0000
hybrid-gated-deltanet-RSA-1L			100.0000	100.0000	97.6000	76.4000	69.6000
hybrid-gated-deltanet-RSA-3L			100.0000	93.8000	48.6000	18.6000	7.0000""",
    """softmax-attention			100.0000	100.0000	89.8000	0.0000	0.0000
mamba2			95.4000	14.4000	9.2000	2.4000	3.0000
gated-deltanet			100.0000	56.8000	40.4000	4.0000	5.0000
RSA			99.6000	48.2000	35.8000	2.4000	7.0000
hybrid-mamba2			100.0000	100.0000	100.0000	0.6000	0.0000
hybrid-gated-deltanet			100.0000	99.6000	85.4000	10.6000	0.8000
hybrid-RSA			100.0000	100.0000	100.0000	85.2000	23.4000
hybrid-gated-deltanet-RSA-1L			100.0000	100.0000	100.0000	82.6000	13.8000
hybrid-gated-deltanet-RSA-3L			100.0000	100.0000	100.0000	76.8000	27.6000""",
    """softmax-attention			98.8000	94.2000	56.4000	0.0000	0.0000
mamba2			42.0000	13.8000	4.4000	1.8000	2.4000
gated-deltanet			95.8000	86.2000	43.4000	14.6000	3.6000
RSA			67.2000	35.4000	17.4000	3.8000	2.0000
hybrid-mamba2			98.0000	97.0000	96.8000	4.6000	0.0000
hybrid-gated-deltanet			94.2000	83.8000	72.2000	25.6000	0.0000
hybrid-RSA			97.0000	99.8000	95.0000	82.8000	0.0000
hybrid-gated-deltanet-RSA-1L			98.6000	99.4000	98.0000	65.8000	13.8000
hybrid-gated-deltanet-RSA-3L			98.6000	96.6000	91.4000	72.6000	11.6000""",
]

sequence_lengths = [1024, 2048, 4096, 8192, 16384]


def plot(raw_text, ax, title):
    results = {}
    for line in raw_text.strip().split("\n"):
        parts = line.split()
        model = parts[0]
        scores = list(map(float, parts[1:]))
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

titles = ["NIAH-1", "NIAH-2", "NIAH-3"]

for ax, raw_text, title in zip(axes, raw_texts, titles):
    plot(raw_text, ax, title)

axes[0].set_ylabel("Accuracy (%)")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)  # adjust depending on #models


plt.tight_layout(rect=[0, 0.1, 1, 1])  # leave space for legend
# plt.show()
plt.savefig("a.svg", format="svg")
