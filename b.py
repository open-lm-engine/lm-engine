# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import matplotlib.pyplot as plt


# Data
lengths = [1024, 2048, 4096, 8192, 16384]
plt.rcParams.update({"font.size": 16})

data = {
    "Mamba2": [25.3321, 25.8392, 24.9781, 24.8765, 23.2763],
    "GDN": [23.8505, 23.6300, 23.5325, 23.1416, 18.5897],
    "M$^2$RNN": [13.7742, 13.1887, 10.2682, 7.4209, 4.5869],
    "Transformer++": [30.4475, 28.6539, 25.4837, 20.9262, 14.1406],
    "Hybrid GDN": [23.9654, 24.3846, 23.6099, 22.8044, 18.3646],
    "Hybrid GDN + M$^2$RNN-1": [23.9516, 23.6469, 22.2693, 22.0600, 17.2564],
}

# Plot
plt.figure(figsize=(9, 6))
for model, values in data.items():
    plt.plot(lengths, values, marker="o", label=model)

plt.xlabel("Context Length")
plt.ylabel("Training Throughput (B tokens/day)")
plt.title("Training Throughput (B tokens/day) vs Context Length")
plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False)
plt.grid(True)
plt.xticks(lengths)
# plt.xscale("log", base=2)  # optional, nice for powers of 2
plt.tight_layout()
plt.subplots_adjust(bottom=0.25)

# plt.show()
plt.savefig("niah.svg", format="svg")
