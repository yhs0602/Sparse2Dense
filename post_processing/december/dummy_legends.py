# Recreate the Figure 10 entropy legend with the updated hex colors provided by the user.

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

colors = [
    ("#5E813F", "Sparse (PPO)"),
    ("#5D84D8", "Dense (PPO + PBRS)"),
    ("#A68461", "D2S"),
    ("#AE4338", "S2D"),
    ("#5EA3EF", "Dense + 1M Entropy"),
    ("#EA3323", "Dense + 2M Entropy"),
    ("#7BB972", "Dense + 3M Entropy"),
]

proxies = [Line2D([0], [0], color=c, linewidth=5.0) for c, _ in colors]
labels = [lbl for _, lbl in colors]

plt.rcParams.update(
    {
        "font.size": 22,
        "font.family": "Times New Roman",
        "font.weight": "bold",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

fig = plt.figure(figsize=(6, 4))
ax = fig.add_subplot(111)
ax.axis("off")

ax.legend(proxies, labels, loc="center", frameon=False)

fig.tight_layout()
pdf_path = "./figures/figure10_entropy_legend_updated.pdf"
png_path = "./figures/figure10_entropy_legend_updated.png"
fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
fig.savefig(png_path, dpi=300, bbox_inches="tight")

(pdf_path, png_path)
