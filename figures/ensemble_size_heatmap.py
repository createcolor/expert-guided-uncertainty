import matplotlib.pyplot as plt
import numpy as np

import matplotlib
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap

base_ensemble_size = [2, 4, 6, 8, 10][::-1]
cae_ensemble_size = [2, 4, 6, 8, 10]

# aac_by_size = np.array(
#     [[130.11, 124.26, 120.59, 121.88, 120.92],
#      [120.34, 118.59, 117.57, 115.60, 113.56],
#      [119.83, 116.42, 116.44, 116.43, 114.23],
#      [119.28, 117.23, 115.92, 114.96, 113.65],
#      [118.00, 119.93, 116.91, 115.00, 112.60]][::-1]
# )

aac_by_size = np.array(
    [[111.57, 108.26, 106.21, 106.93, 105.05],
     [106.08, 104.07, 102.32, 103.08, 101.43],
     [106.17, 104.94, 103.55, 103.61, 101.88],
     [104.91, 103.06, 102.07, 102.24, 100.69],
     [104.86, 103.14, 102.31, 102.25, 100.85]][::-1]
)

# Define the custom colormap
colors = ["#00999E", "#FFFFFF", "#E0695E"]
custom_cmap = LinearSegmentedColormap.from_list("RedWhiteTeal", colors, N=256)

fig, ax = plt.subplots()
im = ax.imshow(aac_by_size, cmap=custom_cmap)

# Create a colorbar
cbar = ax.figure.colorbar(im, ax=ax, cmap=custom_cmap)
cbar.ax.set_ylabel(r"Area above curve, $\times 10^3$", rotation=-90, va="bottom")

# Show all ticks and label them with the respective list entries
ax.set_xticks(range(len(cae_ensemble_size)), labels=cae_ensemble_size)
ax.set_yticks(range(len(base_ensemble_size)), labels=base_ensemble_size)

# Loop over data dimensions and create text annotations.
for i in range(len(base_ensemble_size)):
    for j in range(len(cae_ensemble_size)):
        text = ax.text(j, i, aac_by_size[i, j],
                       ha="center", va="center", color="k")

# ax.set_title("Uncertainty estimation quality by ensemble sizes")
ax.set_xlabel("CAE size")
ax.set_ylabel("Base ensemble size")
fig.tight_layout()
plt.savefig('heatmap.pdf', format='pdf')