import matplotlib.pyplot as plt
from skimage import filters
from skimage.data import checkerboard, horse

image1 = checkerboard()
image2 = horse()

edge_prewitt_h1 = filters.prewitt_h(image1)
edge_prewitt_v1 = filters.prewitt_v(image1)
edge_prewitt_h2 = filters.prewitt_h(image2)
edge_prewitt_v2 = filters.prewitt_v(image2)

fig, axes = plt.subplots(ncols=3, nrows=2, figsize=(10, 10))

# create sublots
axes[0][0].imshow(image1, cmap="gray")
axes[0][0].set_title("Original Image")

axes[0][1].imshow(edge_prewitt_h1, cmap="gray")
axes[0][1].set_title("Horizontal Edge Detection")

axes[0][2].imshow(edge_prewitt_v1, cmap="gray")
axes[0][2].set_title("Vertical Edge Detection")

axes[1][0].imshow(image2, cmap="gray")
axes[1][0].set_title("Original Image")

axes[1][1].imshow(edge_prewitt_h2, cmap="gray")
axes[1][1].set_title("Horizontal Edge Detection")

axes[1][2].imshow(edge_prewitt_v2, cmap="gray")
axes[1][2].set_title("Vertical Edge Detection")

axes = axes.flatten()
for ax in axes:
    ax.set_axis_off()

plt.tight_layout()
plt.show()
