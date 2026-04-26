import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, feature
from skimage.transform import resize

IMAGE_PATH = 'images/malignant/1009.jpg'

IMG_SIZE = (128, 128)

# Load and resize
img = io.imread(IMAGE_PATH)
img_resized = resize(img, IMG_SIZE, anti_aliasing=True)
gray = color.rgb2gray(img_resized)

# Convert gray to uint8 to avoid the warning
gray_uint8 = (gray * 255).astype(np.uint8)

# Compute LBP
lbp = feature.local_binary_pattern(gray_uint8, P=8, R=1, method='uniform')

# Compute histogram
hist, bins = np.histogram(lbp.ravel(), bins=10, density=True)

# Plot
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
fig.suptitle('LBP Feature Extraction — Skin Lesion', fontsize=14, fontweight='bold')

# Original image
axes[0].imshow(img_resized)
axes[0].set_title('Original Skin Image')
axes[0].axis('off')

# LBP image
axes[1].imshow(lbp, cmap='gray')
axes[1].set_title('LBP Pattern Output')
axes[1].axis('off')

# LBP histogram
axes[2].bar(range(len(hist)), hist, color='steelblue', edgecolor='white')
axes[2].set_title('LBP Feature Histogram')
axes[2].set_xlabel('LBP Code')
axes[2].set_ylabel('Frequency (normalised)')

plt.tight_layout()
plt.savefig('lbp_output.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved as lbp_output.png")