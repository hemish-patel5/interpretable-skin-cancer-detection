#First install: 
# pip install scikit-image numpy pandas scikit-learn pillow pywavelets

#Second install: 
# pip install scikit-eLCS


import os
import numpy as np
import pandas as pd
from skimage import io, color, feature
from skimage.transform import resize
import pywt

# --- 1. SETTINGS ---
IMAGE_DIR = 'images'   # Must have two subfolders: benign/ and malignant/
OUTPUT_FILE = 'capstone_features.csv'
IMG_SIZE = (128, 128)  # Resize all images to same size

# --- 2. EXTRACTION LOGIC ---
def get_features(img_path):
    # Load and resize image
    img = io.imread(img_path)
    img_resized = resize(img, IMG_SIZE, anti_aliasing=True)

    # Convert to grayscale
    if len(img_resized.shape) == 3:
        gray = color.rgb2gray(img_resized)
    else:
        gray = img_resized

    # LBP (Texture) - Slide 13
    lbp = feature.local_binary_pattern(gray, P=8, R=1, method='uniform')
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=10, density=True)

    # HOG (Shape/Edges) - Slide 15
    hog_feats = feature.hog(gray, orientations=9, pixels_per_cell=(16, 16),
                            cells_per_block=(2, 2))
    hog_sample = hog_feats[:10]

    # DWT (Frequency) - Slide 15
    coeffs = pywt.dwt2(gray, 'haar')
    LL, (LH, HL, HH) = coeffs
    dwt_sample = [np.mean(LL), np.var(LL), np.mean(LH), np.var(LH)]

    return list(lbp_hist) + list(hog_sample) + dwt_sample

# --- 3. RUNNING THE LOOP ---
# Expects:  images/benign/   and   images/malignant/
data = []
labels_found = []

for label in ['benign', 'malignant']:
    folder = os.path.join(IMAGE_DIR, label)
    if not os.path.exists(folder):
        print(f"WARNING: folder not found — {folder}")
        continue

    print(f"\nProcessing {label} images...")
    for filename in os.listdir(folder):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            print(f"  Extracting: {filename}")
            try:
                feats = get_features(os.path.join(folder, filename))
                data.append([filename] + feats + [label])
                labels_found.append(label)
            except Exception as e:
                print(f"  Error processing {filename}: {e}")

# --- 4. SAVE TO CSV ---
columns = (
    ['image_id'] +
    [f'LBP_{i}' for i in range(10)] +
    [f'HOG_{i}' for i in range(10)] +
    [f'DWT_{i}' for i in range(4)] +
    ['label']
)

df = pd.DataFrame(data, columns=columns)
df.to_csv(OUTPUT_FILE, index=False)

print(f"\nDone! '{OUTPUT_FILE}' created with {len(df)} images.")
print(f"Benign: {labels_found.count('benign')}  |  Malignant: {labels_found.count('malignant')}")
print("\nFirst few rows:")
print(df.head())