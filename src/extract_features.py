import os
import numpy as np
import pandas as pd
from skimage import io, color, feature
from skimage.transform import resize
import pywt
from pathlib import Path
import kagglehub

# --- 1. SETTINGS ---
path = kagglehub.dataset_download("fanconic/skin-cancer-malignant-vs-benign")
current_dir = Path(__file__).resolve()
root_dir = current_dir.parent.parent
IMAGE_DIR = Path(path) / "test"
IMG_SIZE = (128, 128)

# Output files — one per feature set + one combined
OUTPUT_FILES = {
    'LBP': root_dir / 'features_LBP.csv',
    'HOG': root_dir / 'features_HOG.csv',
    'DWT': root_dir / 'features_DWT.csv',
    'ALL': root_dir / 'features_ALL.csv',
}

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

    # Convert float (0-1) to integer (0-255) for LBP
    gray_int = (gray * 255).astype(np.uint8)

    # LBP (Texture)
    lbp = feature.local_binary_pattern(gray_int, P=8, R=1, method='uniform')
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=10, density=True)

    # HOG (Shape/Edges)
    hog_feats = feature.hog(gray, orientations=9, pixels_per_cell=(16, 16),
                            cells_per_block=(2, 2))
    hog_sample = hog_feats[:10]

    # DWT (Frequency)
    coeffs = pywt.dwt2(gray, 'haar')
    LL, (LH, HL, HH) = coeffs
    dwt_sample = [np.mean(LL), np.var(LL), np.mean(LH), np.var(LH)]

    return {
        'LBP': list(lbp_hist),
        'HOG': list(hog_sample),
        'DWT': list(dwt_sample),
    }

# --- 3. RUNNING THE LOOP ---
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
            try:
                feats = get_features(os.path.join(folder, filename))
                data.append({
                    'image_id': filename,
                    'label': label,
                    **{f'LBP_{i}': feats['LBP'][i] for i in range(10)},
                    **{f'HOG_{i}': feats['HOG'][i] for i in range(10)},
                    **{f'DWT_{i}': feats['DWT'][i] for i in range(4)},
                })
                labels_found.append(label)
            except Exception as e:
                print(f"  Error processing {filename}: {e}")

# --- 4. BUILD MASTER DATAFRAME ---
df_all = pd.DataFrame(data)

# --- 5. SAVE SEPARATE CSVs ---
# LBP only
lbp_cols = ['image_id'] + [f'LBP_{i}' for i in range(10)] + ['label']
df_all[lbp_cols].to_csv(OUTPUT_FILES['LBP'], index=False)

# HOG only
hog_cols = ['image_id'] + [f'HOG_{i}' for i in range(10)] + ['label']
df_all[hog_cols].to_csv(OUTPUT_FILES['HOG'], index=False)

# DWT only
dwt_cols = ['image_id'] + [f'DWT_{i}' for i in range(4)] + ['label']
df_all[dwt_cols].to_csv(OUTPUT_FILES['DWT'], index=False)

# ALL combined
df_all.to_csv(OUTPUT_FILES['ALL'], index=False)

# --- 6. SUMMARY ---
total = len(df_all)
benign = labels_found.count('benign')
malignant = labels_found.count('malignant')

print(f"\n{'='*50}")
print(f"Done! {total} images processed.")
print(f"Benign: {benign}  |  Malignant: {malignant}")
print(f"\nCSVs saved:")
for name, path in OUTPUT_FILES.items():
    print(f"  {name:4s} → {path}")
print(f"{'='*50}")
print("\nFirst few rows (ALL):")
print(df_all.head())