import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage.metrics import structural_similarity as ssim
import cv2
from scipy.spatial import distance
from analyze import load_df

df = load_df()

def load_image(image_path):
    base = 'static/images/'
    image_path = base + image_path
    image = mpimg.imread(image_path)
    if image.shape[-1] == 4:  # If RGBA, convert to RGB
        image = image[:, :, :3]
    image = cv2.cvtColor((image * 255).astype('uint8'), cv2.COLOR_RGB2GRAY)  # Convert to grayscale
    return image

def calculate_ssim(img1, img2):
    return ssim(img1, img2, data_range=img2.max() - img2.min())

def calculate_correlation(img1, img2):
    return np.corrcoef(img1.ravel(), img2.ravel())[0, 1]

def calculate_cosine_similarity(hist1, hist2):
    return 1 - distance.cosine(hist1, hist2)

def extract_histogram(image):
    hist = np.histogram(image, bins=256, range=(0, 256))[0]
    hist = hist / np.sum(hist)  # Normalize
    return hist

# Assuming 'df' is your DataFrame and 'image' column contains paths to images
df['image_gray'] = df['image'].apply(load_image)
df['histogram'] = df['image_gray'].apply(extract_histogram)

# Get the unique methods - expecting exactly two entries
methods = df['Method'].unique()
if len(methods) != 2:
    raise ValueError("Expected exactly two methods for comparison.")

# Prepare a DataFrame to collect results
results = []

# Process each group
for name, group in df.groupby(['occupation', 'sample']):
    # Ensure two entries are present
    if group.shape[0] != 2:
        continue  # Skip groups that do not have exactly two entries
    
    # Extract the two rows
    row1, row2 = group.iloc[0], group.iloc[1]
    
    # Calculate metrics
    ssim_value = calculate_ssim(row1['image_gray'], row2['image_gray'])
    correlation = calculate_correlation(row1['image_gray'], row2['image_gray'])
    cosine_similarity = calculate_cosine_similarity(row1['histogram'], row2['histogram'])
    
    # Store results
    results.append({
        'Occupation': name[0],
        'Sample': name[1],
        'SSIM': ssim_value,
        'Correlation': correlation,
        'Cosine Similarity': cosine_similarity,
        'Method Pair': f"{row1['Method']} vs {row2['Method']}"
    })

# Convert results to DataFrame
results_df = pd.DataFrame(results)

print(results_df)