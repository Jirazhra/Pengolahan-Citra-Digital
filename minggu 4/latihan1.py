# ============================================
# HISTOGRAM EQUALIZATION MANUAL
# TANPA MENGGUNAKAN OPENCV UNTUK EQUALIZATION
# ============================================

import cv2
import numpy as np
import matplotlib.pyplot as plt

# ============================================
# FUNGSI HISTOGRAM EQUALIZATION MANUAL
# ============================================

def manual_histogram_equalization(image):
    """
    Manual implementation of histogram equalization
    """

    # 1. Hitung histogram
    hist = np.zeros(256)

    for pixel in image.flatten():
        hist[pixel] += 1

    # 2. Hitung cumulative histogram (CDF)
    cdf = np.cumsum(hist)

    # 3. Normalisasi CDF untuk membuat transformation function
    cdf_min = np.min(cdf[cdf > 0])
    total_pixels = image.size

    transform = ((cdf - cdf_min) / (total_pixels - cdf_min)) * 255
    transform = np.round(transform).astype(np.uint8)

    # 4. Terapkan transformasi ke citra
    equalized_image = transform[image]

    # 5. Return hasil
    return equalized_image, hist, cdf, transform


# ============================================
# LOAD GAMBAR
# ============================================

# Ganti dengan gambar kamu
image = cv2.imread("gambar1.jpg", cv2.IMREAD_GRAYSCALE)

if image is None:
    print("Gambar tidak ditemukan!")
    exit()

# ============================================
# PROSES HISTOGRAM EQUALIZATION
# ============================================

equalized_image, hist, cdf, transform = manual_histogram_equalization(image)

# Histogram hasil
hist_equalized, _ = np.histogram(equalized_image.flatten(), 256, [0,256])

# ============================================
# VISUALISASI HASIL
# ============================================

fig, axes = plt.subplots(2,3, figsize=(15,8))

# Gambar asli
axes[0,0].imshow(image, cmap='gray')
axes[0,0].set_title("Original Image")
axes[0,0].axis("off")

# Histogram asli
axes[0,1].bar(range(256), hist, color='gray')
axes[0,1].set_title("Original Histogram")
axes[0,1].set_xlabel("Intensity")
axes[0,1].set_ylabel("Frequency")

# CDF
axes[0,2].plot(cdf, color='blue')
axes[0,2].set_title("Cumulative Distribution Function (CDF)")
axes[0,2].set_xlabel("Intensity")
axes[0,2].set_ylabel("Cumulative Frequency")

# Gambar hasil equalization
axes[1,0].imshow(equalized_image, cmap='gray')
axes[1,0].set_title("Equalized Image")
axes[1,0].axis("off")

# Histogram hasil
axes[1,1].bar(range(256), hist_equalized, color='black')
axes[1,1].set_title("Equalized Histogram")
axes[1,1].set_xlabel("Intensity")
axes[1,1].set_ylabel("Frequency")

# Fungsi transformasi
axes[1,2].plot(transform, color='red')
axes[1,2].set_title("Transformation Function")
axes[1,2].set_xlabel("Input Intensity")
axes[1,2].set_ylabel("Output Intensity")

plt.suptitle("Manual Histogram Equalization", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()