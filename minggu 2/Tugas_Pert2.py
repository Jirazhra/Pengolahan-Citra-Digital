# ==========================================================
# PROYEK MINI
# Implementasi dan Analisis Konversi Model Warna
# ==========================================================

import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.cluster import KMeans
import os

# ==========================================================
# 1. LOAD 3 CITRA (TERANG, NORMAL, REDUP)
# ==========================================================

image_paths = [
    "gambar1.jpg",
    "gambar2.jpg",
    "gambar3.jpg"
]

images = []
for path in image_paths:
    if not os.path.exists(path):
        print(f"File {path} tidak ditemukan!")
        exit()
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    images.append(img)

print("Semua citra berhasil dimuat.\n")

# ==========================================================
# 2. FUNGSI KUANTISASI
# ==========================================================

def uniform_quantization(image, levels=16):
    step = 256 // levels
    return (image // step) * step

def non_uniform_quantization(image, k=16):
    shape = image.shape
    pixels = image.reshape((-1, 1))
    
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(pixels)
    
    new_pixels = kmeans.cluster_centers_[kmeans.labels_]
    new_image = new_pixels.reshape(shape)
    
    return new_image.astype(np.uint8)

# ==========================================================
# 3. ANALISIS SETIAP CITRA
# ==========================================================

for idx, img in enumerate(images):

    print(f"\n==============================")
    print(f"Analisis Citra ke-{idx+1}")
    print(f"==============================")

    # -----------------------------------
    # KONVERSI MODEL WARNA
    # -----------------------------------
    start_time = time.time()
    
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    
    conversion_time = time.time() - start_time
    
    print("Waktu konversi warna: %.5f detik" % conversion_time)

    # -----------------------------------
    # KUANTISASI UNIFORM
    # -----------------------------------
    start_time = time.time()
    gray_uniform = uniform_quantization(gray)
    uniform_time = time.time() - start_time
    
    print("Waktu kuantisasi uniform: %.5f detik" % uniform_time)

    # -----------------------------------
    # KUANTISASI NON-UNIFORM
    # -----------------------------------
    start_time = time.time()
    gray_nonuniform = non_uniform_quantization(gray)
    nonuniform_time = time.time() - start_time
    
    print("Waktu kuantisasi non-uniform (KMeans): %.5f detik" % nonuniform_time)

    # -----------------------------------
    # HITUNG UKURAN MEMORI
    # -----------------------------------
    original_memory = img.nbytes
    uniform_memory = gray_uniform.nbytes
    nonuniform_memory = gray_nonuniform.nbytes

    print("Ukuran memori RGB asli:", original_memory, "bytes")
    print("Ukuran memori grayscale:", gray.nbytes, "bytes")
    print("Ukuran memori setelah kuantisasi:", uniform_memory, "bytes")

    compression_ratio = original_memory / uniform_memory
    print("Rasio kompresi (RGB → Gray Quantized): %.2f : 1" % compression_ratio)

    # -----------------------------------
    # SEGMENTASI SEDERHANA (Threshold)
    # -----------------------------------
    _, thresh_gray = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    _, thresh_hsv = cv2.threshold(hsv[:,:,0], 90, 255, cv2.THRESH_BINARY)
    _, thresh_lab = cv2.threshold(lab[:,:,1], 128, 255, cv2.THRESH_BINARY)

    # -----------------------------------
    # VISUALISASI
    # -----------------------------------
    plt.figure(figsize=(12,8))

    plt.subplot(2,3,1)
    plt.imshow(img)
    plt.title("RGB Asli")

    plt.subplot(2,3,2)
    plt.imshow(gray, cmap='gray')
    plt.title("Grayscale")

    plt.subplot(2,3,3)
    plt.imshow(gray_uniform, cmap='gray')
    plt.title("Uniform Quantization")

    plt.subplot(2,3,4)
    plt.imshow(gray_nonuniform, cmap='gray')
    plt.title("Non-Uniform Quantization")

    plt.subplot(2,3,5)
    plt.imshow(thresh_gray, cmap='gray')
    plt.title("Segmentasi Gray")

    plt.subplot(2,3,6)
    plt.imshow(thresh_hsv, cmap='gray')
    plt.title("Segmentasi HSV (Hue)")

    plt.tight_layout()
    plt.show()

    # -----------------------------------
    # HISTOGRAM
    # -----------------------------------
    plt.figure(figsize=(10,4))
    
    plt.subplot(1,3,1)
    plt.hist(gray.ravel(), bins=256)
    plt.title("Histogram Gray")

    plt.subplot(1,3,2)
    plt.hist(gray_uniform.ravel(), bins=16)
    plt.title("Histogram Uniform")

    plt.subplot(1,3,3)
    plt.hist(gray_nonuniform.ravel(), bins=16)
    plt.title("Histogram Non-Uniform")

    plt.tight_layout()
    plt.show()

print("\nAnalisis selesai.")