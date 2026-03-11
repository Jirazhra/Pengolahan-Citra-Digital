# ==========================================
# PIPELINE ENHANCEMENT CITRA
# Underexposed, Overexposed, Uneven Lighting
# ==========================================

import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import shannon_entropy

# ================================
# Membaca 3 citra
# ================================
under = cv2.imread("pict1.jpg", 0)
over = cv2.imread("pict2.jpg", 0)
uneven = cv2.imread("pict3.jpg", 0)

images = {
    "Underexposed": under,
    "Overexposed": over,
    "Uneven Lighting": uneven
}

# ================================
# POINT PROCESSING
# ================================

# Negative Transformation
def negative_transform(img):
    return 255 - img

# Log Transformation
def log_transform(img):
    c = 255 / np.log(1 + np.max(img))
    log = c * np.log(1 + img)
    return np.array(log, dtype=np.uint8)

# Gamma Correction
def gamma_transform(img, gamma):
    table = np.array([(i/255.0)**gamma * 255
                      for i in np.arange(256)]).astype("uint8")
    return cv2.LUT(img, table)

# ================================
# HISTOGRAM BASED METHODS
# ================================

# Contrast Stretching Manual
def contrast_stretch_manual(img):
    r_min = np.min(img)
    r_max = np.max(img)

    stretched = (img - r_min) * (255/(r_max - r_min))
    stretched = np.clip(stretched, 0, 255)

    return stretched.astype(np.uint8)

# Contrast Stretching Automatic (percentile)
def contrast_stretch_auto(img):
    p2, p98 = np.percentile(img, (2,98))
    img_rescale = cv2.normalize(img,None,0,255,cv2.NORM_MINMAX)
    return img_rescale

# Histogram Equalization
def histogram_equalization(img):
    return cv2.equalizeHist(img)

# CLAHE
def clahe_enhancement(img):
    clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
    return clahe.apply(img)

# ================================
# METRIK EVALUASI
# ================================

# Contrast Ratio
def contrast_ratio(img):
    Imax = np.max(img)
    Imin = np.min(img)

    if (Imax + Imin) == 0:
        return 0

    return (Imax - Imin) / (Imax + Imin)

# Entropy
def entropy(img):
    return shannon_entropy(img)

# ================================
# HISTOGRAM DISPLAY
# ================================

def plot_histogram(img,title):

    plt.figure()
    plt.hist(img.ravel(),256,[0,256])
    plt.title(title)
    plt.xlabel("Intensity")
    plt.ylabel("Pixel Count")
    plt.show()

# ================================
# PIPELINE PROCESSING
# ================================

for name, img in images.items():

    print("\n================================")
    print("Processing :", name)
    print("================================")

    # Point Processing
    negative = negative_transform(img)
    log_img = log_transform(img)

    gamma05 = gamma_transform(img,0.5)
    gamma1 = gamma_transform(img,1.0)
    gamma2 = gamma_transform(img,2.0)

    # Histogram Methods
    stretch_manual = contrast_stretch_manual(img)
    stretch_auto = contrast_stretch_auto(img)

    hist_eq = histogram_equalization(img)
    clahe = clahe_enhancement(img)

    # ======================
    # METRIK
    # ======================

    methods = {
        "Original": img,
        "Negative": negative,
        "Log": log_img,
        "Gamma 0.5": gamma05,
        "Gamma 1": gamma1,
        "Gamma 2": gamma2,
        "Stretch Manual": stretch_manual,
        "Stretch Auto": stretch_auto,
        "Hist Eq": hist_eq,
        "CLAHE": clahe
    }

    print("\nMETRIC EVALUATION")

    for m_name, m_img in methods.items():

        c = contrast_ratio(m_img)
        e = entropy(m_img)

        print(f"{m_name:15} | Contrast: {c:.4f} | Entropy: {e:.4f}")

    # ======================
    # VISUALIZATION
    # ======================

    titles = list(methods.keys())
    imgs = list(methods.values())

    plt.figure(figsize=(15,8))

    for i in range(len(imgs)):

        plt.subplot(3,4,i+1)
        plt.imshow(imgs[i],cmap='gray')
        plt.title(titles[i])
        plt.axis("off")

    plt.suptitle(name)
    plt.show()

    # ======================
    # HISTOGRAM BEFORE AFTER
    # ======================

    plot_histogram(img, name + " - Original Histogram")
    plot_histogram(hist_eq, name + " - Histogram Equalization")
    plot_histogram(clahe, name + " - CLAHE Histogram")
