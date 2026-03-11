# ============================================
# ADAPTIVE MEDICAL IMAGE ENHANCEMENT
# ============================================

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


# ============================================
# METRIC ANALYSIS
# ============================================

def analyze_image_statistics(image):

    hist,_ = np.histogram(image.flatten(),256,[0,256])

    entropy = stats.entropy(hist + 1e-10)

    stats_dict = {
        "mean": np.mean(image),
        "std": np.std(image),
        "min": np.min(image),
        "max": np.max(image),
        "dynamic_range": np.max(image) - np.min(image),
        "entropy": entropy
    }

    return stats_dict


# ============================================
# MEDICAL ENHANCEMENT PIPELINE
# ============================================

def medical_image_enhancement(medical_image, modality='X-ray'):
    """
    Adaptive enhancement for medical images
    """

    report = {}

    img = medical_image.copy()

    report["original"] = analyze_image_statistics(img)

    # =====================================
    # X-RAY PIPELINE
    # =====================================
    if modality == "X-ray":

        # noise reduction
        img = cv2.GaussianBlur(img,(5,5),0)

        # contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=3.0,tileGridSize=(8,8))
        img = clahe.apply(img)

        # sharpening
        kernel = np.array([[0,-1,0],
                           [-1,5,-1],
                           [0,-1,0]])
        img = cv2.filter2D(img,-1,kernel)

    # =====================================
    # MRI PIPELINE
    # =====================================
    elif modality == "MRI":

        # noise reduction
        img = cv2.medianBlur(img,5)

        # gamma correction
        gamma = 0.8
        img = np.power(img/255.0,gamma)
        img = np.uint8(img*255)

        # CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.5,tileGridSize=(8,8))
        img = clahe.apply(img)

    # =====================================
    # CT SCAN PIPELINE
    # =====================================
    elif modality == "CT":

        # normalization
        img = cv2.normalize(img,None,0,255,cv2.NORM_MINMAX)

        # CLAHE
        clahe = cv2.createCLAHE(clipLimit=4.0,tileGridSize=(8,8))
        img = clahe.apply(img)

        # sharpening
        kernel = np.array([[0,-1,0],
                           [-1,5,-1],
                           [0,-1,0]])
        img = cv2.filter2D(img,-1,kernel)

    # =====================================
    # ULTRASOUND PIPELINE
    # =====================================
    elif modality == "Ultrasound":

        # speckle noise reduction
        img = cv2.medianBlur(img,5)

        # adaptive histogram
        clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
        img = clahe.apply(img)

    else:
        print("Unknown modality")

    report["enhanced"] = analyze_image_statistics(img)

    return img, report


# ============================================
# MAIN PROGRAM
# ============================================

image = cv2.imread("gambar3.jpg",0)

if image is None:
    print("Image not found")
    exit()

# pilih modality
modality = "X-ray"

enhanced, report = medical_image_enhancement(image, modality)

# ============================================
# VISUALIZATION
# ============================================

plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.imshow(image,cmap='gray')
plt.title("Original Image")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(enhanced,cmap='gray')
plt.title("Enhanced Image")
plt.axis("off")

plt.suptitle("Medical Image Enhancement ("+modality+")")
plt.show()


# ============================================
# PRINT REPORT
# ============================================

print("\nEnhancement Report")
print("="*50)

print("\nOriginal Image Statistics")
for k,v in report["original"].items():
    print(k,":",round(v,3))

print("\nEnhanced Image Statistics")
for k,v in report["enhanced"].items():
    print(k,":",round(v,3))