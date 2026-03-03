# ============================================
# PRAKTIKUM 2: MODEL WARNA DAN DIGITALISASI
# ============================================

import cv2
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

print("=== PRAKTIKUM 2: MODEL WARNA DAN DIGITALISASI ===")
print("Materi: Color Models, Sampling, Quantization, Color Conversion\n")

# =============== FUNGSI BANTU ===============

def create_color_patches():
    """Membuat patch warna dasar"""
    patches = []
    colors = [
        ('Red', [0, 0, 255]),
        ('Green', [0, 255, 0]),
        ('Blue', [255, 0, 0]),
        ('Yellow', [0, 255, 255]),
        ('Magenta', [255, 0, 255]),
        ('Cyan', [255, 255, 0]),
        ('White', [255, 255, 255]),
        ('Black', [0, 0, 0])
    ]
    for name, color in colors:
        patch = np.zeros((100, 100, 3), dtype=np.uint8)
        patch[:] = color
        patches.append((name, patch))
    return patches


def analyze_color_model(image, model_name):
    """Konversi citra ke model warna tertentu"""
    if model_name == 'RGB':
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif model_name == 'HSV':
        return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    elif model_name == 'LAB':
        return cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    elif model_name == 'GRAY':
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def analyze_color_model_suitability(image, application):
    """
    Analisis kecocokan model warna berdasarkan aplikasi
    """
    result = {}

    if application == 'text_extraction':
        result['best_model'] = 'Grayscale'
        result['reason'] = (
            'Deteksi teks dan tepi hanya bergantung pada intensitas cahaya, '
            'sehingga grayscale lebih efisien dan stabil.'
        )
        result['output'] = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    elif application == 'skin_detection':
        result['best_model'] = 'HSV'
        result['reason'] = (
            'HSV memisahkan warna dan pencahayaan, '
            'sehingga lebih robust terhadap perubahan iluminasi.'
        )
        result['output'] = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    elif application == 'shadow_removal':
        result['best_model'] = 'LAB'
        result['reason'] = (
            'LAB memisahkan luminansi (L) dari warna (a,b), '
            'sehingga bayangan dapat dikurangi tanpa merusak warna.'
        )
        result['output'] = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    elif application == 'object_detection':
        result['best_model'] = 'HSV'
        result['reason'] = (
            'HSV memudahkan segmentasi objek berwarna menggunakan komponen Hue.'
        )
        result['output'] = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    else:
        result['best_model'] = 'RGB'
        result['reason'] = 'Aplikasi umum.'
        result['output'] = image

    return result


# =============== MAIN PRAKTIKUM ===============

# 1. PERBANDINGAN MODEL WARNA
print("1. PERBANDINGAN MODEL WARNA DASAR")
patches = create_color_patches()
models = ['RGB', 'HSV', 'LAB', 'GRAY']

fig, axes = plt.subplots(4, 8, figsize=(20, 10))

for r, model in enumerate(models):
    for c, (name, patch) in enumerate(patches):
        converted = analyze_color_model(patch, model)

        if model == 'GRAY':
            axes[r, c].imshow(converted, cmap='gray')
        elif model == 'HSV':
            axes[r, c].imshow(cv2.cvtColor(converted, cv2.COLOR_HSV2RGB))
        elif model == 'LAB':
            lab = converted.astype(np.float32)
            lab[:,:,0] = lab[:,:,0] * 255 / 100
            lab[:,:,1:] += 128
            lab = np.clip(lab, 0, 255).astype(np.uint8)
            axes[r, c].imshow(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB))
        else:
            axes[r, c].imshow(converted)

        if r == 0:
            axes[r, c].set_title(name, fontsize=9)
        axes[r, c].axis('off')

    axes[r, 0].text(-0.5, 0.5, model, transform=axes[r, 0].transAxes,
                    fontsize=12, fontweight='bold', va='center')

plt.suptitle("Perbandingan Model Warna", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()


# 2. LOAD CITRA
sample_img = cv2.imread('sample_image.jpg')
if sample_img is None:
    print("Menggunakan citra sintetis")
    sample_img = np.zeros((300, 400, 3), dtype=np.uint8)
    cv2.rectangle(sample_img, (50,50), (150,150), (255,0,0), -1)
    cv2.circle(sample_img, (250,100), 50, (0,255,0), -1)
    cv2.ellipse(sample_img, (200,220), (80,40), 0, 0, 360, (0,0,255), -1)


# 3. ANALISIS KEC0COKAN MODEL WARNA
print("\n2. ANALISIS KEC0COKAN MODEL WARNA")

applications = [
    'text_extraction',
    'skin_detection',
    'shadow_removal',
    'object_detection'
]

for app in applications:
    analysis = analyze_color_model_suitability(sample_img, app)
    print(f"\nAplikasi: {app}")
    print(f"Model warna terbaik: {analysis['best_model']}")
    print(f"Alasan: {analysis['reason']}")

    plt.figure(figsize=(4,4))
    if len(analysis['output'].shape) == 2:
        plt.imshow(analysis['output'], cmap='gray')
    else:
        plt.imshow(cv2.cvtColor(analysis['output'], cv2.COLOR_BGR2RGB))
    plt.title(f"{app} → {analysis['best_model']}")
    plt.axis('off')
    plt.show()


# 4. KUANTISASI
print("\n3. DEMONSTRASI KUANTISASI")

gray = cv2.cvtColor(sample_img, cv2.COLOR_BGR2GRAY)
levels = [256, 64, 16, 4]

fig, axes = plt.subplots(1, len(levels), figsize=(12,4))

for i, lvl in enumerate(levels):
    step = 256 // lvl
    quant = (gray // step) * step
    axes[i].imshow(quant, cmap='gray')
    axes[i].set_title(f"{lvl} Level")
    axes[i].axis('off')

plt.suptitle("Perbandingan Kuantisasi", fontsize=14, fontweight='bold')
plt.show()

print("\n=== PRAKTIKUM SELESAI ===")
