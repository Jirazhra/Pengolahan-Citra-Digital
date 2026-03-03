import cv2
import numpy as np
import matplotlib.pyplot as plt


def simulate_image_aliasing(image, downsampling_factors):
    """
    Simulate aliasing by downsampling image

    Parameters:
    image: Input image (numpy array)
    downsampling_factors: List of factors (2, 4, 8, etc.)

    Returns:
    results: Dictionary of downsampled images and aliasing analysis
    """

    results = {}

    # Konversi ke grayscale agar fokus ke intensitas
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape

    for factor in downsampling_factors:
        # -------------------------------
        # Downsampling TANPA anti-aliasing
        # -------------------------------
        downsampled_nn = gray[::factor, ::factor]
        reconstructed_nn = cv2.resize(
            downsampled_nn,
            (width, height),
            interpolation=cv2.INTER_NEAREST
        )

        # --------------------------------
        # Downsampling DENGAN anti-aliasing
        # --------------------------------
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        downsampled_aa = blurred[::factor, ::factor]
        reconstructed_aa = cv2.resize(
            downsampled_aa,
            (width, height),
            interpolation=cv2.INTER_LINEAR
        )

        # -------------------------------
        # Analisis aliasing
        # -------------------------------
        error_nn = np.mean(np.abs(gray - reconstructed_nn))
        error_aa = np.mean(np.abs(gray - reconstructed_aa))

        results[factor] = {
            "original": gray,
            "reconstructed_nearest": reconstructed_nn,
            "reconstructed_antialias": reconstructed_aa,
            "analysis": {
                "error_without_antialiasing": error_nn,
                "error_with_antialiasing": error_aa
            }
        }

    return results


# ==========================
# PROGRAM UTAMA
# ==========================
if __name__ == "__main__":

    # Baca citra
    image = cv2.imread("IMG1213.jpg")

    if image is None:
        print("Gambar tidak ditemukan!")
        exit()

    # Faktor downsampling
    factors = [2, 4, 8]

    # Jalankan simulasi
    results = simulate_image_aliasing(image, factors)

    # ==========================
    # VISUALISASI HASIL
    # ==========================
    for factor, data in results.items():
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 3, 1)
        plt.imshow(data["original"], cmap="gray")
        plt.title("Citra Asli")
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.imshow(data["reconstructed_nearest"], cmap="gray")
        plt.title(f"Downsampling {factor}x\nTanpa Anti-Aliasing")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.imshow(data["reconstructed_antialias"], cmap="gray")
        plt.title(f"Downsampling {factor}x\nDengan Anti-Aliasing")
        plt.axis("off")

        plt.suptitle(
            f"Aliasing Error (NN): {data['analysis']['error_without_antialiasing']:.2f} | "
            f"Aliasing Error (AA): {data['analysis']['error_with_antialiasing']:.2f}"
        )

        plt.tight_layout()
        plt.show()

    print("Simulasi aliasing selesai.")
