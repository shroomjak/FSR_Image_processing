import numpy as np
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')

def change_intensity(img_array: np.ndarray, c: int) -> np.ndarray:
    result = img_array.astype(np.int16) + c
    return np.clip(result, 0, 255).astype(np.uint8)

def multiply_intensity(img_array: np.ndarray, k: float) -> np.ndarray:
    result = np.round(img_array.astype(np.float64) * k)
    return np.clip(result, 0.0, 255.0).astype(np.uint8)

def gamma_correction(img_array: np.ndarray, gamma: float) -> np.ndarray:
    normalized = img_array.astype(np.float64) / 255.0
    corrected = np.power(normalized, gamma)
    result = np.round(corrected * 255.0)
    return np.clip(result, 0, 255).astype(np.uint8)

def autocontrast_max(img_array: np.ndarray) -> np.ndarray:
    coeff = 255.0 / np.max(img_array)
    return multiply_intensity(img_array=img_array, k=coeff)

def autocontrast_min_max(img_array: np.ndarray) -> np.ndarray:
    shift = int(np.min(img_array))
    shifted_array = change_intensity(img_array=img_array, c=-shift)
    return autocontrast_max(shifted_array)

if __name__ == "__main__":
    img = Image.open('Seminar2/1.jpg')

    img_arr = change_intensity(img_array=multiply_intensity(img_array=np.asarray(img, dtype=np.uint8), k=1./3), c=50.0)

    variants = [
        ("Original", img_arr),
        #("Brightness +50", change_intensity(img_array, c=50)),
        #("Brightness -50", change_intensity(img_array, c=-50)),
        #("Multiply x0.75", multiply_intensity(img_array, k=0.75)),
        #("Gamma 0.5", gamma_correction(img_array, gamma=0.5)),
        #("Gamma 2.0", gamma_correction(img_array, gamma=2.0)),
        #("Multiple/Division", multiply_intensity(multiply_intensity(img_array, k=0.5), k=2.0))
        ("Autocontrast max", autocontrast_max(img_array=img_arr)),
        ("Autocontrast_min_max", autocontrast_min_max(img_array=img_arr))
    ]

    fig, ax = plt.subplots(len(variants), 2, figsize=(8, 14))

    for row, (title, arr) in enumerate(variants):
        histogram, bin_edges = np.histogram(arr, bins=256, range=(0, 255))

        ax[row, 0].set_title(f"Histogram: {title}")
        ax[row, 0].set_xlabel("grayscale value")
        ax[row, 0].set_ylabel("pixel count")
        ax[row, 0].set_xlim([0, 255])
        ax[row, 0].scatter(bin_edges[0:-1], histogram, s=5)

        ax[row, 1].imshow(Image.fromarray(arr), cmap="gray", vmin=0, vmax=255)
        ax[row, 1].set_title(title)
        ax[row, 1].axis('off')

    plt.tight_layout()
    plt.show()