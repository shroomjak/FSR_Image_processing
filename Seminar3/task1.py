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

def autocontrast_quantile_min_max(img_array: np.ndarray, p:float, q: float) -> np.ndarray:
    lower, upper = np.quantile(img_array, [p, 1.0-q])
    shift = int(lower)
    shifted_array = change_intensity(img_array=img_array, c=-shift)
    coeff = 255.0 / (upper - lower)
    return multiply_intensity(img_array=shifted_array, k=coeff)

def autocontrast_quantile_max_mean(img_array: np.ndarray, p:float) -> np.ndarray:
    upper = np.quantile(img_array, 1.0 - p)
    coeff = 128.0 / np.mean(img_array[img_array < upper])
    return multiply_intensity(img_array=img_array, k=coeff)

def add_salt_pepper_noise(img_array: np.ndarray, prob: float, salt_ratio: float = 0.5) -> np.ndarray:

    noisy = img_array.copy()
    rnd = np.random.rand(*img_array.shape[:2])

    salt_mask = rnd < (prob * salt_ratio)
    pepper_mask = (rnd >= (prob * salt_ratio)) & (rnd < prob)

    noisy[salt_mask] = 255
    noisy[pepper_mask] = 0

    return noisy


if __name__ == "__main__":
    img = Image.open('Seminar3/1.jpg')

    raw_img_arr = np.asarray(img, dtype=np.uint8)
    img_arr = multiply_intensity(raw_img_arr, k=1./2)

    noised_arr = add_salt_pepper_noise(img_arr, prob=0.25, salt_ratio=0.5)

    variants = [
        ("Original", raw_img_arr),
        #("Autocontrast max", autocontrast_max(img_array=img_arr)),
        #("Autocontrast_min_max", autocontrast_min_max(img_array=img_arr))
        #("Salt noise", add_salt_pepper_noise(img_arr, prob=0.1, salt_ratio=1.0)),
        #("Pepper noise", add_salt_pepper_noise(img_arr, prob=0.1, salt_ratio=0.0)),
        #("Noised arr", noised_arr),
        ("Auto", autocontrast_quantile_min_max(noised_arr, p=0.00, q=0.5)),
        ("Mean auto", autocontrast_quantile_max_mean(noised_arr, p=0.5))
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