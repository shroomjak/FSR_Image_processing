import numpy as np
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')

def convolve(src_image: np.ndarray, kernel: np.ndarray):
    h, w = src_image.shape[:2]
    kh, kw = kernel.shape

    pad_h = kh // 2
    pad_w = kw // 2

    kernel = kernel.astype(np.float64)
    kernel_sum = np.sum(kernel)
    if kernel_sum != 0:
        kernel /= kernel_sum

    padded_image = np.pad(src_image, ((pad_h, pad_h), (pad_w, pad_w)), mode='edge').astype(np.float64)
    result = np.zeros((h, w), dtype=np.float64)

    for i in range(h):
        for j in range(w):
            region = padded_image[i:i + kh, j:j + kw]
            result[i, j] = np.sum(region * kernel)

    return result.astype(np.uint8)

def median_filter(src_image: np.ndarray, kernel_size: int):
    h, w = src_image.shape[:2]
    pad_size = kernel_size // 2

    padded_image = np.pad(src_image, ((pad_size, pad_size), (pad_size, pad_size)), mode='edge')
    result = np.zeros((h, w), dtype=np.uint8)

    for i in range(h):
        for j in range(w):
            region = padded_image[i:i + kernel_size, j:j + kernel_size]
            result[i, j] = np.median(region)

    return result

def generate_gaussian_kernel(size, sigma):
    ax = np.linspace(-(size - 1) / 2.0, (size - 1) / 2.0, size)
    x, y = np.meshgrid(ax, ax)
    
    kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    return kernel / np.sum(kernel)

def add_salt_pepper_noise(img_array: np.ndarray, prob: float, salt_ratio: float = 0.5) -> np.ndarray:

    noisy = img_array.copy()
    rnd = np.random.rand(*img_array.shape[:2])

    salt_mask = rnd < (prob * salt_ratio)
    pepper_mask = (rnd >= (prob * salt_ratio)) & (rnd < prob)

    noisy[salt_mask] = 255
    noisy[pepper_mask] = 0

    return noisy

if __name__ == "__main__":
    img = Image.open("Seminar6/1.jpg").convert("L")
    img = img.resize((100, 100))
    img_arr = np.asarray(img, dtype=np.uint8)
    print("Image shape:", img_arr.shape)

    print(generate_gaussian_kernel(3, 1.0))

    variants1 = [
                ("Original", img_arr),
                ("Gaussian Blur", convolve(img_arr, generate_gaussian_kernel(9, 1.0))),
                ("Smooth", convolve(img_arr, np.ones((9, 9)))),
            ]

    noised_sp_img_arr = add_salt_pepper_noise(img_arr, prob=0.05, salt_ratio=0.5)
    noised_gauss_img_arr = np.clip(img_arr + np.random.normal(0, 5, img_arr.shape).astype(np.uint8), 0, 255).astype(np.uint8)

    variants2 = [
                    ("Original Noised", noised_sp_img_arr),
                    ("Gaussian Blur", convolve(noised_sp_img_arr, generate_gaussian_kernel(3, 1.0))),
                    ("Median Filter", median_filter(noised_sp_img_arr, 3)),
                    ("Mean Filter", convolve(noised_sp_img_arr, np.ones((3, 3)))),
                ]

    variants3 = [
                    ("Original Noised", noised_gauss_img_arr),
                    ("Gaussian Blur", convolve(noised_gauss_img_arr, generate_gaussian_kernel(11, 1.0))),
                    ("Median Filter", median_filter(noised_gauss_img_arr, 11)),
                    ("Mean Filter", convolve(noised_gauss_img_arr, np.ones((11, 11)))),
                ]
    
    fig, ax = plt.subplots(1, len(variants2), figsize=(8, 14))
    
    for row, (title, arr) in enumerate(variants2):
        ax[row].imshow(Image.fromarray(arr), cmap="gray", vmin=0, vmax=255)
        ax[row].set_title(title)
        ax[row].axis('off')
    
    plt.tight_layout()
    plt.show()
    

