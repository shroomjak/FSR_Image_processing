import numpy as np
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')

def get_pixels(src: np.ndarray, x_idx: np.ndarray, y_idx: np.ndarray) -> np.ndarray:
    src_float = src.astype(np.float64)
    h, w = src.shape[:2]
    valid = (
        (x_idx >= 0) & (x_idx < w) &
        (y_idx >= 0) & (y_idx < h)
    )

    result = np.zeros(x_idx.shape, dtype=np.float64)
    result[valid] = src_float[y_idx[valid], x_idx[valid]]
    return result

def bilinear_sample(
    src: np.ndarray,
    x_src: np.ndarray,
    y_src: np.ndarray,
) -> np.ndarray:

    x0 = np.floor(x_src).astype(np.int64)
    y0 = np.floor(y_src).astype(np.int64)
    x1 = x0 + 1
    y1 = y0 + 1

    # Доли между соседними узлами
    dx = x_src - x0
    dy = y_src - y0

    p00 = get_pixels(src, x0, y0)
    p10 = get_pixels(src, x1, y0)
    p01 = get_pixels(src, x0, y1)
    p11 = get_pixels(src, x1, y1)

    return (
        (1.0 - dx)*(1.0 - dy) * p00 + dx * (1.0 - dy) * p10 + (1.0 - dx) * dy * p01 + dx * dy * p11
    )


def shift(src: np.ndarray, mode: str, dx: float, dy: float) -> np.ndarray:
    h, w = src.shape[:2]
    dst = np.zeros_like(src)

    y_dst, x_dst = np.indices((h, w))

    x_src = x_dst - dx
    y_src = y_dst - dy

    if mode == "nearest":
        x_src_i = np.rint(x_src).astype(np.int64)
        y_src_i = np.rint(y_src).astype(np.int64)

        dst = get_pixels(src, x_src_i, y_src_i) 

    elif mode == "bilinear":
        dst = bilinear_sample(src, x_src, y_src)

    return dst


def rotate(src: np.ndarray, mode: str, angle_deg: float) -> np.ndarray:
    h, w = src.shape[:2]
    dst = np.zeros_like(src)

    theta = np.deg2rad(angle_deg)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    cx = (w - 1) / 2.0
    cy = (h - 1) / 2.0

    y_dst, x_dst = np.indices((h, w))

    x = x_dst - cx
    y = y_dst - cy

    x_src = cx + cos_t * x - sin_t * y
    y_src = cy + sin_t * x + cos_t * y

    if mode == "nearest":
        x_src_i = np.rint(x_src).astype(np.int64)
        y_src_i = np.rint(y_src).astype(np.int64)

        dst = get_pixels(src, x_src_i, y_src_i)   

    elif mode == "bilinear":
        dst = bilinear_sample(src, x_src, y_src)

    return dst

if __name__ == "__main__":
    img = Image.open('Seminar4/1.jpg')
    img = img.resize((50, 50))
    img_arr = np.asarray(img, dtype=np.uint8)

    img_shift100 = img_arr
    for i in range(100):
        img_shift100 = shift(img_shift100, mode="bilinear", dx=0.01, dy=0.01)

    variants = [
            ("Original", img_arr),
            ("Shift", shift(img_arr, mode="bilinear", dx=1.0, dy=1.0)),
            ("100xshift", img_shift100),
            ("Rotate nearest", rotate(img_arr, mode="nearest", angle_deg=45)),
            ("Rotate bilinear", rotate(img_arr, mode="bilinear", angle_deg=45)),
        ]

    fig, ax = plt.subplots(1, len(variants), figsize=(8, 14))
    
    for row, (title, arr) in enumerate(variants):
        ax[row].imshow(Image.fromarray(arr), cmap="gray", vmin=0, vmax=255)
        ax[row].set_title(title)
        ax[row].axis('off')
    
    plt.tight_layout()
    plt.show()
     
    