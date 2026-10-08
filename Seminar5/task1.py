import numpy as np
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')

def get_pixels(img_src: np.ndarray, x_idx: np.ndarray, y_idx: np.ndarray) -> np.ndarray:
    h, w = img_src.shape[:2]
    valid_mask = (
        (x_idx >= 0) & (x_idx < w) &
        (y_idx >= 0) & (y_idx < h)
    )

    result = np.zeros(x_idx.shape, dtype=np.float64)
    result[valid_mask] = img_src[y_idx[valid_mask], x_idx[valid_mask]]
    return result


def nearest_sample(
        img_src: np.ndarray,
        h_new: int,
        w_new: int,
):
    h_src, w_src = img_src.shape[:2]
    s_x = w_src / w_new
    s_y = h_src / h_new

    y_dst, x_dst = np.indices((h_new, w_new))    
    x_src = (x_dst + 0.5) * s_x - 0.5
    y_src = (y_dst + 0.5) * s_y - 0.5

    x_idx = np.rint(x_src).astype(np.int64)
    y_idx = np.rint(y_src).astype(np.int64)

    return get_pixels(
            img_src=img_src,
            x_idx=x_idx,
            y_idx=y_idx
        )


def bilinear_sample(
        img_src: np.ndarray,
        h_new: int,
        w_new: int
):
    h_src, w_src = img_src.shape[:2]
    s_x = w_src / w_new
    s_y = h_src / h_new

    y_dst, x_dst = np.indices((h_new, w_new))    
    x_src = (x_dst + 0.5) * s_x - 0.5
    y_src = (y_dst + 0.5) * s_y - 0.5

    x0 = np.floor(x_src).astype(np.int64)
    y0 = np.floor(y_src).astype(np.int64)
    x1 = x0 + 1
    y1 = y0 + 1

    dx = x_src - x0
    dy = y_src - y0

    img_src_float = img_src.astype(np.float64)
    p00 = get_pixels(img_src_float, x0, y0)
    p01 = get_pixels(img_src_float, x0, y1)
    p10 = get_pixels(img_src_float, x1, y0)
    p11 = get_pixels(img_src_float, x1, y1)

    img_dst = (1 - dx)*(1 - dy)*p00 + dx*(1 - dy)*p10 + (1 - dx)*dy*p01 + dx*dy*p11

    return np.clip(img_dst, 0, 255).astype(np.uint8) 


if __name__ == "__main__":
    img = Image.open('Seminar5/1.jpg')
    img = img.resize((100, 100))
    img_arr = np.asarray(img, dtype=np.uint8)
    h, w = img_arr.shape[:2]

    k = 2
    variants = [
            ("Original", img_arr),
            (f"Multiple {k}, nearest", nearest_sample(img_arr, h * k, w * k)),
            (f"Divide {k}, nearest", nearest_sample(img_arr, h // k, w // k)),
            (f"Multiple {k}, bilinear", bilinear_sample(img_arr, h * k, w * k)),
            (f"Divide {k}, bilinear", bilinear_sample(img_arr, h // k, w // k)),
        ]

    fig, ax = plt.subplots(1, len(variants), figsize=(8, 14))
    
    for row, (title, arr) in enumerate(variants):
        ax[row].imshow(Image.fromarray(arr), cmap="gray", vmin=0, vmax=255)
        ax[row].set_title(title)
        ax[row].axis('off')
    
    plt.tight_layout()
    plt.show()


"""
width_ratios = [arr.shape[1] for _, arr in variants]

fig, ax = plt.subplots(
    1,
    len(variants),
    figsize=(16, 5),
    gridspec_kw={"width_ratios": width_ratios}
)

for axis, (title, arr) in zip(ax, variants):
    axis.imshow(
        arr,
        cmap="gray",
        vmin=0,
        vmax=255,
        interpolation="nearest",
        aspect="equal"
    )
    axis.set_title(f"{title}\n{arr.shape[1]}×{arr.shape[0]}")
    axis.axis("off")

plt.tight_layout()
plt.show()
"""