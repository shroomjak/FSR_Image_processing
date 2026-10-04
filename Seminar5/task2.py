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

def apply_projective(img_src: np.ndarray, matrix: np.ndarray):
    h_src, w_src = img_src.shape[:2]
    cx = (w_src - 1) / 2.0
    cy = (h_src- 1) / 2.0
    inv_matrix = np.linalg.inv(matrix)

    y_dst, x_dst = np.indices((h_src, w_src))

    x = x_dst - cx
    y = y_dst - cy
    
    den = (
        inv_matrix[2, 0] * x + 
        inv_matrix[2, 1] * y + 
        inv_matrix[2, 2]
    )
    x_src = (
        inv_matrix[0, 0] * x + 
        inv_matrix[0, 1] * y + 
        inv_matrix[0, 2]
    ) / den + cx
    y_src = (
        inv_matrix[1, 0] * x + 
        inv_matrix[1, 1] * y + 
        inv_matrix[1, 2]
    ) / den + cy

    return x_src, y_src

def projective_nearest(img_src: np.ndarray, matrix: np.ndarray):
    x_src, y_src = apply_projective(img_src, matrix)

    x_idx = np.rint(x_src).astype(np.int64)
    y_idx = np.rint(y_src).astype(np.int64)

    return get_pixels(
            img_src=img_src,
            x_idx=x_idx,
            y_idx=y_idx
        )
    

def projective_bilinear(img_src: np.ndarray, matrix: np.ndarray):
    x_src, y_src = apply_projective(img_src, matrix)

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

    img_dst = (1 - dx)*(1 - dy)*p00 + dx*(1 - dy)*p10 + (1 - dx)*dy*p01 +dx*dy*p11

    return np.clip(img_dst, 0, 255).astype(np.uint8) 


if __name__ == "__main__":
    img = Image.open('Seminar5/1.jpg')
    img = img.resize((100, 100))
    img_arr = np.asarray(img, dtype=np.uint8)
    h, w = img_arr.shape[:2]

    alpha = np.deg2rad(45)
    matrices = [
            np.array([
                [np.cos(alpha), -np.sin(alpha), 0.0],
                [np.sin(alpha), np.cos(alpha), 0.0],
                [0, 0, 1]
            ], dtype=np.float64),
            np.array([
                [2, 0, 0],
                [0, 2, 0],
                [0, 0, 1]
            ], dtype=np.float64),
            np.array([
                [1, 0, 0],
                [1, 1, 0],
                [0, 0.01, 1]
            ], dtype=np.float64)
    ]

    print(matrices[0])
    
    variants = [
            ("Original", img_arr),
            ("Projective nearest",
             projective_nearest(
                img_src=img_arr, matrix=matrices[2]
            )),
            ("Projective bilinear",
             projective_bilinear(
                img_src=img_arr, matrix=matrices[2]
            ))
        ]

    fig, ax = plt.subplots(1, len(variants), figsize=(8, 14))
    
    for row, (title, arr) in enumerate(variants):
        ax[row].imshow(Image.fromarray(arr), cmap="gray", vmin=0, vmax=255)
        ax[row].set_title(title)
        ax[row].axis('off')
    
    plt.tight_layout()
    plt.show()
