import numpy as np
import matplotlib.pyplot as plt

size = (10, 10)
Y = 200.0

weights1 = np.array([1.0, 1.0, 1.0]) / 3
weights2 = np.array([0.299, 0.587, 0.114])
weights3 = np.array([0.212, 0.715, 0.072])

weights_all = [weights1, weights2, weights3]
names = ['Среднее RGB', 'BT.601', 'Rec.709']

fig, axes = plt.subplots(2, 3, figsize=(15, 9))

rng = np.random.default_rng()

for k, w in enumerate(weights_all):
    a, b, c = w

    arr = np.empty((*size, 3), dtype=np.float64)

    # Генерируем кандидаты, пока B не попадёт в [0, 255].
    for i in range(size[0]):
        for j in range(size[1]):
            while True:
                R = rng.uniform(0, 255)
                G = rng.uniform(0, 255)
                B = (Y - a * R - b * G) / c

                if 0 <= B <= 255:
                    arr[i, j] = (R, G, B)
                    break

    gray = arr @ w
    print(
        f'{names[k]}: '
        f'min={gray.min():.12f}, '
        f'max={gray.max():.12f}, '
        f'max |error|={np.max(np.abs(gray - Y)):.3e}'
    )

    rgb8 = np.rint(arr).astype(np.uint8)

    gray8 = rgb8.astype(np.float64) @ w

    axes[0, k].imshow(rgb8)
    axes[0, k].set_title(names[k])
    axes[0, k].axis('off')

    im = axes[1, k].imshow(gray8, cmap='gray', vmin=0, vmax=255)
    axes[1, k].set_title(
        f'После uint8\n'
        f'min={gray8.min():.3f}, max={gray8.max():.3f}'
    )
    axes[1, k].axis('off')

plt.tight_layout()
plt.show()