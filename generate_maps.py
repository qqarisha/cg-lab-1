from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

SIZE = 512
OUTPUT_DIR = Path("textures")


def perlin_noise(width, height, cells, seed):
    rng = np.random.default_rng(seed)

    gradients = rng.normal(size=(cells, cells, 2))
    gradients /= np.linalg.norm(
        gradients,
        axis=2,
        keepdims=True
    ) + 1e-12

    x = np.linspace(0, cells, width, endpoint=False)
    y = np.linspace(0, cells, height, endpoint=False)
    xx, yy = np.meshgrid(x, y)

    x0 = np.floor(xx).astype(int)
    y0 = np.floor(yy).astype(int)

    x1 = (x0 + 1) % cells
    y1 = (y0 + 1) % cells

    xf = xx - x0
    yf = yy - y0

    def fade(t):
        return 6 * t**5 - 15 * t**4 + 10 * t**3

    u = fade(xf)
    v = fade(yf)

    g00 = gradients[y0 % cells, x0 % cells]
    g10 = gradients[y0 % cells, x1]
    g01 = gradients[y1, x0 % cells]
    g11 = gradients[y1, x1]

    n00 = g00[..., 0] * xf + g00[..., 1] * yf
    n10 = g10[..., 0] * (xf - 1) + g10[..., 1] * yf
    n01 = g01[..., 0] * xf + g01[..., 1] * (yf - 1)
    n11 = g11[..., 0] * (xf - 1) + g11[..., 1] * (yf - 1)

    nx0 = n00 * (1 - u) + n10 * u
    nx1 = n01 * (1 - u) + n11 * u

    return (nx0 * (1 - v) + nx1 * v) * 1.6


def fractal_noise(width, height, seed, octaves):
    result = np.zeros((height, width))
    amplitude = 1.0
    amplitude_sum = 0.0

    for octave in range(octaves):
        cells = 4 * (2 ** octave)

        result += amplitude * perlin_noise(
            width,
            height,
            cells,
            seed + octave * 37
        )

        amplitude_sum += amplitude
        amplitude *= 0.5

    result /= amplitude_sum

    return (result - result.min()) / (
        result.max() - result.min() + 1e-12
    )


def smoothstep(edge0, edge1, value):
    t = np.clip((value - edge0) / (edge1 - edge0), 0, 1)
    return t * t * (3 - 2 * t)


def generate_specular():
    noise = fractal_noise(SIZE, SIZE, 17, 4)

    mask = smoothstep(0.38, 0.78, noise) ** 1.35

    image = np.zeros((SIZE, SIZE, 4), dtype=np.uint8)

    gray = (mask * 255).astype(np.uint8)

    image[:, :, 0] = gray
    image[:, :, 1] = gray
    image[:, :, 2] = gray
    image[:, :, 3] = 255

    plt.imsave(OUTPUT_DIR / "specular.png", image)


def generate_emissive():
    noise = fractal_noise(SIZE, SIZE, 53, 5)

    mask = smoothstep(0.58, 0.82, noise) ** 1.6

    image = np.zeros((SIZE, SIZE, 4), dtype=np.uint8)

    image[:, :, 0] = (mask * 45).astype(np.uint8)
    image[:, :, 1] = (mask * 210).astype(np.uint8)
    image[:, :, 2] = (mask * 255).astype(np.uint8)
    image[:, :, 3] = 255

    plt.imsave(OUTPUT_DIR / "emissive.png", image)


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(exist_ok=True)

    generate_specular()
    generate_emissive()

    print("Texture maps generated successfully")