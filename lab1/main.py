import argparse
import math
import statistics
import time
from pathlib import Path

import numpy as np
from PIL import Image


def gaussian_kernel(size, sigma):
    radius = size // 2
    weights = []
    for offset in range(-radius, radius + 1):
        distance = abs(offset) / sigma
        weights.append(0.0 if distance > 40 else math.exp(-0.5 * distance * distance))
    total = sum(weights)
    return [weight / total for weight in weights]


def blur_numpy(image, kernel):
    radius = len(kernel) // 2

    padded = np.pad(image, ((0, 0), (radius, radius), (0, 0)), mode="edge")
    horizontal = np.apply_along_axis(lambda line: np.convolve(line, kernel, mode="valid"), 1, padded)

    padded = np.pad(horizontal, ((radius, radius), (0, 0), (0, 0)), mode="edge")
    vertical = np.apply_along_axis(lambda line: np.convolve(line, kernel, mode="valid"), 0, padded)

    return np.rint(np.clip(vertical, 0, 255)).astype(np.uint8)


def blur_python(pixels, width, height, kernel):
    radius = len(kernel) // 2
    horizontal = [[[0.0, 0.0, 0.0] for _ in range(width)] for _ in range(height)]

    for y in range(height):
        for x in range(width):
            for index, weight in enumerate(kernel):
                source_x = min(max(x + index - radius, 0), width - 1)
                source = pixels[y * width + source_x]
                for channel in range(3):
                    horizontal[y][x][channel] += source[channel] * weight

    result = []
    for y in range(height):
        for x in range(width):
            color = [0.0, 0.0, 0.0]
            for index, weight in enumerate(kernel):
                source_y = min(max(y + index - radius, 0), height - 1)
                source = horizontal[source_y][x]
                for channel in range(3):
                    color[channel] += source[channel] * weight
            result.append(tuple(round(min(max(value, 0), 255)) for value in color))

    return result


def measure(function, repeats):
    times = []
    result = None
    for _ in range(repeats):
        start = time.perf_counter()
        result = function()
        times.append(time.perf_counter() - start)
    return result, statistics.median(times)


def main():
    parser = argparse.ArgumentParser(description="Gaussian blur with NumPy and pure Python")
    parser.add_argument("input", type=Path, help="Input image")
    parser.add_argument("--kernel-size", type=int, default=5, help="Positive odd kernel size")
    parser.add_argument("--sigma", type=float, default=1.2, help="Positive Gaussian standard deviation")
    parser.add_argument("--repeats", type=int, default=5, help="Number of timing runs")
    parser.add_argument("--output-dir", type=Path, default=Path("output"), help="Output directory")
    args = parser.parse_args()

    if args.kernel_size < 1 or args.kernel_size % 2 == 0:
        parser.error("--kernel-size must be a positive odd number")
    if not np.isfinite(args.sigma) or args.sigma <= 0:
        parser.error("--sigma must be a positive finite number")
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if not args.input.is_file():
        parser.error(f"input file does not exist: {args.input}")

    with Image.open(args.input) as source:
        image = source.convert("RGB")

    width, height = image.size
    image_pixels = image.load()
    pixels = [image_pixels[x, y] for y in range(height) for x in range(width)]
    array = np.asarray(image, dtype=np.float64)
    kernel = gaussian_kernel(args.kernel_size, args.sigma)

    numpy_result, numpy_time = measure(lambda: blur_numpy(array, kernel), args.repeats)
    python_result, python_time = measure(lambda: blur_python(pixels, width, height, kernel), args.repeats)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    numpy_path = args.output_dir / "gaussian_numpy.png"
    python_path = args.output_dir / "gaussian_python.png"
    Image.fromarray(numpy_result).save(numpy_path)
    native_image = Image.new("RGB", (width, height))
    native_image.putdata(python_result)
    native_image.save(python_path)

    difference = np.abs(numpy_result.astype(np.int16) - np.asarray(native_image, dtype=np.int16))
    print(f"Image: {width} x {height}")
    print(f"Kernel: {args.kernel_size} x {args.kernel_size}; sigma: {args.sigma}")
    print(f"NumPy median: {numpy_time * 1000:.3f} ms")
    print(f"Python median: {python_time * 1000:.3f} ms")
    print(f"Speedup: {python_time / numpy_time:.1f}x")
    print(f"Maximum channel difference: {difference.max()}")
    print(f"Saved: {numpy_path}, {python_path}")


if __name__ == "__main__":
    main()
