import unittest

import numpy as np

from main import blur_numpy, blur_python, gaussian_kernel


class GaussianBlurTests(unittest.TestCase):
    def test_kernel_is_symmetric_and_normalized(self):
        for size in (1, 3, 5, 7):
            kernel = gaussian_kernel(size, 1.2)
            self.assertEqual(kernel, kernel[::-1])
            self.assertAlmostEqual(sum(kernel), 1.0)

    def test_constant_image_stays_constant(self):
        image = np.full((4, 6, 3), 91.0)
        pixels = [(91, 91, 91)] * 24
        kernel = gaussian_kernel(5, 1.2)
        self.assertTrue(np.all(blur_numpy(image, kernel) == 91))
        self.assertEqual(blur_python(pixels, 6, 4, kernel), pixels)

    def test_implementations_agree_on_small_images(self):
        random = np.random.default_rng(3)
        for height, width in ((1, 1), (2, 3), (7, 9)):
            image = random.integers(0, 256, (height, width, 3), dtype=np.uint8)
            pixels = [tuple(map(int, pixel)) for pixel in image.reshape(-1, 3)]
            for size, sigma in ((1, 0.5), (3, 0.8), (5, 1.2), (7, 2.0)):
                kernel = gaussian_kernel(size, sigma)
                actual = blur_numpy(image.astype(np.float64), kernel)
                expected = np.asarray(blur_python(pixels, width, height, kernel), dtype=np.uint8)
                self.assertTrue(np.all(actual.reshape(-1, 3) == expected))


if __name__ == "__main__":
    unittest.main()
