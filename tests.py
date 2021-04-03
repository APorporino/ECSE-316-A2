import unittest
import fft
import numpy as np


class TestFFTProgram(unittest.TestCase):

    def test1_naive_dft(self):
        input_array_1d = np.array([1, 2, 3, 4])

        oracle = np.fft.fft(input_array_1d)
        result = fft.naive_dft(input_array_1d)
        self.assertTrue(np.allclose(oracle, result))

    def test2_naive_dft(self):
        input_array_1d_big = np.arange(64)

        oracle = np.fft.fft(input_array_1d_big)
        result = fft.naive_dft(input_array_1d_big)
        self.assertTrue(np.allclose(oracle, result))

    def test1_naive_dft_2d(self):
        input_array_2d = np.array([[1, 2, 3],[4, 5, 6], [2, 2, 2]])

        oracle = np.fft.fft2(input_array_2d)
        result = fft.naive_2d_dft(input_array_2d)
        self.assertTrue(np.allclose(oracle, result))

    def test2_naive_dft_2d(self):
        input_array_2d = np.array([[4, 2],[4, 6], [2, 2], [1, 1], [9, 8]])

        oracle = np.fft.fft2(input_array_2d)
        result = fft.naive_2d_dft(input_array_2d)
        self.assertTrue(np.allclose(oracle, result))

    def test1_fft(self):
        input_array_1d = np.arange(16)

        oracle = np.fft.fft(input_array_1d)
        result = fft.outer_fft_dft(input_array_1d)
        self.assertTrue(np.allclose(oracle, result))

    def test2_fft(self):
        input_array_1d = np.arange(256)

        oracle = np.fft.fft(input_array_1d)
        result = fft.outer_fft_dft(input_array_1d)
        self.assertTrue(np.allclose(oracle, result))

    def test1_fft_2d(self):
        input_array_2d = np.random.rand(3, 32)

        oracle = np.fft.fft2(input_array_2d)
        result = fft.fft_2d_dft(input_array_2d)
        self.assertTrue(np.allclose(oracle, result))

    def test2_fft_2d(self):
        input_array_2d = np.random.rand(3, 256)

        oracle = np.fft.fft2(input_array_2d)
        result = fft.fft_2d_dft(input_array_2d)
        self.assertTrue(np.allclose(oracle, result))


if __name__ == '__main__':
    unittest.main()
