import unittest
import fft
import numpy as np


input_array_2d_index0 = np.arange(64)
input_array_2d_index1 = np.arange(64)
input_array_2d_index2 = np.arange(64)

input_array_2d = np.array([input_array_2d_index0, input_array_2d_index1, input_array_2d_index2])


class TestFFTProgram(unittest.TestCase):

    def test1_naive_fft(self):
        input_array_1d = np.array([1, 2, 3, 4])

        oracle = np.fft.fft(input_array_1d)
        result = fft.naive_dft(input_array_1d)
        self.assertTrue(np.allclose(oracle, result))

    def test2_naive_fft(self):
        input_array_1d_big = np.arange(64)

        oracle = np.fft.fft(input_array_1d_big)
        result = fft.naive_dft(input_array_1d_big)
        self.assertTrue(np.allclose(oracle, result))


if __name__ == '__main__':
    unittest.main()
