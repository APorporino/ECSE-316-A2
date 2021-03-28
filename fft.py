import numpy as np
import matplotlib
# import cv2
import math


def naive_dft(input_array):
    N = input_array.size
    output_array = np.zeros(N, dtype=np.complex_)
    for k in range(N):
        X = 0
        for n in range(N):
            xn = input_array[n]
            arg = -1j * 2 * math.pi * k * n / N
            X += xn * np.exp(arg)
        output_array[k] = X
    return np.array(output_array)


def naive_2d_dft(input_array):
    M, N = input_array.shape
    output_array = np.zeros((M, N), dtype=np.complex_)
    for k in range(M):
        for l in range(N):
            Fkl = 0
            for n in range(N):
                Fk = 0
                for m in range(M):
                    fmn = input_array[m][n]
                    arg = -1j * 2 * math.pi * k * m / M
                    Fk += fmn * np.exp(arg)
                arg = -1j * 2 * math.pi * l * n / N
                Fkl += Fk * np.exp(arg)
            output_array[k][l] = Fkl
    return np.array(output_array)


# Assuming input_array size is a power of 2
def fft_dft(input_array):
    size = input_array.size
    if size > 16:
        even = fft_dft(input_array[::2])
        odd = fft_dft(input_array[1::2])
        odd_series_constant = fft_constant_coefficients(size)
        return np.concatenate([even + odd_series_constant[:size / 2] * odd,
                               even + odd_series_constant[size / 2:] * odd])
    else:
        return naive_dft(input_array)


def fft_constant_coefficients(size):
    output_array = np.zeros(size, dtype=np.complex_)
    for k in range(size):
        arg = -1j * 2 * math.pi * k / size
        output_array[k] = np.exp(arg)
    return output_array


def inverse_fft_dft():
    pass


def fft_2d_dft():
    pass


def inverse_fft_2d_dft():
    pass


if __name__ == "__main__":
    input_arr = np.arange(64)
    output = fft_dft(input_arr)
    input2 = np.array([[1, 2, 3],
                       [4, 5, 6]])
    output2 = naive_2d_dft(input2)
    print("Output ", output)
    print("Output2 ", output2)
