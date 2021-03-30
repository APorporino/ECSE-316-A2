import numpy as np
import matplotlib
import cv2
import math
import os
import argparse

smallest_size_array = 16
coeff_array = [np.exp(-1j * 2 * np.pi * m / smallest_size_array) for m in range(smallest_size_array)]


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


def naive_dft_k(input_array, k):
    N = input_array.size
    X = 0
    for n in range(N):
        xn = input_array[n]
        arg = -1j * 2 * math.pi * k * n / N
        X += xn * np.exp(arg)
    return X


def naive_2d_dft(input_array):
    R, C = input_array.shape
    output_array = np.zeros((R, C), dtype=np.complex_)
    for k in range(R):
        for l in range(C):
            output_array[k][l] = naive_2d_dft_k_l(input_array, k, l)
    return np.array(output_array)


def naive_2d_dft_k_l(input_array, k, l):
    R, C = input_array.shape
    input_array_transpose = input_array.transpose()

    temp = np.zeros(C, dtype=np.complex_)

    for i in range(C):
        temp[i] = naive_dft_k(input_array_transpose[i], k)

    return naive_dft_k(temp, l)


# Assuming input_array size is a power of 2
def inner_fft_dft(input_array, k):
    size = input_array.size
    if size > 16:
        even = inner_fft_dft(input_array[::2], k)
        odd = inner_fft_dft(input_array[1::2], k)
        return even + np.exp(-1j * 2 * np.pi * k / size) * odd
    else:
        return naive_dft_k_precomputed(input_array, k)


def outer_fft_dft(input_array):
    size = input_array.size
    output_array = np.zeros(size, dtype=np.complex_)
    for k in range(input_array.size):
        output_array[k] = inner_fft_dft(input_array, k)
    return output_array


def naive_dft_k_precomputed(input_array, k):
    N = input_array.size
    X = 0
    for n in range(N):
        xn = input_array[n]
        a = np.power(coeff_array[n], k)
        X += xn * a
    return X


def inverse_fft_dft():
    pass


def fft_2d_dft(input_array):
    R, C = input_array.shape
    output_array = np.zeros((R, C), dtype=np.complex_)
    for k in range(R):
        for l in range(C):
            output_array[k][l] = fft_2d_dft_k_l(input_array, k, l)
    return np.array(output_array)


def fft_2d_dft_k_l(input_array, k, l):
    R, C = input_array.shape
    input_array_transpose = input_array.transpose()

    temp = np.zeros(C, dtype=np.complex_)

    for i in range(C):
        temp[i] = inner_fft_dft(input_array_transpose[i], k)

    return inner_fft_dft(temp, l)


def inverse_fft_2d_dft():
    pass


def main():
    parser = argparse.ArgumentParser(description='Arguments for FFT program')
    parser.add_argument('--mode', '-m', required=False, default='1', help='Mode')
    parser.add_argument('--image', '-i', required=False, default='moonlanding.png')
    args = parser.parse_args()

    dir_path = os.path.dirname(os.path.realpath(__file__))
    img_path = os.path.join(dir_path, args.image)
    img = cv2.imread(img_path)

    if not img:
        print("Invalid path provided")
        return -1

    old_width = img.shape[1]
    old_height = img.shape[0]

    new_height = np.power(2, np.ceil(np.log2(old_height)))
    new_width = np.power(2, np.ceil(np.log2(old_width)))

    np.resize(img, (new_width, new_height))

    mode = 0
    try:
        mode = int(args.mode)
        if mode < 1 or mode > 4:
            raise ValueError("Invalid")
    except ValueError:
        print("Invalid entry for mode. Must be an integer from 1 to 4.")
        return -1

    if mode == 1:
        print()
    elif mode == 2:
        print()
    elif mode == 3:
        print()
    elif mode == 4:
        print()
    else:
        return -1


if __name__ == "__main__":
    input_array = np.array([[1, 2, 3], [2, 2, 2], [3, 4, 5]])
    a2 = naive_2d_dft(input_array)
    a3 = np.fft.fft2(input_array)
    print("CORRECT output ", a3)
    print("TY output ", a1)
    print("MY output ", a2)
