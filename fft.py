import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.colors import LogNorm
import cv2
import math
import os
import argparse
import csv
from datetime import datetime

smallest_size_array = 4
dft_coeff_array = [np.exp(-1j * 2 * np.pi * m / smallest_size_array) for m in range(smallest_size_array)]
inverse_dft_coeff_array = [np.exp(1j * 2 * np.pi * k / smallest_size_array) for k in range(smallest_size_array)]


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
    if size > smallest_size_array:
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
        if (N < 16):
            arg = -1j * 2 * math.pi * k * n / N
            a = np.exp(arg)
        else:
            a = np.power(dft_coeff_array[n], k)
        X += xn * a
    return X


def naive_inverse_dft_n_precomputed(input_array, n):
    N = input_array.size
    X = 0
    for k in range(N):
        xn = input_array[k]
        a = np.power(inverse_dft_coeff_array[k], n)
        X += xn * a
    return X


def outer_inverse_fft_dft(input_array):
    size = input_array.size
    output_array = np.zeros(size, dtype=complex)

    for k in range(size):
        output_array[k] = (1 / size) * inner_inverse_fft_dft(input_array, k)
    return output_array


def inner_inverse_fft_dft(input_array, n):
    size = input_array.size
    if size > smallest_size_array:
        even = inner_inverse_fft_dft(input_array[::2], n)
        odd = inner_inverse_fft_dft(input_array[1::2], n)
        return even + np.exp(1j * 2 * np.pi * n / size) * odd
    else:
        return naive_inverse_dft_n_precomputed(input_array, n)


def fft_2d_dft(input_array):
    R, C = input_array.shape
    output_array = np.zeros((R, C), dtype=np.complex_)

    def fft_2d_dft_k_l(input_array, k, l):
        R, C = input_array.shape
        input_array_transpose = input_array.transpose()

        temp = np.zeros(C, dtype=np.complex_)

        for i in range(C):
            temp[i] = inner_fft_dft(input_array_transpose[i], k)

        return inner_fft_dft(temp, l)

    # print("R, C:", R, C)
    for k in range(R):
        print("K: ", k)
        for l in range(C):
            # now = datetime.now()
            #
            # current_time = now.strftime("%H:%M:%S")
            # print("Current Time =", current_time)
            # print("L:", l)
            output_array[k][l] = fft_2d_dft_k_l(input_array, k, l)

    return np.array(output_array)


def inverse_fft_2d_dft(input_array):
    R, C = input_array.shape
    output_array = np.zeros((R, C), dtype=np.complex_)

    def inverse_fft_2d_dft_m_n(input_array, m, n):
        R, C = input_array.shape
        input_array_transpose = input_array.transpose()

        temp = np.zeros(C, dtype=np.complex_)

        for i in range(C):
            temp[i] = inner_inverse_fft_dft(input_array_transpose[i], m)

        return (1 / (R * C)) * inner_inverse_fft_dft(temp, n)

    for m in range(R):
        for n in range(C):
            output_array[m][n] = inverse_fft_2d_dft_m_n(input_array, m, n)
    return np.array(output_array)


def first_mode(img):
    img_fft = np.fft.fft2(img).real

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
    fig.suptitle('Original Image and Fourier Transform')
    ax[0].imshow(img)
    ax[1].imshow(img_fft, norm=LogNorm())
    plt.show()


def second_mode(img):
    img_fft = np.fft.fft2(img)
    M, N = img_fft.shape

    removed_high_freq = np.copy(img_fft)

    k_cut_off = int(0.10 * M)
    l_cut_off = int(0.10 * N)

    for k in range(k_cut_off, M - k_cut_off):
        for l in range(l_cut_off, N - l_cut_off):
            removed_high_freq[k, l] = 0

    denoised_img = np.fft.ifft2(removed_high_freq).real

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
    fig.suptitle('Original Image and Denoised Image')
    ax[0].imshow(img)
    ax[1].imshow(denoised_img)
    plt.show()
    print("Number of total fourier coefficients in original image: {}".format(M * N))
    print("Number of nonzero Fourier coefficients in image 1: {}".format(.9*M * .9*N))
    return denoised_img


def third_mode_compression_threshold_option(img):
    img_fft = np.fft.fft2(img)
    R, C = img_fft.shape

    img_compressed_fft = [np.copy(img_fft) for i in range(5)]
    compression_levels = [.15, .30, .50, .70, .95]

    for i in range(5):
        minimum = img_compressed_fft[i].min().real
        maximum = img_compressed_fft[i].max().real
        t = calculate_threshold(minimum, maximum, compression_levels[i])
        img_compressed_fft[i] = compress_threshold(img_compressed_fft[i], t)
    compression_15 = np.fft.ifft2(img_compressed_fft[0]).real
    compression_30 = np.fft.ifft2(img_compressed_fft[1]).real
    compression_50 = np.fft.ifft2(img_compressed_fft[2]).real
    compression_70 = np.fft.ifft2(img_compressed_fft[3]).real
    compression_95 = np.fft.ifft2(img_compressed_fft[4]).real
    # plot data
    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(10, 5))
    fig.suptitle('Original and Compressed Images')
    ax[0][0].imshow(img)
    ax[0][1].imshow(compression_15)
    ax[0][2].imshow(compression_30)
    ax[1][0].imshow(compression_50)
    ax[1][1].imshow(compression_70)
    ax[1][2].imshow(compression_95)
    plt.show()


def calculate_threshold(min_mag, max_mag, percentile):
    difference = max_mag - min_mag
    if (min_mag < 0):
        difference = max_mag + (-1) * min_mag
    return percentile * difference + min_mag


def compress_threshold(img, threshold):
    R, C = img.shape
    for k in range(R):
        for l in range(C):
            if img[k][l] > threshold:
                img[k][l] = 0
    return img


def third_mode(img):
    img_fft = np.fft.fft2(img)
    R, C = img_fft.shape

    img_compressed_fft = [np.copy(img_fft) for i in range(5)]
    compression_levels = [.15, .30, .50, .70, .95]

    for i in range(5):
        img_compressed_fft[i] = compress(img_compressed_fft[i], compression_levels[i])

        # saves fft as csv. Huge file
        np.savetxt("fft_compression_level_{}.csv".format(compression_levels[i]), img_compressed_fft[i], delimiter=",")
    compression_15 = np.fft.ifft2(img_compressed_fft[0]).real
    compression_30 = np.fft.ifft2(img_compressed_fft[1]).real
    compression_50 = np.fft.ifft2(img_compressed_fft[2]).real
    compression_70 = np.fft.ifft2(img_compressed_fft[3]).real
    compression_95 = np.fft.ifft2(img_compressed_fft[4]).real

    # #saves data as png
    # data = Image.fromarray(compression_15)
    # data = data.convert("L")
    # data.save('15.png')
    # data = Image.fromarray(compression_30)
    # data = data.convert("L")
    # data.save('30.png')
    # data = Image.fromarray(compression_50)
    # data = data.convert("L")
    # data.save('50.png')
    # data = Image.fromarray(compression_70)
    # data = data.convert("L")
    # data.save('70.png')
    # data = Image.fromarray(compression_95)
    # data = data.convert("L")
    # data.save('95.png')

    # plot data
    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(10, 5))
    fig.suptitle('Original and Compressed Images')
    ax[0][0].imshow(img)
    ax[0][1].imshow(compression_15)
    ax[0][2].imshow(compression_30)
    ax[1][0].imshow(compression_50)
    ax[1][1].imshow(compression_70)
    ax[1][2].imshow(compression_95)
    plt.show()
    print("Number of total fourier coefficients in original image: {}".format(R * C))
    print("Number of nonzero Fourier coefficients in image 1: {}".format(R * C * (1 - .15)))
    print("Number of nonzero Fourier coefficients in image 2: {}".format(R * C * (1 - .30)))
    print("Number of nonzero Fourier coefficients in image 3: {}".format(R * C * (1 - .50)))
    print("Number of nonzero Fourier coefficients in image 4: {}".format(R * C * (1 - .70)))
    print("Number of nonzero Fourier coefficients in image 5: {}".format(R * C * (1 - .95)))


def compress(img, percentage_removed):
    axis_percentage_removed = math.sqrt(percentage_removed)
    R, C = img.shape

    starting_point_rows = int(np.floor((R - axis_percentage_removed * R) / 2))
    starting_point_column = int(np.floor((C - axis_percentage_removed * C) / 2))

    for k in range(starting_point_rows, R - starting_point_rows):
        for l in range(starting_point_column, C - starting_point_column):
            img[k, l] = 0
    return img


def fourth_mode():
    try:
        y_axis = [32, 64]
        fft_points = [6]
        naive_points = [1]
        with open("runtime_data_fft.csv") as csvfile:
            reader = csv.reader(csvfile, delimiter=",")
            for row in reader:
                fft_points.append(float(row[0]))
        with open("runtime_data_naive.csv") as csvfile:
            reader = csv.reader(csvfile, delimiter=",")
            for row in reader:
                naive_points.append(float(row[0]))

        plt.plot(y_axis, fft_points, label="FFT")
        plt.plot(y_axis, naive_points, label="Naive DFT")
        plt.xlabel('Size of rows and columns of input matrix')
        plt.ylabel('Mean runtime for 10 experiments (seconds)')
        plt.title('Runtime Stats Plot')
        plt.legend()
        plt.show()
    except FileNotFoundError:
        print("File 'runtime_data' does not exist")


def main():
    parser = argparse.ArgumentParser(description='Arguments for FFT program')
    parser.add_argument('--mode', '-m', required=False, default='1', help='Mode')
    parser.add_argument('--image', '-i', required=False, default='moonlanding.png')
    args = parser.parse_args()

    dir_path = os.path.dirname(os.path.realpath(__file__))
    img_path = os.path.join(dir_path, args.image)
    with Image.open(img_path) as img_file:
        img = np.array(img_file)

    if img is None:
        print("Invalid path provided")
        return -1

    old_width = img.shape[1]
    old_height = img.shape[0]

    new_height = int(np.power(2, np.ceil(np.log2(old_height))))
    new_width = int(np.power(2, np.ceil(np.log2(old_width))))

    img = cv2.resize(img, (new_width, new_height))

    try:
        mode = int(args.mode)
        if mode < 1 or mode > 4:
            raise ValueError("Invalid")
    except ValueError:
        print("Invalid entry for mode. Must be an integer from 1 to 4.")
        return -1

    if mode == 1:
        first_mode(img)
    elif mode == 2:
        print('Mode 2')
        second_mode(img)
    elif mode == 3:
        print('Mode 3')
        third_mode(img)
    elif mode == 4:
        print('Mode 4')
        fourth_mode()
    else:
        return -1


if __name__ == "__main__":
    main()
