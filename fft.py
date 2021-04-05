import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.colors import LogNorm
import cv2
import math
import os
import argparse
import csv

smallest_size_array = 16
dft_coeff_array = [np.exp(-1j * 2 * np.pi * m / smallest_size_array) for m in range(smallest_size_array)]
inverse_dft_coeff_array = [np.exp(1j * 2 * np.pi * k / smallest_size_array) for k in range(smallest_size_array)]


def naive_dft(input_array):
    """
    Will take naive DFT of input 1 dimensional array.
    :param input_array: 1 dimensional array
    :return: Array representing the fourier transform of the input array
    """
    N = input_array.size
    output_array = np.zeros(N, dtype=np.complex_)
    for k in range(N):
        X = 0
        for n in range(N):
            xn = input_array[n]
            arg = -1j * 2 * math.pi * k * n / N
            X += xn * np.exp(arg)
        output_array[k] = X
    return output_array


def naive_inverse_dft(input_array):
    """
        ONLY FOR USE BY inverse_fft_dft. Will take naive IDFT of input 1 dimensional array.
        :param input_array: 1 dimensional array
        :return: Array representing the inverse fourier transform of the input array
        """
    N = input_array.size
    output_array = np.zeros(N, dtype=np.complex_)
    for k in range(N):
        X = 0
        for n in range(N):
            xn = input_array[n]
            arg = 1j * 2 * math.pi * k * n / N
            X += xn * np.exp(arg)
        output_array[k] = X
    return output_array


def naive_dft_precomputed(input_array):
    """
    Will take naive DFT of input 1 dimensional array of size smallest_size_array. Used by fft_dft only.
    :param input_array: 1 dimensional array
    :return: Array representing the fourier transform of the input array
    """

    N = input_array.size
    output_array = np.zeros(N, dtype=np.complex_)
    for k in range(N):
        coeff = np.power(dft_coeff_array, np.full(N, k))
        temp = np.multiply(input_array, coeff)
        X = np.sum(temp)
        output_array[k] = X
    return np.array(output_array)


def naive_inverse_dft_precomputed(input_array):
    """
    Will take naive DFT of input 1 dimensional array of size smallest_size_array. Used by fft_dft only.
    :param input_array: 1 dimensional array
    :return: Array representing the fourier transform of the input array
    """

    N = input_array.size
    output_array = np.zeros(N, dtype=np.complex_)
    for k in range(N):
        coeff = np.power(inverse_dft_coeff_array, np.full(N, k))
        temp = np.multiply(input_array, coeff)
        X = np.sum(temp)
        output_array[k] = X
    # size_coeff = np.full(N, 1 / N)
    # return np.multiply(size_coeff, np.array(output_array))
    return output_array


def naive_dft_k(input_array, k):
    """
    This function will use the naive method to calculate the DFT but only for 1 k value.
    :param input_array:
    :param k: Represents the index to use for the DFT function
    :return: Returns DFT of the input array with k = {k}
    """
    N = input_array.size
    X = 0
    for n in range(N):
        xn = input_array[n]
        arg = -1j * 2 * math.pi * k * n / N
        X += xn * np.exp(arg)
    return X


def naive_2d_dft(input_array):
    """
    Will take naive DFT of input 2 dimensional array.
    :param input_array: 2 dimensional array.
    :return: Array representing the fourier transform of the input array
    """
    R, C = input_array.shape
    output_array = np.zeros((R, C), dtype=np.complex_)
    for k in range(R):
        for l in range(C):
            output_array[k][l] = naive_2d_dft_k_l(input_array, k, l)
    return np.array(output_array)


def naive_2d_dft_k_l(input_array, k, l):
    """
    This function will use the naive method to calculate the DFT
    of a 2d array but only for 1 (k,l) value.
    :param input_array: 2 dimensional array.
    :param k:
    :param l:
    :return:
    """
    R, C = input_array.shape
    input_array_transpose = input_array.transpose()
    temp = np.zeros(C, dtype=np.complex_)
    for i in range(C):
        temp[i] = naive_dft_k(input_array_transpose[i], k)
    return naive_dft_k(temp, l)


def fft_dft(input_array):
    """
    Will take the FFT of input array.
    :param input_array: A 1d array with size that is a power of 2.
    :return: Array that represents the DFT of the input array.
    """

    size = input_array.size

    if size == smallest_size_array:
        return naive_dft_precomputed(input_array)
    else:
        x_even = fft_dft(input_array[0::2])
        x_odd = fft_dft(input_array[1::2])

        arg = -1j * 2 * np.pi / size

        coeff = np.exp(arg * np.arange(size))

        odd_with_coeff_first_half = coeff[:size // 2] * x_odd
        odd_with_coeff_second_half = coeff[size // 2:] * x_odd

        return np.concatenate([x_even + odd_with_coeff_first_half, x_even + odd_with_coeff_second_half])


def inverse_fft_dft(input_array):
    def inner_inverse_fft_dft(input_array):
        """
        Will take the inverse fft of input array.
        :param input_array: A 1d array with size that is a power of 2.
        :return: Array that represents the IDFT of the input array.
        """

        size = input_array.size

        if size == smallest_size_array:
            return naive_inverse_dft_precomputed(input_array)
        elif size < smallest_size_array:
            return naive_inverse_dft(input_array)
        else:
            x_even = inner_inverse_fft_dft(input_array[0::2])
            x_odd = inner_inverse_fft_dft(input_array[1::2])

            arg = 1j * 2 * np.pi / size

            coeff = np.exp(arg * np.arange(size))

            odd_with_coeff_first_half = coeff[:size // 2] * x_odd
            odd_with_coeff_second_half = coeff[size // 2:] * x_odd

            return np.concatenate([x_even + odd_with_coeff_first_half, x_even + odd_with_coeff_second_half])

    return (1 / input_array.size) * inner_inverse_fft_dft(input_array)


def fft_2d(input_array):
    input_array = np.apply_along_axis(fft_dft, 1, input_array)
    input_array = np.apply_along_axis(fft_dft, 0, input_array)

    return input_array


def inverse_fft_2d(input_array):
    input_array = np.apply_along_axis(inverse_fft_dft, 1, input_array)
    input_array = np.apply_along_axis(inverse_fft_dft, 0, input_array)

    return input_array


def first_mode(img):
    print("Starting FFT for the image - this will take around 90 seconds")
    img_fft = fft_2d(img).real
    print("FFT completed")

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
    fig.suptitle('Original Image and Fourier Transform')
    ax[0].imshow(img, cmap='gray')
    ax[1].imshow(img_fft, norm=LogNorm(), cmap='gray')
    plt.show()


def second_mode(img):
    """
    This method is for the second mode and is used to denoise the input img. It will graph the
    original image as well as the denoised one.
    :param img:
    :return:
    """
    print("Starting FFT for the image - this will take around 90 seconds")
    img_fft = fft_2d(img)
    print("FFT completed")
    M, N = img_fft.shape

    removed_high_freq = np.copy(img_fft)

    k_cut_off = int(0.10 * M)
    l_cut_off = int(0.10 * N)

    for k in range(k_cut_off, M - k_cut_off):
        for l in range(l_cut_off, N - l_cut_off):
            removed_high_freq[k, l] = 0

    denoised_img = inverse_fft_2d(removed_high_freq).real

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
    fig.suptitle('Original Image and Denoised Image')
    ax[0].imshow(img, cmap='gray')
    ax[1].imshow(denoised_img, cmap='gray')
    plt.show()
    print("Number of total fourier coefficients in original image: {}".format(M * N))
    print("Number of nonzero Fourier coefficients in image 1: {}".format(.9 * M * .9 * N))
    return denoised_img


def third_mode_compression_threshold_option(img):
    """
    This method was used to compress the input img using the threshold algorithm idea.
    As with the other compression algorithm, it saves the fourier coefficients for each level of compression
    in a csv file.
    :param img:
    :return:
    """
    img_fft = fft_2d(img)
    R, C = img_fft.shape

    img_compressed_fft = [np.copy(img_fft) for i in range(5)]
    compression_levels = [.15, .30, .50, .70, .95]

    for i in range(5):
        minimum = img_compressed_fft[i].min().real
        maximum = img_compressed_fft[i].max().real
        t = calculate_threshold(minimum, maximum, compression_levels[i])
        img_compressed_fft[i] = compress_threshold(img_compressed_fft[i], t)
    compression_15 = inverse_fft_2d(img_compressed_fft[0]).real
    compression_30 = inverse_fft_2d(img_compressed_fft[1]).real
    compression_50 = inverse_fft_2d(img_compressed_fft[2]).real
    compression_70 = inverse_fft_2d(img_compressed_fft[3]).real
    compression_95 = inverse_fft_2d(img_compressed_fft[4]).real
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
    """
    This method will be used to calculate the threshold value.
    :param min_mag: Minimum magnitude of an element in the array
    :param max_mag: Maximum magnitude of an element in the array
    :param percentile: Percentage of compression
    :return:
    """
    difference = max_mag - min_mag
    if (min_mag < 0):
        difference = max_mag + (-1) * min_mag
    return percentile * difference + min_mag


def compress_threshold(img, threshold):
    """
    This method will actually compress teh image given the threshold.
    :param img:
    :param threshold:
    :return:
    """
    R, C = img.shape
    for k in range(R):
        for l in range(C):
            if img[k][l] > threshold:
                img[k][l] = 0
    return img


def third_mode(img):
    """
    Actual compression algorithm implemented. It uses same the technique for denoising.
    It will also save the fourier coefficients for each level of compression, as well as a png
    file of that compressed image.
    :param img:
    :return:
    """
    print("Starting FFT for the image - this will take around 90 seconds")
    img_fft = fft_2d(img)
    print("FFT completed")
    R, C = img_fft.shape

    img_compressed_fft = [np.copy(img_fft) for i in range(5)]
    compression_levels = [.15, .30, .50, .70, .95]

    for i in range(5):
        img_compressed_fft[i] = compress(img_compressed_fft[i], compression_levels[i])
        print("Compressed image with level: {}".format(compression_levels[i]))
        # saves fft as csv. Huge file
        np.savetxt("fft_compression_level_{}.csv".format(compression_levels[i]), img_compressed_fft[i], delimiter=",")
    print("Reverting image to time domain for compression level: {}. This will take around 90 seconds".format(compression_levels[0]))
    compression_15 = inverse_fft_2d(img_compressed_fft[0]).real
    print("Reverting image to time domain for compression level: {}. This will take around 90 seconds".format(compression_levels[1]))
    compression_30 = inverse_fft_2d(img_compressed_fft[1]).real
    print("Reverting image to time domain for compression level: {}. This will take around 90 seconds".format(compression_levels[2]))
    compression_50 = inverse_fft_2d(img_compressed_fft[2]).real
    print("Reverting image to time domain for compression level: {}. This will take around 90 seconds".format(compression_levels[3]))
    compression_70 = inverse_fft_2d(img_compressed_fft[3]).real
    print("Reverting image to time domain for compression level: {}. This will take around 90 seconds".format(compression_levels[4]))
    compression_95 = inverse_fft_2d(img_compressed_fft[4]).real

    # #saves data as png
    data = Image.fromarray(compression_15)
    data = data.convert("L")
    data.save('15.png')
    data = Image.fromarray(compression_30)
    data = data.convert("L")
    data.save('30.png')
    data = Image.fromarray(compression_50)
    data = data.convert("L")
    data.save('50.png')
    data = Image.fromarray(compression_70)
    data = data.convert("L")
    data.save('70.png')
    data = Image.fromarray(compression_95)
    data = data.convert("L")
    data.save('95.png')

    # plot data
    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(10, 5))
    fig.suptitle('Original and Compressed Images')
    ax[0][0].imshow(img, cmap='gray')
    ax[0][1].imshow(compression_15, cmap='gray')
    ax[0][2].imshow(compression_30, cmap='gray')
    ax[1][0].imshow(compression_50, cmap='gray')
    ax[1][1].imshow(compression_70, cmap='gray')
    ax[1][2].imshow(compression_95, cmap='gray')
    plt.show()
    print("Number of total fourier coefficients in original image: {}".format(R * C))
    print("Number of nonzero Fourier coefficients in image 1: {}".format(R * C * (1 - .15)))
    print("Number of nonzero Fourier coefficients in image 2: {}".format(R * C * (1 - .30)))
    print("Number of nonzero Fourier coefficients in image 3: {}".format(R * C * (1 - .50)))
    print("Number of nonzero Fourier coefficients in image 4: {}".format(R * C * (1 - .70)))
    print("Number of nonzero Fourier coefficients in image 5: {}".format(R * C * (1 - .95)))


def compress(img, percentage_removed):
    """
    This method will be used to actually compress the img given the level of compression.
    :param img:
    :param percentage_removed:
    :return:
    """
    axis_percentage_removed = math.sqrt(percentage_removed)
    R, C = img.shape

    starting_point_rows = int(np.floor((R - axis_percentage_removed * R) / 2))
    starting_point_column = int(np.floor((C - axis_percentage_removed * C) / 2))

    for k in range(starting_point_rows, R - starting_point_rows):
        for l in range(starting_point_column, C - starting_point_column):
            img[k, l] = 0
    return img


def fourth_mode():
    """
    The fourth mode will simply read the data from two files:
        "runtime_data_fft.csv" and "runtime_data_naive.csv"
    and graph the runtimes using matplotlib. It assumes all data is there.

    To actually generate the runtime data, we must run the runtime.py file in this same dir.
    :return:
    """
    try:
        y_axis = [32, 64, 128, 256, 512, 1024]
        fft_points = []
        fft_std = []

        naive_points = []
        naive_std = []

        # see report to see why these points are used.
        naive_points_estimations = [28548, 879947878, 8.36005472344308e+17, 7.545942018890406e+35]
        naive_std_estimations = [0, 0, 0, 0]
        with open("runtime_data_fft.csv") as csvfile:
            reader = csv.reader(csvfile, delimiter=",")
            for row in reader:
                fft_points.append(float(row[0]))
                fft_std.append(float(row[1]))
        with open("runtime_data_naive.csv") as csvfile:
            reader = csv.reader(csvfile, delimiter=",")
            for row in reader:
                naive_points.append(float(row[0]))
                naive_std.append(float(row[1]))
            naive_points += naive_points_estimations
            naive_std += naive_std_estimations

        plt.errorbar(y_axis, fft_points,yerr = fft_std, label="FFT")
        # remove the line below to see the unskewed FFT runtime graph.
        plt.errorbar(y_axis, naive_points, yerr = naive_std, label="Naive DFT")
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
