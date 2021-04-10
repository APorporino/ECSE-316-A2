"""
This file will be used to calculate the runtime complexity of FFT and Naive DFT.

It will store results in a file so that we do not have to run the experiments everytime.
"""
from fft import naive_2d_dft, fft_2d
import numpy as np
import time


def generate_data():
    """
    Running this function will generate all data needed for our mode 4 method.

    It will save the data in two csv files. One for naive and one for FFT
    """

    sizes = [32, 64, 128, 256, 512, 1024]
    data_naive = []
    data_fft = []
    for size in sizes:
        print("Starting runtime calculations for size {}".format(size))
        data = runtime_calculation(size, 10)
        mean_naive = sum(data[0]) / len(data[0])
        std_naive = np.std(data[0])
        mean_fft = sum(data[1]) / len(data[1])
        std_fft = np.std(data[1])
        data_naive.append([mean_naive, std_naive])
        data_fft.append([mean_fft, std_fft])
    np.savetxt("runtime_data_naive.csv", data_naive, delimiter=",")
    np.savetxt("runtime_data_fft.csv", data_fft, delimiter=",")


def runtime_calculation(size, num_runs):
    """
    This function will run {num_runs} amount of experiments on an square matrix of size {size}x{size}.
    The experiments time the runtime of the naive DFT, and the FFT where the input is the array.

    :param size: Size of the rows and columns of the random array to test the methods.
    :param num_runs: Number of times the experiment should run
    :return: Array with two elements.
            The first element is the array containing the data for the naive DFT
            The second element is the array containing the data for the FFT
    """
    data_naive = []
    data_fft = []
    for i in range(num_runs):
        random_2d_array = np.random.rand(size, size)

        print("Naive round: {}, size: {}".format(i, size))
        start_naive = time.time()
        naive_2d_dft(random_2d_array)
        end_naive = time.time()

        print("FFT round: {}, size: {}".format(i, size))
        start_fft = time.time()
        fft_2d(random_2d_array)
        end_fft = time.time()

        data_naive.append(end_naive - start_naive)
        data_fft.append(end_fft - start_fft)
    return data_naive, data_fft


if __name__ == '__main__':
    generate_data()