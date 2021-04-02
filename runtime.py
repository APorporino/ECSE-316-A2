"""
This file will be used to calculate the runtime complexity of FFT and Naive DFT.

It will store results in a file so that we do not have to run the experiments everytime.
"""
from fft import naive_2d_dft, fft_2d_dft
import numpy as np
import time


def generate_data():
    """ Note that to run Naive method for a 2d array of size 128 takes around 1 hour. """

    sizes = [32, 64]
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
    data_naive = []
    data_fft = []
    for i in range(num_runs):
        random_2d_array = np.random.rand(size, size)

        start_naive = time.time()
        print("Naive round: {}, size: {}".format(i, size))
        naive_2d_dft(random_2d_array)
        end_naive = time.time()

        start_fft = time.time()
        print("FFT round: {}, size: {}".format(i, size))
        fft_2d_dft(random_2d_array)
        end_fft = time.time()

        data_naive.append(end_naive - start_naive)
        data_fft.append(end_fft - start_fft)
    return data_naive, data_fft

if __name__ == '__main__':
    generate_data()