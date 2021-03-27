import numpy as np
import matplotlib
import cv2
import math

def naive_dft(input):
    N = input.size
    output = np.zeros(N, dtype=np.complex_)
    for k in range(N):
        X = 0
        for n in range(N):
            xn = input[n]
            arg = 2 * math.pi * k * n / N
            X += xn * (-1j * math.sin(arg))
        output[k] = X
    return np.array(output)

def naive_2d_dft(input):
    N,M = input.shape
    output = np.zeros((N,M), dtype=np.complex_)
    for k in range(M):
        for l in range(N):
            Fkl = 0
            for n in range(N):
                Fk = 0
                for m in range(M):
                    fmn = input[m][n]
                    arg = 2 * math.pi * k * m / M
                    Fk += fmn*(-1j*math.sin(arg))
                arg = 2 * math.pi * l * n / N
                Fkl += Fk * (-1j * math.sin(arg))
            output[k][l] = Fkl
    return np.array(output)

if __name__ == "__main__":
    input = np.array([1,2,3])
    output = naive_dft(input)
    input2 = np.array([[1,2,3],[1,2,3],[1,2,3]])
    ouput2 = naive_2d_dft(input2)
    i = 5








