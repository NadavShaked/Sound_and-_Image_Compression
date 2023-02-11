import numpy as np
import matplotlib.pyplot as plt
import librosa


def SNR(in_vec, out_vec):
    noise = np.std(np.abs(in_vec - out_vec)) ** 2
    if noise == 0:
        return np.inf
    return 10 * np.log10(np.std(in_vec) ** 2 / noise)


def task6():
    x, sr = librosa.load("vega.wav")
    res = np.zeros(15)

    for idx, n in enumerate(range(1, 16)):
        xq = np.floor((x + 1) * 2 ** (n - 1))
        xq = xq / 2 ** (n - 1)
        xq = xq - (2 ** n - 1) / 2 ** n

        res[idx] = SNR(x, xq)

    return res


def main():
    res = task6()
    plt.plot(res)
    plt.ylabel('SNR')
    plt.xlabel('n')
    plt.xticks(range(0, 15), range(1, 16))
    plt.show()


if __name__ == '__main__':
    main()