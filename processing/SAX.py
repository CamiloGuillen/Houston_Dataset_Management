import numpy as np
import matplotlib.pyplot as plt
import string

from sklearn.preprocessing import normalize
from scipy.stats import norm


def get_breakpoints(a):
    """
    Get the break points from N(0, 1) according to a
    :param a: Alphabet size
    :return: Sort list of betas
    """
    betas = np.arange(0, a, 1) / a

    return norm.ppf(betas)


def get_alphabet(a):
    """
    Get the list of possibles symbols
    :param a: Alphabet size
    :return: String with the 'a' symbols
    """
    alpha = string.ascii_lowercase

    return alpha[:a]


class SAX:
    """
    Symbolic Aggregate Approximation
    """

    def __init__(self, time_series, w, a):
        """
        Obtain the SAX symbolization for a time series
        :param time_series: list with the data of the time series
        :param w: string length
        :param a: alphabet size
        """
        self.C = self.normalize(time_series)
        self.C_bar = self.PAA(w)
        self.C_hat = self.discretization(a, w)
        print(self.C_hat)
        plt.figure(figsize=(16, 4))
        plt.title("Piecewise Aggregate Approximation", size=14, loc="right")
        plt.plot(range(1, 16), self.C, marker="o", linewidth=0.5)
        plt.step([1, 3, 5, 7, 9, 11, 13, 15], np.insert(self.C_bar, 0, self.C_bar[0]),
                 color="red", linestyle=":")
        plt.xticks(range(1, 16))
        plt.xlabel("Index")
        plt.ylabel("TS Value")
        plt.show()

    @staticmethod
    def normalize(data):
        """
        Time series normalization
        :param data: List with data
        :return: numpy array with data normalized
        """
        data_norm = normalize([data])

        return data_norm[0]

    def PAA(self, w):
        """
        Dimensionality reduction via PAA
        :param w: Number of PAA segments
        :return: numpy array with C bar
        """
        n = self.C.shape[0]
        C_bar = np.zeros(w)
        if n % w == 0:
            step = n // w
            for i in range(n):
                index = i // step
                np.add.at(C_bar, index, self.C[i])
            C_bar /= step
        else:
            for i in range(w * n):
                i_n = i // w
                i_w = i // n
                np.add.at(C_bar, i_w, self.C[i_n])
            C_bar = C_bar / n

        return C_bar

    def discretization(self, a, w):
        """
        Discretization of the C bar data
        :return: List with C hat
        """
        betas = get_breakpoints(a)
        alpha = get_alphabet(a)
        C_hat = []
        for i in range(w):
            C_bar_i = self.C_bar[i]
            if C_bar_i >= 0:
                j = a - 1
                while j > 0 and betas[j] >= C_bar_i:
                    j -= 1
                C_hat_i = alpha[j]
            else:
                j = 1
                while j < a and betas[j] <= C_bar_i:
                    j += 1
                C_hat_i = alpha[j]
            C_hat.append(C_hat_i)

        return C_hat
