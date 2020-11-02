import numpy as np

from math import ceil


class DataWindowing:
    def __init__(self, signal, labels, window_size, sample_rate):
        self.signal = signal
        self.labels = labels
        self.n_samples = ceil(window_size * sample_rate)
        self.len = int(len(self.signal) - self.n_samples + 1)
        self.idx = 0

    def __len__(self):
        return self.len

    def __iter__(self):
        return self

    def __next__(self):
        if self.idx <= self.len:
            indexes = list(range(self.idx, min(self.idx + self.n_samples, len(self.signal))))
            self.idx += 1
        else:
            raise StopIteration

        X = np.empty((len(indexes), 6))
        for i in range(len(indexes)):
            X[i, :] = self.signal[indexes[i]]

        labels = [self.labels[i] for i in indexes]

        y_tmp = np.unique(labels)
        y_tmp = [int(i) for i in y_tmp]

        if len(y_tmp) > 1:
            y = 3
        else:
            y = y_tmp[0]

        return X, np.array([y]*len(X))
