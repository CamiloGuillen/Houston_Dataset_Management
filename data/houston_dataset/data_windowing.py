import numpy as np
import pandas as pd

from math import ceil


class DataWindowing:
    def __init__(self, trial, window_size):
        self.signal = trial[0]
        self.ground_event = trial[1]
        self.gait_event = trial[2]
        sample_rate = trial[3]
        self.n_samples = ceil(sample_rate * window_size)

    def segment_by(self, gait_event):
        idx = np.where(self.gait_event == gait_event)[0]

        event_ids = [idx[0]]
        tmp = [idx[0]]
        for i in range(1, len(idx)):
            if idx[i] == tmp[-1] + 1:
                tmp.append(idx[i])
            else:
                tmp = [idx[i]]
                event_ids.append(idx[i])

        windows, labels = self.__get_windows(event_ids)
        win_features = self.__extract_features(windows, labels)
        # print(win_features)

        return win_features

    def __get_windows(self, event_ids):
        windows = []
        labels = []
        for idx in event_ids:
            start_idx = idx - self.n_samples
            end_idx = idx
            data_window = self.signal[start_idx:end_idx]
            labels_window = self.ground_event[start_idx:end_idx]
            if len(data_window) > 0:
                windows.append(data_window)
                labels.append(labels_window)

        return windows, labels

    @staticmethod
    def __extract_features(windows, labels):
        min_RK = []
        max_RK = []
        mean_RK = []
        std_RK = []
        end_value_RK = []
        min_RA = []
        max_RA = []
        mean_RA = []
        std_RA = []
        end_value_RA = []
        min_RH = []
        max_RH = []
        mean_RH = []
        std_RH = []
        end_value_RH = []

        for win in windows:
            # Window data
            RK_win_data = [i for i in win[:, 0]]
            RA_win_data = [i for i in win[:, 1]]
            RH_win_data = [i for i in win[:, 2]]

            # Windows features
            min_RK.append(np.min(RK_win_data))
            max_RK.append(np.max(RK_win_data))
            mean_RK.append(np.mean(RK_win_data))
            std_RK.append(np.std(RK_win_data))
            end_value_RK.append(RK_win_data[-1])
            min_RA.append(np.min(RA_win_data))
            max_RA.append(np.max(RA_win_data))
            mean_RA.append(np.mean(RA_win_data))
            std_RA.append(np.std(RA_win_data))
            end_value_RA.append(RA_win_data[-1])
            min_RH.append(np.min(RH_win_data))
            max_RH.append(np.max(RH_win_data))
            mean_RH.append(np.mean(RH_win_data))
            std_RH.append(np.std(RH_win_data))
            end_value_RH.append(RH_win_data[-1])

        # Assign a label
        ground_event_labels = []
        for label in labels:
            classes, freq = np.unique(label, return_counts=True)
            ground_event_labels.append(classes[0])

        features = pd.DataFrame()
        features["MIN_RK"] = min_RK
        features["MAX_RK"] = max_RK
        features["MEAN_RK"] = mean_RK
        features["STD_DEV_RK"] = std_RK
        features["END_VALUE_RK"] = end_value_RK
        features["MIN_RA"] = min_RA
        features["MAX_RA"] = max_RA
        features["MEAN_RA"] = mean_RA
        features["STD_DEV_RA"] = std_RA
        features["END_VALUE_RA"] = end_value_RA
        features["MIN_RH"] = min_RH
        features["MAX_RH"] = max_RH
        features["MEAN_RH"] = mean_RH
        features["STD_DEV_RH"] = std_RH
        features["END_VALUE_RH"] = end_value_RH
        features["GROUND_EVENT"] = ground_event_labels

        return features
