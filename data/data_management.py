import pandas as pd
import numpy as np

from sklearn.utils import resample


class Data_Management:
    """
    Return the data required
    """
    def __init__(self, data, label, ground_event=None):
        self.original_data = data
        self.original_label = label
        self.ground_event = ground_event
        self.df = None

    def clean_Missing_Labels(self):
        """
        Erase the missing labels of the data and separate by ground event
        :return: Pandas Data Frame
        """
        temp = []
        for i in range(len(self.original_label)):
            if not self.original_label[i][0] == 5:
                if self.ground_event is None:
                    temp.append(np.insert(self.original_data[i], [3],
                                          [self.original_label[i][1], self.original_label[i][0]]))
                else:
                    if self.original_label[i][0] == self.ground_event:
                        temp.append(np.insert(self.original_data[i], [3], [self.original_label[i][1], 1]))
                    else:
                        temp.append(np.insert(self.original_data[i], [3], [self.original_label[i][1], 0]))
            else:
                continue
        self.df = pd.DataFrame(temp, index=range(len(temp)), columns=["X", "Y", "Z", "GaitCycle", "GroundEvent"])

        return self.df

    def balance_classes(self):
        # Separate majority and minority classes
        df_majority = self.df[self.df["GroundEvent"] == self.df["GroundEvent"].value_counts().keys()[0]]
        df_minority = self.df[self.df["GroundEvent"] == self.df["GroundEvent"].value_counts().keys()[1]]

        # Down sample majority class
        df_down_sampled = resample(df_majority, replace=False, n_samples=len(df_minority), random_state=123)

        # Combine minority class with down sampled majority class
        df_balanced = pd.concat([df_down_sampled, df_minority])

        return df_balanced
