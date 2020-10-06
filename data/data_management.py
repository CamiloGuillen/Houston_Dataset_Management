import pandas as pd
import numpy as np


class Data_Management:
    """
    Return the data required
    """
    def __init__(self, data, label):
        self.original_data = data
        self.original_label = label

    def by_Ground_Event(self, ground_event=0, save=False, file_name="data"):
        """
        Take the data then create and save the pandas data frame with the X, Y and Z information of the interest ground
         event
        :param ground_event: Int with ground event
        :param save: boolean
        :param file_name: string
        :return: Pandas data frame
        """
        temp = []
        for i in range(len(self.original_label)):
            if self.original_label[i][0] == ground_event:
                temp.append(np.insert(self.original_data[i], [3], [self.original_label[i][1], self.original_label[i][0]]
                                      ))
            else:
                continue
        df = pd.DataFrame(temp, index=range(len(temp)), columns=["X", "Y", "Z", "GaitCycle", "GroundEvent"])
        if save:
            df.to_csv(file_name + ".csv", index=False)
        return df

    def clean_Missing_Labels(self, save=False, file_name="data"):
        """
        Erase the missing labels of the data
        :param save: boolean
        :param file_name: string
        :return: Pandas Data Frame
        """
        temp = []
        for i in range(len(self.original_label)):
            if not self.original_label[i][0] == 5:
                temp.append(np.insert(self.original_data[i], [3], [self.original_label[i][1], self.original_label[i][0]]
                                      ))
            else:
                continue
        df = pd.DataFrame(temp, index=range(len(temp)), columns=["X", "Y", "Z", "GaitCycle", "GroundEvent"])
        if save:
            df.to_csv(file_name + ".csv", index=False)
        return df
