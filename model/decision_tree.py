import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from data_loader import DataLoader
from data.data_windowing import DataWindowing
from tqdm import tqdm
from sklearn import tree


class DecisionTree:
    def __init__(self, ground_event=None):
        """
        :param ground_event: Int with ground event
        """
        self.model = None
        self.ground_event = ground_event

    def LOSO_train(self, train_data_info):
        """
        :param train_data_info: Pandas Data Frame with train information
        :return: Trained model
        """
        # Leave One Subject Out Cross Validation
        print("||||||||||||||||||Leave One Subject Out||||||||||||||||||")
        tags = np.unique(train_data_info["Tag"].tolist())
        best_acc = 0
        for tag in tags:
            # Load the data
            print("|-----------Subject out: " + tag + "-----------|")
            LOSO_TrainSet = train_data_info[train_data_info["Tag"] != tag]
            LOSO_TestSet = train_data_info[train_data_info["Tag"] == tag]
            train_dataset = DataLoader(LOSO_TrainSet, gait_cycle=True, event=self.ground_event)
            test_dataset = DataLoader(LOSO_TestSet, gait_cycle=True, event=self.ground_event)
            window_size = 0.25

            # Load the train data
            print("Training...")
            print("Extracting features...")
            windows_data = []
            windows_labels = []
            for trial in tqdm(train_dataset):
                windows = DataWindowing(trial, window_size)
                data_heel_contact = windows.segment_by(0)
                data_toe_off = windows.segment_by(3)
                final_data = pd.concat([data_heel_contact, data_toe_off], ignore_index=True)
                windows_data.append(final_data[final_data.columns[0:15]])
                windows_labels.append(final_data[final_data.columns[-1]])

            X_train = pd.concat(windows_data, ignore_index=True)
            y_train = pd.concat(windows_labels, ignore_index=True)

            clf = tree.DecisionTreeClassifier()
            clf.fit(X_train, y_train)
            
            print("Validating...")
            print("Extracting features...")
            windows_data = []
            windows_labels = []
            for trial in tqdm(test_dataset):
                windows = DataWindowing(trial, window_size)
                data_heel_contact = windows.segment_by(0)
                data_toe_off = windows.segment_by(3)
                final_data = pd.concat([data_heel_contact, data_toe_off], ignore_index=True)
                windows_data.append(final_data[final_data.columns[0:15]])
                windows_labels.append(final_data[final_data.columns[-1]])

            X_val = pd.concat(windows_data, ignore_index=True)
            y_val = pd.concat(windows_labels, ignore_index=True)

            mean_acc = clf.score(X_val, y_val)
            print("Mean Accuracy: ", mean_acc)
            if mean_acc > best_acc:
                best_acc = mean_acc
                self.model = clf

        return self.model

    @staticmethod
    def __load_data(dataset):
        """
        Load all the data from the files of the dataset
        :param dataset: Data loader object
        :return: X and Y numpy arrays
        """
        X, Y = dataset[0]
        for i in tqdm(range(1, len(dataset))):
            x, y = dataset[i]
            X = np.insert(X, -1, x, axis=0)
            Y = np.insert(Y, -1, y, axis=0)

        return X, Y

    def explore_tree(self):
        # Gini vs Thresholds
        feature = self.model.tree_.feature
        threshold = self.model.tree_.threshold
        gini = self.model.tree_.impurity

        features = {0: "min_RK",
                    1: "max_RK",
                    2: "mean_RK",
                    3: "std_RK",
                    4: "end_value_RK",
                    5: "min_RA",
                    6: "max_RA",
                    7: "mean_RA",
                    8: "std_RA",
                    9: "end_value_RA",
                    10: "min_RH",
                    11: "max_RH",
                    12: "mean_RH",
                    13: "std_RH",
                    14: "end_value_RH"}

        for key in features.keys():
            gini_values = []
            threshold_values = []
            for i, f in enumerate(feature):
                if key == f:
                    gini_values.append(gini[i])
                    threshold_values.append(threshold[i])

            plt.figure()
            plt.scatter(gini_values, threshold_values)
            plt.xlabel('Gini Values')
            plt.ylabel('Threshold Values')
            name = features[key] + '.png'
            plt.savefig(name)

        # Rules of decision tree
        rules = tree.export_text(self.model,
                                 feature_names=["RightKnee", "LeftKnee", "RightAnkle",
                                                "LeftAnkle", "RightHip", "LeftHip"])
        text_file = open("rules.txt", "wt")
        n = text_file.write(rules)
        text_file.close()

        return None
