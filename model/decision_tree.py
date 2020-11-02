import numpy as np
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
            train_dataset = DataLoader(LOSO_TrainSet, gait_cycle=False, event=self.ground_event)
            test_dataset = DataLoader(LOSO_TestSet, gait_cycle=False, event=self.ground_event)

            print("Training...")
            clf = None
            for trial in train_dataset:
                # Data Windowing
                signal = trial[0]
                labels = trial[1]
                sample_rate = trial[2]
                window_size = 1.1
                train_windows = DataWindowing(signal, labels, window_size, sample_rate)
                clf = tree.DecisionTreeClassifier()
                X = []
                y = []
                for window in train_windows:
                    windows = window[0]
                    labels = window[1]
                    for i in range(len(windows)):
                        X.append(windows[i])
                        y.append(labels[i])

                clf.fit(X, y)

            print("Validating...")
            # Mean accuracy on test data
            for test_trial in test_dataset:
                # Data Windowing
                signal = test_trial[0]
                labels = test_trial[1]
                sample_rate = test_trial[2]
                window_size = 1.1
                test_windows = DataWindowing(signal, labels, window_size, sample_rate)
                X_val = []
                y_val = []
                for window in test_windows:
                    windows = window[0]
                    labels = window[1]
                    for i in range(len(windows)):
                        X_val.append(windows[i])
                        y_val.append(labels[i])
                mean_acc = clf.score(X_val, y_val)
                print("Mean Accuracy: ", mean_acc)
                if mean_acc > best_acc:
                    best_acc = mean_acc
                    self.model = clf

        print("Best Mean Accuracy: ", best_acc)

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

        features = {0: "RightKnee",
                    1: "LeftKnee",
                    2: "RightAnkle",
                    3: "LeftAnkle",
                    4: "RightHip",
                    5: "LeftHip"}

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
