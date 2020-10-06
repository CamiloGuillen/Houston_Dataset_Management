from data_loader import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.tree import export_graphviz
from sklearn import tree
from tqdm import tqdm

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class RFClassifier:
    def __init__(self, train_data_info, test_data_info, normalize=True):
        """
        :param train_data_info: Pandas Data Frame
        :param test_data_info: Pandas Data Frame
        """
        # Load the data
        print("Loading Train data...")
        self.X_train, self.Y_train = self.__load_data(train_data_info)
        print(str(self.X_train.shape[0]) + " data were loaded for Train!")
        print("Loading Test data...")
        self.X_test, self.Y_test = self.__load_data(test_data_info)
        print(str(self.X_test.shape[0]) + " data were loaded for Test!")
        # Normalize the data
        if normalize:
            scaler = StandardScaler().fit(self.X_train)
            self.X_train = scaler.transform(self.X_train)
            self.X_test = scaler.transform(self.X_test)
        # Best classifier
        self.best_clf = None

    @staticmethod
    def __load_data(data_info):
        """
        Load all the data from the files of the dataset
        :param data_info:
        :return: X and Y numpy arrays
        """
        dataset = DataLoader(data_info)
        X, Y = dataset[0]
        for i in tqdm(range(1, len(dataset))):
            x, y = dataset[i]
            X = np.insert(X, -1, x, axis=0)
            Y = np.insert(Y, -1, y, axis=0)

        return X, Y

    def train(self, n_estimators=100):
        # Separate Train into Train and Validation sets
        X_train, X_val, y_train, y_val = train_test_split(self.X_train, self.Y_train, test_size=0.20, random_state=42)
        # Train the Classifier
        print("||||||||||||||||||||||||||||TRAIN||||||||||||||||||||||||||||")
        best_acc = 0
        best_i = None
        acc_hist = []
        for i in range(1, n_estimators):
            clf = RandomForestClassifier(n_estimators=i, max_depth=None)
            clf.fit(X_train, y_train)
            val_predictions = clf.predict(X_val)
            acc = accuracy_score(y_val, val_predictions)
            acc_hist.append(acc)
            if acc > best_acc:
                best_i = i
                self.best_clf = clf
                best_acc = acc
                print("Accuracy: " + str(best_acc) + "; " + str(best_i) + " estimators.")

        print("Best classifier: " + str(best_i) + " estimators.")
        plt.plot(range(len(acc_hist)), acc_hist)
        plt.grid(True)
        plt.show()

        return self.best_clf

    def test(self):
        print("||||||||||||||||||||||||||||TEST|||||||||||||||||||||||||||||")
        test_predictions = self.best_clf.predict(self.X_test)
        acc_test = accuracy_score(self.Y_test, test_predictions)
        cm = confusion_matrix(self.Y_test, test_predictions)
        print("Test Accuracy: " + str(acc_test))
        print(cm)

    def explore_tree(self):
        estimators = self.best_clf.estimators_
        for i in range(len(estimators)):
            tree.plot_tree(estimators[i])
            # name = "tree_" + str(i)
            # export_graphviz(estimators[i], out_file=name + '.dot',
            #                 feature_names=['RightKnee', 'LeftKnee', 'RightAnkle', 'LeftAnkle', 'RightHip', 'LeftHip'],
            #                 class_names=['Walk', 'RampDescent', 'RampAscent', 'StairDescent', 'StairAscent'],
            #                 rounded=True, proportion=False,
            #                 precision=2, filled=True)

    def analyze_results(self):
        features_importance = pd.DataFrame(
            {'Feature': ['RightKnee', 'LeftKnee', 'RightAnkle', 'LeftAnkle', 'RightHip', 'LeftHip'],
             'Importance': self.best_clf.feature_importances_}).sort_values('Importance', ascending=False)
        print("|||||||||||||||||Feature Importance|||||||||||||||||")
        print(features_importance)
