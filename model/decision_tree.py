import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from data_loader import DataLoader
from data.data_windowing import DataWindowing
from tqdm import tqdm
from sklearn import tree
from sklearn.metrics import confusion_matrix, plot_confusion_matrix


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
        # Get data information
        window_size = 0.25
        dataset = DataLoader(train_data_info, gait_cycle=True, event=self.ground_event)
        print("Calculating balance data...")
        _, y = self.__extract_features(dataset, window_size)
        self.__balance_data_info(y)

        # Leave One Subject Out Cross Validation
        print("||||||||||||||||||Leave One Subject Out||||||||||||||||||")
        best_acc = 0
        best_X_val = None
        best_y_val = None
        tags = np.unique(train_data_info["Tag"].tolist())
        for tag in tags:
            # Load the data
            print("|-----------Subject out: " + tag + "-----------|")
            LOSO_TrainSet = train_data_info[train_data_info["Tag"] != tag]
            LOSO_TestSet = train_data_info[train_data_info["Tag"] == tag]
            train_dataset = DataLoader(LOSO_TrainSet, gait_cycle=True, event=self.ground_event)
            test_dataset = DataLoader(LOSO_TestSet, gait_cycle=True, event=self.ground_event)

            # Load the train data
            print("Training...")
            print("Extracting features...")
            X_train, y_train = self.__extract_features(train_dataset, window_size)
            clf = tree.DecisionTreeClassifier()
            clf.fit(X_train, y_train)
            
            print("Validating...")
            print("Extracting features...")
            X_val, y_val = self.__extract_features(test_dataset, window_size)

            mean_acc = clf.score(X_val, y_val)
            y_pred = clf.predict(X=X_val)
            conf_matrix = confusion_matrix(y_true=y_val, y_pred=y_pred, labels=[0.0, 1.0, 2.0, 3.0, 4.0])
            print("Mean Accuracy: ", mean_acc)
            print("Confusion Matrix", conf_matrix)

            if mean_acc > best_acc:
                best_acc = mean_acc
                self.model = clf
                best_X_val = X_val
                best_y_val = y_val

        print("________________________________________")
        print("Best accuracy: " + str(best_acc))
        plot_confusion_matrix(self.model, best_X_val, best_y_val, labels=[0.0, 1.0, 2.0, 3.0, 4.0])
        plt.show()

        return self.model

    @staticmethod
    def __balance_data_info(y):
        classes, freq = np.unique(y, return_counts=True)
        print("In total " + str(len(y)) + " windows.")
        print("Class frequency:")
        print("- LW: " + str((freq[0]/len(y))*100))
        print("- RD: " + str((freq[1]/len(y))*100))
        print("- RA: " + str((freq[2]/len(y))*100))
        print("- SD: " + str((freq[3]/len(y))*100))
        print("- SA: " + str((freq[4]/len(y))*100))

        return None

    @staticmethod
    def __extract_features(dataset, window_size):
        """
        Load all the data from the files of the dataset
        :param dataset: Data loader object
        :return: data and labels numpy arrays
        """
        windows_data = []
        windows_labels = []
        for trial in tqdm(dataset):
            windows = DataWindowing(trial, window_size)
            data_heel_contact = windows.segment_by(0)
            data_toe_off = windows.segment_by(3)
            final_data = pd.concat([data_heel_contact, data_toe_off], ignore_index=True)
            windows_data.append(final_data[final_data.columns[0:15]])
            windows_labels.append(final_data[final_data.columns[-1]])

        X = pd.concat(windows_data, ignore_index=True)
        y = pd.concat(windows_labels, ignore_index=True)

        return X, y

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
                                 feature_names=["min_RK", "max_RK", "mean_RK", "std_RK", "end_value_RK",
                                                "min_RA", "max_RA", "mean_RA", "std_RA", "end_value_RA",
                                                "min_RH", "max_RH", "mean_RH", "std_RH", "end_value_RH"
                                                ])
        text_file = open("rules.txt", "wt")
        n = text_file.write(rules)
        text_file.close()

        return n
