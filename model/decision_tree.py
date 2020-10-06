import numpy as np

from data_loader import DataLoader
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
            print("Loading train data...")
            X_train, y_train = self.__load_data(train_dataset)
            print("Loading test data...")
            X_test, y_test = self.__load_data(test_dataset)

            # Train the tree
            clf = tree.DecisionTreeClassifier(criterion='entropy', random_state=0)
            clf.fit(X_train, y_train)
            # Mean accuracy on test data
            acc = clf.score(X_test, y_test)
            if acc > best_acc:
                best_acc = acc
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
