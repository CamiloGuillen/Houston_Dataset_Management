import os
import random
import pandas as pd


def train_test_files(data_path, labels_path, train_percentage):
    """
    Split the data files in train and test set
    :param data_path: String with the path of the files
    :param labels_path: String with the path of the labels
    :param train_percentage:Float percentage of train videos (0.0, 1.0)
    :return: Two pandas data frame with train and test information
    """
    # List all the files
    files = os.listdir(data_path)
    labels = os.listdir(labels_path)

    # List all the subjects
    subjects = []
    for file in files:
        subject = file.split('T')[0]
        if subject not in subjects:
            subjects.append(subject)

    # Select the subjects for train and test
    n_train = int(train_percentage * len(subjects))
    train_subjects = random.sample(subjects, n_train)
    train_subjects.sort()
    test_subjects = [s for s in subjects if s not in train_subjects]
    test_subjects.sort()
    # print(str(n_train) + " subjects were selected for train and " + str(len(subjects)-n_train) + " for test!")

    # Take all the files and labels for train and test subjects
    train_tags = [file.split('T')[0] for file in files if file.split('T')[0] in train_subjects]
    train_files = [os.path.join(data_path, file) for file in files if file.split('T')[0] in train_subjects]
    train_labels = [os.path.join(labels_path, label) for label in labels if label.split('-')[0] in train_subjects]
    test_tags = [file.split('T')[0] for file in files if file.split('T')[0] in test_subjects]
    test_files = [os.path.join(data_path, file) for file in files if file.split('T')[0] in test_subjects]
    test_labels = [os.path.join(labels_path, label) for label in labels if label.split('-')[0] in test_subjects]

    # Make the Train and Test Data Frames
    train_subjects_data = pd.DataFrame()
    train_subjects_data["Tag"] = train_tags
    train_subjects_data["File"] = train_files
    train_subjects_data["Label"] = train_labels
    test_subjects_data = pd.DataFrame()
    test_subjects_data["Tag"] = test_tags
    test_subjects_data["File"] = test_files
    test_subjects_data["Label"] = test_labels

    return train_subjects_data, test_subjects_data
