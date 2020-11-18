import os

from model.utils import train_test_files
from model.decision_tree import DecisionTree

path = "C:/Users/Camilo Guillen/Documents/Universidad de los Andes/Tesis/Datasets/University of Houston Dataset/" \
       "UH Dataset"
files_path = os.path.join(path, "kin_data")
labels_path = os.path.join(path, "labels")

if __name__ == '__main__':
    train_data_info, test_data_info = train_test_files(files_path, labels_path, train_percentage=1)
    decision_tree = DecisionTree()
    decision_tree.LOSO_train(train_data_info)
    decision_tree.explore_tree()
