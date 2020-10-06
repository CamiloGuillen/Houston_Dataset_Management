import os

from model.utils import train_test_files
from model.random_forest import RFClassifier

path = "C:/Users/Camilo Guillen/Documents/Universidad de los Andes/Proyecto de Investigación/Datasets/University of " \
       "Houston Dataset/UH Dataset"
files_path = os.path.join(path, "kin_data")
labels_path = os.path.join(path, "labels")

if __name__ == '__main__':
    train_data_info, test_data_info = train_test_files(files_path, labels_path, train_percentage=0.7)
    RF = RFClassifier(train_data_info, test_data_info, normalize=False)
    RF.train(n_estimators=100)
    RF.test()
    RF.explore_tree()
    RF.analyze_results()
