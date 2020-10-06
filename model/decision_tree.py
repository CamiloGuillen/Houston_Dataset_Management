from data_loader import DataLoader


class DecisionTree:
    def __init__(self, train_data_info):
        """
        """
        self.model = None
        train_dataset = DataLoader(train_data_info, event=0)
        a = train_dataset[0]
