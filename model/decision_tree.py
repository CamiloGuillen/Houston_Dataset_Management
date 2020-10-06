from data_loader import DataLoader


class DecisionTree:
    def __init__(self, ground_event=None):
        """
        :param ground_event: Int with ground event
        """
        self.model = None
        self.ground_event = ground_event
