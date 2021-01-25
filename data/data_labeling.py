import numpy as np

from scipy.io import loadmat

gait_event_labels = {'Right Heel Strike': 0,
                     'Left Toe Off': 1,
                     'Left Heel Strike': 2,
                     'Right Toe Off': 3,
                     'None': 4}

event_labels = {'LW0F': 0,
                'LW0B': 0,
                'LW1F': 0,
                'LW1B': 0,
                'LW2F': 0,
                'LW2B': 0,
                'LW3F': 0,
                'LW3B': 0,
                'LW4F': 0,
                'LW4B': 0,
                'RD': 1,
                'RA': 2,
                'SD': 3,
                'SA': 4,
                'None': 5}


class Load_Labels:
    """
    Make the structure for labels information.
    Labeling the data according to:
    - LWxy = Level Walking on terrain x in direction y. For example, LW2F is level walking on section two in the forward
     direction. Forward and Back are not the way they are walking. They always walk facing forward. The F and B just
     indicate if they are walking away from the start platform or returning to the start platform.
    - RA/RD = Ramp ascent and ramp descent
    - SA/SD = Stair ascent and Stair descent
    We define:
    - 0: Walking
    - 1: Ramp descent
    - 2: Ramp ascent
    - 3: Stair descent
    - 4: Stair ascent
    - 5: None
    Also we label according the gait cycle event:
    - 0: Right Heel Strike
    - 1: Left Toe Off
    - 2: Left Heel Strike
    - 3: Right Toe Off
    - 4: None
    """
    def __init__(self, label_path, length_data):
        label = loadmat(label_path)['gc']
        # Take relevant information of the file
        self.index = label[0][0][1]
        self.time = label[0][0][2]
        self.label = [i[0][0] for i in label[0][0][0]]
        # Organize the labels information terrain events and gait cycle event
        self.length_data = length_data
        self.labels_info = {}
        self.unlabeled_index = []
        gait_events = ['Right Heel Strike', 'Left Toe Off', 'Left Heel Strike', 'Right Toe Off', 'Right Heel Strike']
        event = self.label
        for i in range(len(self.index)):
            for j in range(len(gait_events)):
                self.labels_info[self.index[i][j]] = {'event': event[i], 'gait_event': gait_events[j]}
            if i < len(self.index) - 1:
                if not (self.index[i][-1] + 1 == self.index[i + 1][0]) and not (
                        self.index[i][-1] == self.index[i + 1][0]):
                    self.unlabeled_index.append(range(self.index[i][-1], self.index[i + 1][0]))

    def labelled(self):
        """
        Labeled of the data
        :return: List with labels of data
        """
        labels = []
        index = np.unique(self.index)
        for i in range(self.length_data):
            flag = False
            label_gait_event = None
            label_event = None
            for j in range(len(index)-1):
                if i < index[0] or i > index[-1]:
                    label_gait_event = gait_event_labels['None']
                    label_event = event_labels['None']
                elif index[j] <= i < index[j+1]:
                    for k in range(len(self.unlabeled_index)):
                        if i in self.unlabeled_index[k]:
                            flag = True
                    if flag is False:
                        label_gait_event = gait_event_labels[self.labels_info[index[j]]['gait_event']]
                        label_event = event_labels[self.labels_info[index[j]]['event']]
                    else:
                        label_gait_event = gait_event_labels['None']
                        label_event = event_labels['None']
                elif index[-2] <= i <= index[-1]:
                    label_gait_event = gait_event_labels[self.labels_info[index[-1]]['gait_event']]
                    label_event = event_labels[self.labels_info[index[-1]]['event']]

            labels.append([label_event, label_gait_event])

        return labels
