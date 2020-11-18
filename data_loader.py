import numpy as np

from data.healthy_subject import Healthy_Subject
from data.data_labeling import Load_Labels
from data.data_management import Data_Management


class DataLoader:
    def __init__(self, data_info, gait_cycle=False, event=None):
        self.files_path = data_info["File"].tolist()
        self.labels_path = data_info["Label"].tolist()
        self.gait_cycle = gait_cycle
        self.event = event

    def __len__(self):
        return len(self.files_path)

    def __getitem__(self, idx):
        # Load the subject data
        subject = Healthy_Subject(self.files_path[idx])
        # Take the physic variable
        data_RightKnee = subject.joint_angle('jRightKnee')
        data_RightAnkle = subject.joint_angle('jRightAnkle')
        data_RightHip = subject.joint_angle('jRightHip')

        # Sample rate
        sample_rate = subject.get_sample_rate()

        # Load the labels
        labels = Load_Labels(self.labels_path[idx], len(data_RightKnee))
        subject_label = labels.labelled()

        # Create the Data Management objects
        RightKnee = Data_Management(data_RightKnee, subject_label, self.event)
        RightAnkle = Data_Management(data_RightAnkle, subject_label, self.event)
        RightHip = Data_Management(data_RightHip, subject_label, self.event)

        # Clean the missing labels and separate the data by ground event
        data_RightKnee = RightKnee.clean_Missing_Labels()
        data_RightAnkle = RightAnkle.clean_Missing_Labels()
        data_RightHip = RightHip.clean_Missing_Labels()

        # Sagittal Plane information
        splane_data_RightKnee = data_RightKnee['Z'].tolist()
        splane_data_RightAnkle = data_RightAnkle['Z'].tolist()
        splane_data_RightHip = data_RightHip['Z'].tolist()

        # Labels
        gait_cycle_labels = data_RightKnee['GaitCycle'].tolist()
        ground_event_labels = data_RightKnee['GroundEvent'].tolist()

        # Data
        final_data = np.array((
            splane_data_RightKnee,
            splane_data_RightAnkle,
            splane_data_RightHip)).T

        if self.gait_cycle:
            return final_data, np.array(ground_event_labels), np.array(gait_cycle_labels), sample_rate
        else:
            return final_data, np.array(ground_event_labels), sample_rate
