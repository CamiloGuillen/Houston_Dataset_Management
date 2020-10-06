import numpy as np

from data.healthy_subject import Healthy_Subject
from data.data_labeling import Load_Labels
from data.data_management import Data_Management


class DataLoader:
    def __init__(self, data_info, gait_cycle=False, event=None):
        self.files_path = data_info["File"]
        self.labels_path = data_info["Label"]
        self.gait_cycle = gait_cycle
        self.event = event

    def __len__(self):
        return len(self.files_path)

    def __getitem__(self, idx):
        # Load the subject data
        subject = Healthy_Subject(self.files_path[idx])

        # Take the physic variable
        data_RightKnee = subject.joint_angle('jRightKnee')
        data_LeftKnee = subject.joint_angle('jLeftKnee')
        data_RightAnkle = subject.joint_angle('jRightAnkle')
        data_LeftAnkle = subject.joint_angle('jLeftAnkle')
        data_RightHip = subject.joint_angle('jRightHip')
        data_LeftHip = subject.joint_angle('jLeftHip')

        # Load the labels
        labels = Load_Labels(self.labels_path[idx], len(data_RightKnee))
        subject_label = labels.labelled()

        # Create the Data Management objects
        RightKnee = Data_Management(data_RightKnee, subject_label, self.event)
        LeftKnee = Data_Management(data_LeftKnee, subject_label, self.event)
        RightAnkle = Data_Management(data_RightAnkle, subject_label, self.event)
        LeftAnkle = Data_Management(data_LeftAnkle, subject_label, self.event)
        RightHip = Data_Management(data_RightHip, subject_label, self.event)
        LeftHip = Data_Management(data_LeftHip, subject_label, self.event)

        # Clean the missing labels and separate the data by ground event
        data_RightKnee = RightKnee.clean_Missing_Labels()
        data_LeftKnee = LeftKnee.clean_Missing_Labels()
        data_RightAnkle = RightAnkle.clean_Missing_Labels()
        data_LeftAnkle = LeftAnkle.clean_Missing_Labels()
        data_RightHip = RightHip.clean_Missing_Labels()
        data_LeftHip = LeftHip.clean_Missing_Labels()

        # Sagittal Plane information
        splane_data_RightKnee = data_RightKnee['Z'].tolist()
        splane_data_LeftKnee = data_LeftKnee['Z'].tolist()
        splane_data_RightAnkle = data_RightAnkle['Z'].tolist()
        splane_data_LeftAnkle = data_LeftAnkle['Z'].tolist()
        splane_data_RightHip = data_RightHip['Z'].tolist()
        splane_data_LeftHip = data_LeftHip['Z'].tolist()

        # Labels
        gait_cycle_labels = data_RightKnee['GaitCycle'].tolist()
        ground_event_labels = data_RightKnee['GroundEvent'].tolist()

        # Data
        final_data = np.array((
            splane_data_RightKnee,
            splane_data_LeftKnee,
            splane_data_RightAnkle,
            splane_data_LeftAnkle,
            splane_data_RightHip,
            splane_data_LeftHip)).T

        if self.gait_cycle:
            return final_data, np.array(gait_cycle_labels)
        else:
            return final_data, np.array(ground_event_labels)
