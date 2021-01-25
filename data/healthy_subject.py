from scipy.io import loadmat


class Healthy_Subject:
    """
    Make structure for kinetic information
    """
    def __init__(self, data_path):
        data = loadmat(data_path)
        self.kin = data['kin']
        # Setup Information
        kin_setup_info = self.kin[0][0][0][0][0]
        self.kin_setup = {'segment_label': [i[0][0] for i in kin_setup_info[0]],
                          'sensor_label': [i[0][0] for i in kin_setup_info[1]],
                          'joint_label': [i[0][0] for i in kin_setup_info[2]],
                          'num_trials': kin_setup_info[3][0][0]}
        # Kinematic Data
        kin_data_info = self.kin[0][0][1][0][0]
        self.kin_data = {'joint_angle': {self.kin_setup['joint_label'][i]: kin_data_info[0][0][0][i] for i in
                                         range(len(self.kin_setup['joint_label']))},
                         'acceleration': {self.kin_setup['segment_label'][i]: kin_data_info[1][0][0][i] for i in
                                          range(len(self.kin_setup['segment_label']))},
                         'velocity': {self.kin_setup['segment_label'][i]: kin_data_info[2][0][0][i] for i in
                                      range(len(self.kin_setup['segment_label']))},
                         'position': {self.kin_setup['segment_label'][i]: kin_data_info[3][0][0][i] for i in
                                      range(len(self.kin_setup['segment_label']))},
                         'orientation_quaternion': {self.kin_setup['segment_label'][i]: kin_data_info[4][0][0][i] for i
                                                    in range(len(self.kin_setup['segment_label']))},
                         'orientation_euler': {self.kin_setup['segment_label'][i]: kin_data_info[5][0][0][i] for i in
                                               range(len(self.kin_setup['segment_label']))},
                         'angular_acceleration': {self.kin_setup['segment_label'][i]: kin_data_info[6][0][0][i] for i in
                                                  range(len(self.kin_setup['segment_label']))},
                         'angular_velocity': {self.kin_setup['segment_label'][i]: kin_data_info[7][0][0][i] for i in
                                              range(len(self.kin_setup['segment_label']))},
                         'sensor_acceleration': {self.kin_setup['sensor_label'][i]: kin_data_info[8][0][0][i] for i in
                                                 range(len(self.kin_setup['sensor_label']))},
                         'sensor_orientation': {self.kin_setup['sensor_label'][i]: kin_data_info[9][0][0][i] for i in
                                                range(len(self.kin_setup['sensor_label']))},
                         'sensor_angular_velocity': {self.kin_setup['sensor_label'][i]: kin_data_info[10][0][0][i] for i
                                                     in range(len(self.kin_setup['sensor_label']))}
                         }
        # Sample Rate
        self.sample_rate = self.kin[0][0][2][0][0]

    def segment_label(self):
        """
        :return: 23×1 array containing the names of the body segments
        """
        return self.kin_setup['segment_label']

    def sensor_label(self):
        """
        :return: 17×1 array containing the sensor placement sites
        """
        return self.kin_setup['sensor_label']

    def joint_label(self):
        """
        :return: 22×1 array containing the joint names

        """
        return self.kin_setup['joint_label']

    def num_trials(self):
        """
        :return: Scalar value indicating the number of times the gait course was completed for that session
         (down and back indicates one completion). This is either one or two.
        """
        return self.kin_setup['num_trials']

    def sensor_acceleration(self, sensor_label):
        """
        Get the string of the sensor label
        :return: N×3 matrix containing sensor acceleration vector (x,y,z) at each time point N (units: m/s2)
        """
        return self.kin_data['sensor_acceleration'][sensor_label]

    def sensor_angular_velocity(self, sensor_label):
        """
        Get the string of the sensor label
        :return: N×3 matrix containing sensor angular velocity vector (x,y,z) at each time point N (units: rad/s2)
        """
        return self.kin_data['sensor_angular_velocity'][sensor_label]

    def sensor_orientation(self, sensor_label):
        """
        Get the string of the sensor label
        :return: N×3 matrix containing sensor orientation vector (x,y,z) at each time point N in the global frame
         (units: m)
        """
        return self.kin_data['sensor_orientation'][sensor_label]

    def orientation_quaternion(self, segment_label):
        """
        Get the string of the segment label
        :return: N×4 matrix containing segment orientation quaternion (q0, q1, q2, q3) at each time point N
        """
        return self.kin_data['orientation_quaternion'][segment_label]

    def orientation_euler(self, segment_label):
        """
        Get the string of the segment label
        :return: N×3 matrix containing segment orientation (x,y,z) at each time point N (units: m)
        """
        return self.kin_data['orientation_euler'][segment_label]

    def position(self, segment_label):
        """
        Get the string of the segment label
        :return: N×3 matrix containing the position vector (x, y, z) of the origin of the segment in the global frame
         at each time point N (units: m)
        """
        return self.kin_data['position'][segment_label]

    def velocity(self, segment_label):
        """
        Get the string of the segment label
        :return: N×3 matrix containing the velocity vector (x, y, z) of the origin of the segment in the global frame at
         each time point N (units: m/s)
        """
        return self.kin_data['velocity'][segment_label]

    def acceleration(self, segment_label):
        """
        Get the string of the segment label
        :return: N×3 matrix containing the acceleration vector (x, y, z) of the origin of the segment in the global
         frame at each time point N (units: m/s2)
        """
        return self.kin_data['acceleration'][segment_label]

    def angular_velocity(self, segment_label):
        """
        Get the string of the segment label
        :return: N×3 matrix containing the angular velocity vector (x, y, z) of the segment in the global frame in
        (units: rad/s)
        """
        return self.kin_data['angular_velocity'][segment_label]

    def angular_acceleration(self, segment_label):
        """
        Get the string of the segment label
        :return: N×3 matrix containing the angular acceleration vector (x, y, z) of the segment in the global frame in
        (units: rad/s2)
        """
        return self.kin_data['angular_acceleration'][segment_label]

    def joint_angle(self, joint_label):
        """
        Get the string of the joint label
        :return: N×3 matrix containing the Euler representation of the joint angle vector (x, y, z) calculated using the
         Euler sequence ZXY (units: deg)
        """
        return self.kin_data['joint_angle'][joint_label]

    def get_sample_rate(self):
        """
        :return: Scalar value indicating sampling rate of system (30 Hz or 60 Hz)
        """
        return self.sample_rate
