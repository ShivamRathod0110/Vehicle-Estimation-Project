# importing libraries
import numpy as np
import itertools

# Task 1 ( Reading the dataset )

# Subtask 1
#
# (REMOVED) file = "C:/Users/srath/Desktop/folder_2025-05-24__06-06-04__900/dataset/tracks_08.csv"
#
# We now pass the file_path in as an argument to the function
def read(file_path):
    # use a function to read the data (e.g. with numpy)
    data = np.loadtxt(file_path, delimiter=',')
    return data


# Subtask 2
# Define a function that take the data as an input and creates measurements as a 2d Numpy array
def array2d(data):
    measurements = []
    for row in data:
        measurements.append(tuple(row))
    return measurements

# (REMOVED) The test code at the bottom (a = read(file), etc.)
# A file should only contain function/class definitions to be importable

def extract_meas(data, kf_choice):

    # Group data by ID
    data_sorted = sorted(data, key=lambda x: x[0])  # Group by trackID
    groups = itertools.groupby(data_sorted, key=lambda x: x[0])

    if kf_choice == 'kf':
        # Extracts only x,y,vx,vy from the measurements
        lst = [[[item[2], item[3], item[4], item[5]] for item in group] for _, group in groups]

        # Extracts ego car (first car) and reshape the measurements
        return [np.array(track).reshape(len(track), 4, 1) for track in lst]

    else:
        # Do the same method for EKF,you also need heading value for calculation
        lst = [[[item[2], item[3], item[4], item[5], item[6]] for item in group]
               for _, group in groups]
        
        all_meas = []

        for track in lst:
            # Extract elements from lst
            x, y, vx, vy, h = np.array(track).T
            # Calculate the magnitude of velocity
            mag_v = np.sqrt(vx**2 + vy**2)
            # Convert the heading value to radian
            h = np.deg2rad(h)

            # Creates measurements array based on space state model for EKF then reshapes the data
            meas = np.column_stack((x, y, mag_v, h)).reshape(len(x), 4, 1)
            all_meas.append(meas)
            
        return all_meas