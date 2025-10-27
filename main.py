from Data_Preparation import *
from Kalman_Filter import *
from Extended_KF import *
import Visualization

# Example Visualization (given later)
# from script_4 import Visualizer

# Read input from user
selection = input("Enter 1, 2, or 3 to select the dataset: ")
kf_choice = input("Enter filter type ('kf' or 'ekf'): ").lower()

# Define base path
base_folder = "C:/Users/srath/Desktop/folder_2025-05-24__06-06-04__900/dataset/"

# Map selection to specific file and image names
file_map = {
    "1": ("tracks_08.csv", "08_background.png"),
    "2": ("tracks_26.csv", "26_background.png"),
    "3": ("tracks_32.csv", "32_background.png"),
}

# Validate user input
if selection not in file_map:
    print("Invalid selection. Please enter 1, 2, or 3.")
    exit()

# Set the selected file and image paths
csv_file, image_file = file_map[selection]
csv_path = base_folder + csv_file
image_path = base_folder + image_file
  
# Read data
data = read(file=csv_path)
print(f"Data shape: {data.shape}")
#kf = "kf"    # Options are: "kf" or "ekf"
forecasting = True
n=5  # Number of iterations for forecasting
f_dt = 0.2  # Time step for forecasting
#===========================================================================================
meas_list = extract_meas(data, kf_choice)

all_filtered = []
all_forecasted = []


for meas in meas_list:
    if kf_choice == "kf":
        filter_obj = KF(x=meas[0])
    else:
        filter_obj = EKF(x=meas[0])

    filtered_states = []
    for i in range(meas.shape[0]):
        if kf_choice == "kf":
            x, y, vx, vy = meas[i,:,0]
            filter_obj.predict(vx, vy)
            updated = filter_obj.update(np.array([[x],[y]]))
        else:
            x, y, v, h = meas[i,:,0]
            filter_obj.predict(v, h)
            updated = filter_obj.update(np.array([[x],[y]]))

        filtered_states.append(updated)

    filtered_states = np.concatenate(filtered_states).reshape(-1, 4, 1)
    all_filtered.append(filtered_states)

    if forecasting:
        if kf_choice == "kf":
            forecast = forecast_kf(f_dt, meas, n)
        else:
            forecast = forecast_ekf(f_dt, meas, n)
        all_forecasted.append(forecast)
#===========================================================================================
'''
# Extract measurements and visualize
meas = extract_meas(data, kf_choice)
print(f"Measurement shape: {meas.shape}")
raw_positions = meas[:, :2, 0]

print("Visualizing raw measurements...")
#Visualization.visualize(meas[:, :2, 0], image_path)  # only positions x,y

# Initialize filter
if kf_choice == "kf":
    filter_obj = KF(x=meas[0])
elif kf_choice == "ekf":
    filter_obj = EKF(x=meas[0])
else:
    print("Invalid filter choice, must be 'kf' or 'ekf'")
    exit()

# Creating an empty list to append filtered data by KF/EKF
filtered_states = []

for i in range(meas.shape[0]):
    if kf_choice == "kf":
        # meas[i] shape: (4,1): [x,y,vx,vy]
        x, y, vx, vy = meas[i,:,0]
        predicted = filter_obj.predict(vx, vy)
        updated = filter_obj.update(np.array([[x],[y]]))
    else:
        # EKF: meas[i] shape: (4,1): [x,y,v,h]
        x, y, v, h = meas[i,:,0]
        predicted = filter_obj.predict(v, h)
        updated = filter_obj.update(np.array([[x],[y]]))

    filtered_states.append(updated)

filtered_states = np.concatenate(filtered_states).reshape(-1, 4, 1)
filtered_positions = filtered_states[:, :2, 0]
# Visualize filtered positions
print("First 5 raw:", raw_positions[:5])
print("First 5 filtered:", filtered_positions[:5])
print("Visualizing filtered states...")
#Visualization.visualize(filtered_states[:, :2, 0], image_path)  # x,y positions filtered
print("Filtering complete.")

# Task 4, Subtask 1: Forecasting with Linear KF or EKF

# timestep for forcasting
f_dt = 0.2

# number of iteration for forecasting 
n=5

if forecasting:

    if kf_choice == "kf":

        forecast_data = forecast_kf(f_dt,meas,n)
        
    else:
    
        forecast_data = forecast_ekf(f_dt,meas,n)
'''
#Visualization.animate_trajectory_with_controls(meas[:, :2, 0], filtered_positions, image_path, forecast_data)
'''
Visualization.animate_trajectory_with_controls(
    [m[:, :2, 0] for m in meas_list],
    [f[:, :2, 0] for f in all_filtered],
    image_path,
    all_forecasted
)
'''

# Task 5: ADE
# add your ADE calculation for the forecast here 
'''
def compute_ade(forecast_list, meas_list):
    total_error = 0
    total_points = 0

    for forecast, meas in zip(forecast_list, meas_list):
        forecast_xy = forecast[:, :, :2, 0]  # (T, N, 2)
        gt_positions = meas[:, :2, 0]        # (T, 2)

        max_gt_idx = gt_positions.shape[0] - 1
        max_forecast_steps = forecast_xy.shape[1]

        for i in range(forecast_xy.shape[0]):  # start times for forecast
            for j in range(max_forecast_steps):
                gt_idx = i + j
                if gt_idx > max_gt_idx:
                    # can't compare beyond ground truth range
                    break
                error = np.linalg.norm(forecast_xy[i, j, :] - gt_positions[gt_idx])
                total_error += error
                total_points += 1

    ade = total_error / total_points if total_points > 0 else float('nan')
    return ade
'''

def compute_ade_per_track(forecast_data_list, ground_truth_list):
    """
    Compute ADE per track (car) using squared error formulation.

    Args:
        forecast_data_list: list of np.arrays, each shape (T, pred_len, 2, 1)
        ground_truth_list: list of np.arrays, each shape (T+pred_len, 2, 1)

    Returns:
        ade_per_track: list of lists with ADE values over time steps per car.
    """
    ade_per_track = []
    for car_idx, forecast in enumerate(forecast_data_list):
        gt = ground_truth_list[car_idx]
        car_ade = []
        T, pred_len, _, _ = forecast.shape

        for t in range(T):
            if t + pred_len > len(gt):
                break
            dx = 0.0
            dy = 0.0
            for p in range(pred_len):
                pred_x = forecast[t, p, 0, 0]
                pred_y = forecast[t, p, 1, 0]
                true_x = gt[t + p, 0, 0]
                true_y = gt[t + p, 1, 0]

                dx += (pred_x - true_x) ** 2
                dy += (pred_y - true_y) ** 2

            ade = np.sqrt((dx + dy) / (2 * pred_len))
            car_ade.append(ade)
        ade_per_track.append(car_ade)

    all_ades = [ade for car_ades in ade_per_track for ade in car_ades]
    if all_ades:
        final_ade = np.mean(all_ades)
        print(f"Final average ADE across all cars and time steps: {final_ade:.4f} meters")
    else:
        print("No ADE values computed.")

    return ade_per_track

ade_per_track = compute_ade_per_track(all_forecasted, meas_list)  


Visualization.animate_trajectory_with_controls(
    [m[:, :2, 0] for m in meas_list],
    [f[:, :2, 0] for f in all_filtered],
    image_path,
    all_forecasted,
    ade_per_track=ade_per_track
)

'''
if forecasting:
    ade = compute_ade(all_forecasted, meas_list)
    print(f"\n Average Displacement Error (ADE): {ade:.4f} meters")
'''
# Visualization
# add your own visualization fucntion here


# Example Visualization (given later)
# vis = Visualizer(data=data,filter_data=filter_data,forecast_points= forecast_points, 
#                 background_path=path, mini=0, maxi=10000, interval=40, blit=False )

# vis.show()
