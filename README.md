# Vehicle Estimation Project

This project implements and compares sensor fusion algorithms, specifically the **Kalman Filter (KF)** and the **Extended Kalman Filter (EKF)**, for vehicle state estimation. The goal is to accurately track a vehicle's state (e.g., position, velocity) by fusing noisy sensor measurements.



## Project Overview

In many real-world applications, like autonomous driving or robotics, we need to know the precise state of a system. Sensor data (like RADAR or LIDAR) is often noisy and incomplete. This project demonstrates how to use filtering techniques to produce an optimal estimate of the vehicle's state by combining a motion model with sensor measurements over time.

### Features
* **Standard Kalman Filter**: Implementation of a linear Kalman Filter.
* **Extended Kalman Filter**: Implementation of an Extended Kalman Filter for non-linear models.
* **Data Handling**: A module for loading and preparing sensor data.
* **Visualization**: A utility to plot the estimated trajectory, sensor measurements, and ground truth data for comparison.

## Getting Started

Follow these instructions to get a copy of the project up and running on your local machine.

### Prerequisites

* Python 3.x
* pip (Python package installer)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/ShivamRathod0110/Vehicle-Estimation-Project.git](https://github.com/ShivamRathod0110/Vehicle-Estimation-Project.git)
    cd Vehicle-Estimation-Project
    ```

2.  **Install the required dependencies:**
    All necessary Python libraries are listed in `requirements.txt`.
    ```bash
    pip install -r requirements.txt
    ```
    Common dependencies include:
    * `numpy`: For numerical operations.
    * `matplotlib`: For plotting the results.

## How to Run

To run the main estimation and visualization script, execute `main.py` from the root directory:

```bash
python main.py
```

This script will:
1.  Load the sensor data from the `data/` directory.
2.  Process the data using the Kalman Filter and/or Extended Kalman Filter.
3.  Generate plots (using `Visualization.py`) to show the filter's performance and save them to the `images/` directory.

**OR**
You can view the file from: https://colab.research.google.com/drive/1ye3Z39z9OlYKb94RZ2Ot1MRSEI_8QGrv?usp=sharing

