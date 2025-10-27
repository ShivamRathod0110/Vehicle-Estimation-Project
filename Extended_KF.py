import numpy as np
from numpy.linalg import inv

# Task 3 
# Define Extended Kalman Filter algorithm

# Subtask 1
# Define parameters of Kalman Filter based on what you have learned in the lecture


# Jacobian of measurement model
# JH = np.array([[1,0,0,0],
#               [0,1,0,0]])

# timestep
#dt = 0.04

class EKF:
    def __init__(self, x, dt=0.04):

        self.dt = dt
        
        # Initial state
        self.x = np.zeros((4,1)) if x is None else x.reshape(4,1)
        
        # Covariance matrix of state estimate
        self.P = np.diag([1000, 1000, 1000, 1000])
        
        # Process noise covariance (including heading noise)
        q_pos = 0.1
        q_vel = 0.1
        q_heading = np.deg2rad(1)  # 1 degree noise in radians
        self.Q = np.diag([q_pos, q_pos, q_vel, q_heading])**2
        
        # Measurement noise covariance (measuring position only)
        r_pos = 0.1
        self.R = np.diag([r_pos, r_pos])**2
        
        # Measurement matrix (position only)
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        
        # Identity matrix
        self.I = np.eye(4)

# Prediction step by Extended Kalman Filter

    def _state_transition(self, x, u):
        """
        Nonlinear state transition function f(x,u).
        x: current state (4x1)
        u: control input vector (2x1): [velocity, heading_rate]
        Returns predicted state (4x1).
        """
        px, py, v, h = x.flatten()
        velocity, heading_rate = u.flatten()
        dt = self.dt
        
        px_new = px + velocity * np.cos(h) * dt
        py_new = py + velocity * np.sin(h) * dt
        v_new = velocity  # assuming velocity control input is current velocity
        h_new = h + heading_rate * dt
        
        return np.array([[px_new], [py_new], [v_new], [h_new]])

    def _jacobian_F(self, x, u):
        """
        Compute Jacobian of f(x,u) w.r.t. state x.
        Returns 4x4 Jacobian matrix JF.
        """
        px, py, v, h = x.flatten()
        velocity, heading_rate = u.flatten()
        dt = self.dt
        
        JF = np.eye(4)
        JF[0,2] = 0  # partial px_new w.r.t velocity is via u, so zero here
        JF[0,3] = -velocity * np.sin(h) * dt
        JF[1,2] = 0
        JF[1,3] = velocity * np.cos(h) * dt
        JF[2,2] = 0  # velocity directly updated by control input
        JF[3,3] = 1
        
        return JF

    def predict(self, velocity, heading_rate):
        """
        EKF Prediction step.
        velocity: current velocity control input
        heading_rate: current heading rate control input (radians per timestep)
        """
        u = np.array([[velocity], [heading_rate]])
        
        # Predict next state with nonlinear function
        self.x = self._state_transition(self.x, u)
        
        # Compute Jacobian of state transition
        JF = self._jacobian_F(self.x, u)
        
        # Predict covariance
        self.P = JF @ self.P @ JF.T + self.Q
        
        return self.x

# Update step by Extended Kalman Filter

    def update(self, z):
        """
        EKF Update step.
        z: measurement vector (position only), shape (2,1)
        """
        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R
        
        # Kalman gain
        K = self.P @ self.H.T @ inv(S)
        
        # Measurement residual (innovation)
        y = z - self.H @ self.x
        
        # Update state estimate
        self.x = self.x + K @ y
        
        # Update covariance estimate
        self.P = (self.I - K @ self.H) @ self.P
        
        return self.x
'''

# Task 4 (Forecasting with EKF):

# Subtask 1
# Define a function for forecasting with Extended Kalman Filter
# In this task you must do forecasting for each point in measurement data 

#def forecast_ekf(f_dt,meas,n):  # f_dt is timestep for forecasting

#    u1 = []
#    B1 = []
#
#    for i in range(meas.shape[0]):
#        u1.append(np.array([[meas[i,2,0], meas[i,3,0]]], np.float64).T)
#        arr_u = np.concatenate(u1).reshape(len(u1),2,1)

#        B1.append(np.array([[np.cos(meas[i,3,0])*f_dt,0],
#                           [np.sin(meas[i,3,0])*f_dt,0],
#                           [0,f_dt],
#                           [1,0]],np.float64))

#    F= np.array([[ , , , ],
#                 [ , , , ],
#                 [ , , , ],
#                 [ , , , ]])
                
#    forecast_list = []
   
#    This for loop iterate over the measurment data, because we want to forecast next step for each individual point in measurement
   
#    for m, b, u in zip(meas, B1, arr_u):
      
#        lst = []
#        This for loop performs the forecasting for n iterations
#        for _ range(n):
             # Write the equation for forecasting in this for loop

#            forecast_point = ...
#            lst.append(forecast_point)
#            m = forecast_point
        
#        forecast_list.append(lst)
#
#    forecast_array = np.stack(forecast_list)

#    return forecast_array
'''

def forecast_ekf(f_dt, meas, n):
    forecast_list = []

    for i in range(meas.shape[0]):
        x = meas[i, :, 0].copy()  # state vector [x, y, v, h]
        # Normalize heading to [-pi, pi]
        x[3] = (x[3] + np.pi) % (2 * np.pi) - np.pi
       
        if i > 0:
            prev_h = meas[i - 1, 3, 0]
            curr_h = x[3]
            dh = (curr_h - prev_h + np.pi) % (2 * np.pi) - np.pi
            heading_rate = dh / f_dt
        else:
            heading_rate = 0.0

        v = x[2]
        lst = []

        for _ in range(n):
            h = x[3]
            v = x[2]

            # Nonlinear state update (same as EKF predict state_transition)
            x_pred = np.zeros_like(x)
            x_pred[0] = x[0] + v * np.cos(h) * f_dt
            x_pred[1] = x[1] + v * np.sin(h) * f_dt
            x_pred[2] = v           # constant velocity assumption
            x_pred[3] = h + heading_rate * f_dt         # variable heading assumption

            # Normalize heading angle
            x_pred[3] = (x_pred[3] + np.pi) % (2 * np.pi) - np.pi

            lst.append(x_pred.copy())
            x = x_pred

        forecast_list.append(lst)

    forecast_array = np.array(forecast_list)  # shape: (num_points, n, 4)
    forecast_array = forecast_array.reshape(forecast_array.shape[0], forecast_array.shape[1], 4, 1)
    return forecast_array

def run_ekf(measurements, dt=0.04):
    """
    Runs the Extended Kalman Filter for a single track.
    
    Args:
        measurements: A (N, 4, 1) numpy array of measurements [x, y, v, h]
        dt: Time step
    
    Returns:
        A (N, 4, 1) numpy array of filtered states.
    """
    
    # 1. Initialize the filter
    x_init = measurements[0]  # Initial state [x, y, v, h]
    ekf = EKF(x=x_init, dt=dt)
    
    filtered_states = [x_init]
    prev_h = x_init[3, 0] # Get initial heading
    
    # 2. Loop through all measurements (starting from the second one)
    for i in range(1, len(measurements)):
        
        # --- PREDICTION STEP ---
        # We need to calculate the control inputs (velocity, heading_rate)
        
        # Get velocity from the *current* measurement (as control input)
        v = measurements[i, 2, 0] 
        
        # Calculate heading_rate from the change in heading
        curr_h = measurements[i, 3, 0]
        dh = (curr_h - prev_h + np.pi) % (2 * np.pi) - np.pi # Handle angle wrap
        heading_rate = dh / dt
        
        ekf.predict(velocity=v, heading_rate=heading_rate)
        
        # --- UPDATE STEP ---
        # Get the position [x, y] from the current measurement
        z = measurements[i, 0:2, :]  # This correctly creates a (2, 1) column vector
        
        ekf.update(z)
        
        # Store the corrected state
        filtered_states.append(ekf.x.copy())
        
        # Update previous heading for the next loop
        prev_h = curr_h

    # 3. Return all states as a numpy array
    return np.array(filtered_states).reshape(len(measurements), 4, 1)




