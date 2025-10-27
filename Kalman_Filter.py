import numpy as np 
from numpy.linalg import inv

# Task 2 
# Implement Linear Kalman Filter algorithm

# Subtask 1
# Define variables and algorithm of Linear Kalman Filter based on what you have learned in the lecture

# Timestep
#dt = 0.04

class KF:
    def __init__(self, x, dt=0.04):
        # initial state
        self.x = np.zeros((4,1)) if x is None else x.reshape(4,1)

        # Covariance matrix of state estimate
        self.P = np.diag([1000, 1000, 1000, 1000])

        # State transition matrix
        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1,  0],
            [0, 0, 0,  1]
        ])
        
        # Control input matrix
        self.B = np.array([
            [0.5 * dt**2, 0],
            [0, 0.5 * dt**2],
            [dt, 0],
            [0, dt]
        ])
        
        # Process noise covariance (tune as needed)
        q = 0.1
        self.Q = np.diag([q, q, q, q])**2
        
        # Measurement model (we measure position only)
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        
        # Covariance matrix of observation noise (tune as needed)
        r = 0.1
        self.R = np.diag([r, r])**2
        
        # Identity matrix
        self.I = np.eye(4)

# Prediction step by Kalman Filter

    def predict(self, vx, vy):

        # Control input vector
        u = np.array([[vx], [vy]])
        
        # Predict the next state
        self.x = self.F @ self.x + self.B @ u
        
        # Predict the covariance
        self.P = self.F @ self.P @ self.F.T + self.Q
        
        return self.x

# Update step by Kalman Filter

    def update(self, z):
        
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
# Task 4 (Forecasting with linear KF):

# Subtask 1
# Define a function for forecasting with Linear Kalman Filter 
# In this task you must do forecasting for each point in measurement data

#def forecast_kf(f_dt,meas,n): # f_dt is dt for forcasting

    # F= np.array([[ , , , ],
    #              [ , , , ],
    #              [ , , , ],
    #              [ , , , ]])


    # B = np.array([[(f_dt**(2))/2, 0],
    #              [0, (f_dt**(2))/2 ],
    #              [f_dt, 0],
    #              [0,f_dt]])
    
    # u1 = []
    # for i in range(meas.shape[0]):
    #           
    #    u1.append(np.array([[ , ]]).T)
    #    arr_u = np.concatenate(u1).reshape(len(u1),2,1)
        

    #forecast_list = []

    # This for loop iterate over the measurment data, because we want to forecast next step for each individual point in measurement

    # for m, u in zip(meas,arr_u):                      # zip is being used to iterate over two iterables
          
        # lst = []
        # This for loop performs forecasting for n iterations
        # for _ in range(n):
        
        # write the equation for forecasting in for loop
        #     forecast_point = ...
        #     lst.append(forecast_point)
        #     m = forecast_point
        
        # forecast_list.append(lst)
        
    # forecast_array = np.stack(forecast_list)

    #return forecast_array
'''
def forecast_kf(f_dt, meas, n):

    # Define state transition matrix F
    F = np.array([
        [1, 0, f_dt, 0],
        [0, 1, 0, f_dt],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ], dtype=np.float64)

    # Define control input matrix B
    B = np.array([
        [(f_dt**2)/2, 0],
        [0, (f_dt**2)/2],
        [f_dt, 0],
        [0, f_dt]
    ], dtype=np.float64)

    forecast_list = []

    # Loop over each measurement (initial state)
    for i in range(meas.shape[0]):
        # Initial state
        x, y, vx, vy = meas[i, :, 0]
        state = np.array([[x], [y], [vx], [vy]], dtype=np.float64)

        lst = []

        for _ in range(n):
            # Control input (velocity vector)
            u = np.array([[vx], [vy]], dtype=np.float64)

            # Predict next state
            forecast_point = F @ state + B @ u
            lst.append(forecast_point)
            state = forecast_point  # update for next step

            # Update velocity for next iteration
            vx = state[2, 0]
            vy = state[3, 0]

        forecast_list.append(lst)

    # Shape: (num_points, n_forecasts, 4, 1)
    forecast_array = np.stack(forecast_list)

    return forecast_array
