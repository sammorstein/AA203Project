# Optimal Routing and Control for Drone-Based Delivery Services
## Project Overview
Drone-based delivery services are an emerging field with significant potential to revolutionize logistics in urban areas. While many companies, such as Zipline, focus on single-package deliveries to specific locations, this project aims to find the optimal routing solution for delivering multiple packages to various locations. The ultimate goal is to enable drones to efficiently handle delivery routes in dense urban areas.
![updated_image](https://github.com/user-attachments/assets/c052d931-dbc5-4b91-80ac-a93be9de8fbf)

## Problem Statement
- **Grid Space**: The project operates within a 20x20x5 discrete grid space.
- **Delivery Points**: Three delivery locations at coordinates (15, 15, 0), (19, 0, 0), and (0, 19, 0).
- **Obstacle**: A static obstacle is positioned at the center of the grid space.
- **Drone Model**: A standard quadrotor model is used as the delivery drone [1].

## Approach

The project is split into two main parts: trajectory generation and drone control.

### Trajectory Generation

- **Method**: Uses value iteration to generate a value map of the grid space.
- **Objective**: Determines the optimal policy by maximizing the reward function, guiding the drone through the best routes while avoiding obstacles.


### Drone Control

- **Control Strategy**: Implements a nonlinear Model Predictive Control (MPC) algorithm for reference trajectory tracking.
- **Goal**: Ensures the drone accurately follows the generated trajectory, compensating for any disturbances or dynamic changes in the environment.

MPC Cost Function:

![Screenshot_14](https://github.com/user-attachments/assets/65993960-c233-4fe1-8236-ec7c030ae75f)

## Results

### Video Demonstration


https://github.com/user-attachments/assets/429d4a78-ad2b-4eaa-858e-87e5862f82a8

### Trajectory Generation:
![contour_qlearn](https://github.com/user-attachments/assets/44a8ad5f-c5cd-425b-b4de-9034694b39fc) 

Contour plots of Q-Learning around desirable points

![discrete_traj](https://github.com/user-attachments/assets/f983ac20-1ce1-4992-b468-b53c70f0bbdf)

Discretized trajectories to deliver all packages

![Figure_1](https://github.com/user-attachments/assets/19318797-e965-4c4d-a19f-dc202e3f0568)

Smoothed Trajectory

### Control:
![Screenshot_1](https://github.com/user-attachments/assets/e5610401-8625-460d-92b5-67bb0746c4da)

The actual and reference trajectories for each segment of the drone’s path.

![Screenshot_2](https://github.com/user-attachments/assets/af67c354-e30d-4e01-96e8-6ac08afe6b74)

The control inputs to the drone for each segment of the trajectory.





## References
[1] Teppo Luukkonen. Modelling and control of quadcopter. Independent research project in applied mathematics, 2011
