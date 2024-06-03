'''
Created May 27: RL Path Planning
- script to do
'''
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import MPC_function
from scipy.interpolate import CubicSpline
from scipy.interpolate import splprep, splev
'''
Replace with function of this from ML_PathPlanning
'''
class Drone:
    def __init__(self, position, fuel, payload):
        self.position = np.array(position)
        self.fuel = fuel
        self.payload = payload

    def move(self, move):
        self.position += np.array(move)
        self.fuel -= np.linalg.norm(move) * self.payload

    def state(self):
        return tuple(self.position), self.fuel, self.payload


class DroneEnvironment:
    def __init__(self, gridsize, warehouse_location, houses_location, obstacles):
        self.grid = gridsize
        self.warehouse = np.array(warehouse_location)
        self.houses = [np.array(house) for house in houses_location]
        self.drone = Drone(self.warehouse, 100, 0)
        self.obstacles = [np.array(obstacle) for obstacle in obstacles]
        self.delivered = np.zeros(len(self.houses), dtype=bool)
        self.delivery = self.houses.copy()

        self.utility = np.zeros(gridsize)
        self.utility[tuple(self.warehouse)] = 0
        for i, h in enumerate(self.delivery):
            self.utility[tuple(h)] = 100
        for obstacle in self.obstacles:
            self.utility[tuple(obstacle)] = -10000
        # print(self.utility)

    def isdone(self):
        return np.all(self.delivered)

    def reset(self):
        self.drone = Drone(self.warehouse, 10, 1)
        self.delivered = np.zeros(len(self.houses), dtype=bool)
        self.delivery = self.houses.copy()
        return self.get_state()

    def get_state(self):
        return self.drone.position, self.drone.fuel, self.delivery

    def step(self, action):
        prev_fuel = self.drone.fuel
        self.drone.move(action)
        cost = -(self.drone.fuel - prev_fuel)
        # print('state',self.get_state()[0])
        future_state = self.get_state()
        # print(future_state)
        # print('utility at that state',self.utility[tuple(future_state[0])])
        return self.get_state(), self.utility[tuple(future_state[0])], self.isdone()

    def possible_actions(self, pos):
        possible_moves = np.array([[i, j, k] for i in [-1, 0, 1] for j in [-1, 0, 1] for k in [-1, 0, 1]])
        valid_moves = []
        if pos == None:
            pos = self.drone.position
        for move in possible_moves:
            new_position = np.array(pos) + np.array(move)
            if (0 <= new_position[0] < self.grid[0] and
                    0 <= new_position[1] < self.grid[1] and
                    0 <= new_position[2] < self.grid[2]):
                # print('new position', new_position)
                valid_moves.append(move)
        return np.array(valid_moves)


def value_iteration(k_max, drone_mdp, gamma):
    dim = np.array(drone_mdp.grid + (27, 4))
    Qvalues = np.empty(dim)
    optimal_utility = np.copy(drone_mdp.utility)
    alpha = 0.5
    for k_iter in range(k_max):
        print('Iteration: ', k_iter)
        for i in range(dim[0]):
            for j in range(dim[1]):
                for k in range(dim[2]):
                    poss_a = drone_mdp.possible_actions((i, j, k))
                    current_cost = optimal_utility[i, j, k]
                    for a in range(len(poss_a)):
                        next_step_cost = optimal_utility[i + poss_a[a][0], j + poss_a[a][1], k + poss_a[a][2]]
                        Qvalues[i, j, k, a] = np.append(poss_a[a], Qvalues[i, j, k, a, 3] + alpha * (
                                    current_cost + gamma * next_step_cost))
                        q__ = Qvalues[i, j, k]
                        optimal_utility[i, j, k] = np.max(q__[:, 3])
    return Qvalues, optimal_utility


# Initialize the environment
grid_size = (20, 20, 5)
warehouse_pos = (0, 0, 0)
houses_location = [(19, 19, 4)]

obstacles = []

for i in range(20):
    for j in range(20):
        for k in range(5):
            # if k < 2:
            #     obstacles.append((i,j,k))
            if j < 9:
                if i > 2:
                    obstacles.append((i, j, k))
            if j > 11:
                if i < 18:
                    obstacles.append((i, j, k))
env = DroneEnvironment(grid_size, warehouse_pos, houses_location, obstacles)
util = env.utility
Qvalues, opt_util = value_iteration(50, env, 0.1)

# Extract the Q-values for plotting
x, y, z, q_values = [], [], [], []
for i in range(grid_size[0]):
    for j in range(grid_size[1]):
        for k in range(grid_size[2]):
            Q = Qvalues[i, j, k]
            Q = max(Q[:, 3])
            if Q > 0:
                x.append(i)
                y.append(j)
                z.append(k)
                q_values.append(Q)

# Create a 3D scatter plot
if x:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(x, y, z, c=q_values, cmap='viridis')
    plt.colorbar(sc, label='Q-Value')
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_zlabel('Z Coordinate')
    plt.title('3D Plot of Q-Values (|Q| > 1)')
    plt.show()
else:
    print("No points with |Q| != 0 to plot.")


def simulate_optimal_path(Qvalues, warehouse_position):
    current_position = warehouse_position
    path = [current_position]

    for _ in range(100):
        # Extract the Q-values for the current position
        i, j, k = current_position
        Q = Qvalues[i, j, k]

        # Find the action with the maximum Q-value
        max_idx_q = np.argmax(Q[:, 3])
        optimal_action = Q[max_idx_q, :3]

        # Calculate the next position based on the optimal action
        next_position = tuple(int(x) for x in np.array(current_position) + optimal_action)

        # Append the next position to the path
        path.append(next_position)

        # Check if the next position is a terminal state (i.e., no more value to gain)
        if np.all(Q[max_idx_q, 3] == 0):
            break

        # Update the current position
        current_position = next_position
    return path


# Simulate the optimal path starting from the warehouse position
optimal_path = simulate_optimal_path(Qvalues, warehouse_pos)

# Print the optimal path
print("Optimal Path:")
for i, position in enumerate(optimal_path):
    print(f"Step {i + 1}: {position}")

# Extract x, y, z coordinates from the path
x_path = [position[0] for position in optimal_path]
y_path = [position[1] for position in optimal_path]
z_path = [position[2] for position in optimal_path]

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the optimal path
ax.plot(x_path, y_path, z_path, marker='o', markersize=5, color='blue')

# Set labels and title
ax.set_xlabel('X Coordinate')
ax.set_ylabel('Y Coordinate')
ax.set_zlabel('Z Coordinate')
plt.title('Optimal Path in 3D')

# Show the plot
plt.show()
'''
Replace with function of this from ML_PathPlanning
'''
def truncate_optimal_path(optimal_path):
    truncated_path = []
    for i in range(1, len(optimal_path)):
        if optimal_path[i] == optimal_path[i - 1]:
            break
        truncated_path.append(optimal_path[i - 1])
    truncated_path.append(optimal_path[i - 1])  # Add the last moving step
    return truncated_path

truncated_path = truncate_optimal_path(optimal_path)

t = np.arange(len(truncated_path))

x_path = [position[0] for position in truncated_path]
y_path = [position[1] for position in truncated_path]
z_path = [position[2] for position in truncated_path]

# # Create CubicSpline objects for each coordinate
# cs_x = CubicSpline(t, x_path)
# cs_y = CubicSpline(t, y_path)
# cs_z = CubicSpline(t, z_path)
#
# # Generate a dense range of indices for a smooth path
# t_dense = np.linspace(0, len(truncated_path) - 1, 300)
# x_smooth = cs_x(t_dense)
# y_smooth = cs_y(t_dense)
# z_smooth = cs_z(t_dense)

tck, u = splprep([x_path, y_path, z_path], s=3)  # Increase the smoothing factor `s` for more smoothing
u_dense = np.linspace(0, 1, 300)
x_smooth, y_smooth, z_smooth = splev(u_dense, tck)

# Create a 3D plot for the smooth path
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the original optimal path
ax.plot(x_path, y_path, z_path, marker='o', markersize=5, color='blue', linestyle='--', label='Original Path')

# Plot the smooth path
ax.plot(x_smooth, y_smooth, z_smooth, color='red', label='Smooth Path')

# Set labels and title
ax.set_xlabel('X Coordinate')
ax.set_ylabel('Y Coordinate')
ax.set_zlabel('Z Coordinate')
plt.title('Smoothed Optimal Path in 3D')
plt.legend()

# Show the plot
plt.show()

initial_state = np.zeros(12)
trajectory = np.array([x_smooth,y_smooth,z_smooth])
MPC_function.drone_control(trajectory,initial_state,0.1,0)
