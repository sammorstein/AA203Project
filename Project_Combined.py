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
from matplotlib import animation
'''
Replace with function of this from ML_PathPlanning
'''
class Drone:
    def __init__(self, position, fuel, payload):
        self.position = np.array(position)
        self.fuel = fuel
        self.payload = payload
    
    def move(self, move):
        newpos = np.add(self.position, np.array(move))
        self.position =newpos
        self.fuel -= np.linalg.norm(move) * self.payload

    def state(self):
        return tuple(self.position), self.fuel, self.payload

class DroneEnvironment:
    def __init__(self, gridsize, warehouse_location, init, houses_location, houses_delivered, obstacles):
        self.grid = gridsize
        self.warehouse = np.array(warehouse_location)
        self.init = np.array(init)
        self.houses = [np.array(house) for house in houses_location]
        self.drone = Drone(self.warehouse, 100, np.array([0.1,0.1,0.1]))
        self.obstacles = [np.array(obstacle) for obstacle in obstacles]
        self.delivered = np.array(houses_delivered)
        self.delivery = self.houses.copy()
        
        self.utility = np.zeros(gridsize)
        if self.delivered.all():
            self.utility[(0,0,0)] = 90000
        else:
            self.utility[(0,0,0)] = -100
        for i, h in enumerate(self.delivery):
            if self.delivered[i]:
                self.utility[tuple(h)] = -100
            else: 
                self.utility[tuple(h)] = 90000
        for obstacle in self.obstacles:
            self.utility[tuple(obstacle)] = -100
        # print(self.utility)

    def isdone(self):
        return np.all(self.delivered)

    def reset(self):
        self.drone = Drone(self.warehouse, 10, 0.1)
        self.delivered = np.zeros(len(self.houses), dtype=bool)
        self.delivery = self.houses.copy()
        return self.get_state()
    
    def get_state(self):
        return self.drone.position, self.drone.fuel, self.delivery

    def step(self, action):
        prev_fuel = self.drone.fuel
        action = np.array(action).astype(int)
        self.drone.move(action)
        cost = -(self.drone.fuel - prev_fuel)
        future_state = self.get_state()
        
        return self.get_state(), self.utility[tuple(future_state[0])]

    def possible_actions(self, pos):
        possible_moves = np.array([[i, j, k] for i in [-1, 0, 1] for j in [-1, 0, 1] for k in [-1, 0, 1] ])
        valid_moves = []
        if pos == None:
            pos = self.drone.position
        for move in possible_moves:
            new_position = np.array(pos) + np.array(move)
            if (0 <= new_position[0] < self.grid[0] and
                0 <= new_position[1] < self.grid[1] and
                0 <= new_position[2] < self.grid[2]):
                    valid_moves.append(move)
        return np.array(valid_moves)

    def check_house(self, house):
        for i in range(len(self.delivery)):

            if tuple(house) == tuple(self.delivery[i]):
                return i
        return 100
    
    def reset_house(self, house):
        house_idx = self.check_house(house)
        if house_idx == 100:
            return "house is not valid. stopped at the wrong location"
        removed_house = self.delivery[house_idx]
        self.delivered[house_idx] = True
        self.drone.payload[int(house_idx)] = 0
        print('payload weight has been reduced', self.drone.payload)
        print('now new delivery locations',self.delivered)
        
        self.utility[tuple(removed_house)] = -100


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
                        next_step_cost = optimal_utility[i + poss_a[a][0], j + poss_a[a][1], k + poss_a[a][2]] - 0.001*np.linalg.norm(poss_a[a])*np.sum(env.drone.payload)
                        Qvalues[i, j, k, a] = np.append(poss_a[a], Qvalues[i, j, k, a, 3] + alpha * (
                                    current_cost + gamma * next_step_cost))
                        q__ = Qvalues[i, j, k]
                        optimal_utility[i, j, k] = np.max(q__[:, 3])
    return Qvalues, optimal_utility


# Initialize the environment
grid_size = (20, 20, 5)
warehouse_pos = (0, 0, 0)
houses_location = [(0, 19, 0), (15,15,0), (19,0,0)]
deliv = np.array([False, False, False])

obstacles = []

for i in range(grid_size[0]):
    for j in range(grid_size[1]):
        for k in range(grid_size[2]):

            if (5<i<15 and 5<j<15 and k < 3):
                obstacles.append((i, j, k))
            
env = DroneEnvironment(grid_size, warehouse_pos,warehouse_pos, houses_location, deliv, obstacles)
util = env.utility
Qvalues, opt_util = value_iteration(50, env, 0.1)

def simulate_optimal_path(Qvalues, warehouse_position, drone):
    current_position = warehouse_position
    path = [current_position]

    for _ in range(100):
        
        Q = Qvalues[current_position]
        max_idx_q = np.argmax(Q[:, 3])
        optimal_action = Q[max_idx_q, :3]
        if tuple(optimal_action) == (0,0,0):
            optimal_action = (0,0,1)
        next_position = tuple(int(x) for x in np.array(current_position) + optimal_action)
        path.append(next_position)

        if env.check_house(next_position) < 10:
            print('this the position???', next_position)
            break
        current_position = next_position
    return np.sum(drone.payload), path


# Simulate the optimal path starting from the warehouse position
optimal_path = []
current_pos = warehouse_pos
while not env.delivered.all():

    curr_payload, path_segment = simulate_optimal_path(Qvalues, current_pos, env.drone)
    optimal_path.extend((curr_payload,path_segment))
    current_pos = path_segment[-1]
    if env.check_house(current_pos) < 10:
        '''
        only need to do this if the current end position is a house '''
        env.reset_house(current_pos)
        deliv = env.delivered

        new_env = DroneEnvironment(grid_size, warehouse_pos, current_pos, houses_location, deliv, obstacles)
        Qvalues, opt_util = value_iteration(50,new_env,0.1)
curr_payload, path_segment = simulate_optimal_path(Qvalues, current_pos, env.drone)
optimal_path.extend((curr_payload, path_segment))
# Print the optimal path
print("Optimal Path:", optimal_path)
for i, position in enumerate(optimal_path):
    print(f"Step {i + 1}: {position}")

path_3_blocks = optimal_path[1]
path_2_blocks = optimal_path[3]
path_1_blocks = optimal_path[5]
path_0_blocks = optimal_path[7]

index = path_0_blocks.index((1, 1, 1))

# Slice the path to keep elements up to the point right before the first (1, 1, 1)
path_0_blocks = path_0_blocks[:index]

# Add (1, 0, 0) and (0, 0, 0) to the shortened path
path_0_blocks.extend([(1, 0, 0), (0, 0, 0)])

# Extract x, y, z coordinates from the path
x_path_3 = [position[0] for position in path_3_blocks]
y_path_3 = [position[1] for position in path_3_blocks]
z_path_3 = [position[2] for position in path_3_blocks]

x_path_2 = [position[0] for position in path_2_blocks]
y_path_2 = [position[1] for position in path_2_blocks]
z_path_2 = [position[2] for position in path_2_blocks]

x_path_1 = [position[0] for position in path_1_blocks]
y_path_1 = [position[1] for position in path_1_blocks]
z_path_1 = [position[2] for position in path_1_blocks]

x_path_0 = [position[0] for position in path_0_blocks]
y_path_0 = [position[1] for position in path_0_blocks]
z_path_0 = [position[2] for position in path_0_blocks]

x_path = x_path_3 + x_path_2 + x_path_1 + x_path_0
y_path = y_path_3 + y_path_2 + y_path_1 + y_path_0
z_path = z_path_3 + z_path_2 + z_path_1 + z_path_0

# Print the combined paths
print("Combined x_path:", x_path)
print("Combined y_path:", y_path)
print("Combined z_path:", z_path)

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

tck, u = splprep([x_path_3, y_path_3, z_path_3], s=3)  # Increase the smoothing factor `s` for more smoothing
u_dense = np.linspace(0, 1, 300)
x_smooth_3, y_smooth_3, z_smooth_3 = splev(u_dense, tck)

tck, u = splprep([x_path_2, y_path_2, z_path_2], s=3)  # Increase the smoothing factor `s` for more smoothing
u_dense = np.linspace(0, 1, 300)
x_smooth_2, y_smooth_2, z_smooth_2 = splev(u_dense, tck)

tck, u = splprep([x_path_1, y_path_1, z_path_1], s=3)  # Increase the smoothing factor `s` for more smoothing
u_dense = np.linspace(0, 1, 300)
x_smooth_1, y_smooth_1, z_smooth_1 = splev(u_dense, tck)

tck, u = splprep([x_path_0, y_path_0, z_path_0], s=3)  # Increase the smoothing factor `s` for more smoothing
u_dense = np.linspace(0, 1, 300)
x_smooth_0, y_smooth_0, z_smooth_0 = splev(u_dense, tck)

# Create a 3D plot for the smooth path
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the original optimal path
ax.plot(x_path, y_path, z_path, marker='o', markersize=5, color='blue', linestyle='--', label='Original Path')

# Plot the smooth path
ax.plot(x_smooth_3, y_smooth_3, z_smooth_3, color='red', label='Smooth Path 3 Blocks')
ax.plot(x_smooth_2, y_smooth_2, z_smooth_2, color='red', label='Smooth Path 2 Blocks')
ax.plot(x_smooth_1, y_smooth_1, z_smooth_1, color='red', label='Smooth Path 1 Block')
ax.plot(x_smooth_0, y_smooth_0, z_smooth_0, color='red', label='Smooth Path 0 Blocks')


# Set labels and title
ax.set_xlabel('X Coordinate')
ax.set_ylabel('Y Coordinate')
ax.set_zlabel('Z Coordinate')
plt.title('Smoothed Optimal Path in 3D')
plt.legend()

# Show the plot
plt.show()
#
initial_state = np.zeros(12)
trajectory_3 = np.array([x_smooth_3,y_smooth_3,z_smooth_3])
trajectory_2 = np.array([x_smooth_2,y_smooth_2,z_smooth_2])
trajectory_1 = np.array([x_smooth_1,y_smooth_1,z_smooth_1])
trajectory_0 = np.array([x_smooth_0,y_smooth_0,z_smooth_0])
xvals_3,_ = MPC_function.drone_control(trajectory_3,initial_state,0.1,3)
initial_state = xvals_3[-1,:]
xvals_3 = xvals_3.T
xvals_2,_ = MPC_function.drone_control(trajectory_2,initial_state,0.1,2)
initial_state = xvals_2[-1,:]
xvals_2 = xvals_2.T
xvals_1,_ = MPC_function.drone_control(trajectory_1,initial_state,0.1,1)
initial_state = xvals_1[-1,:]
xvals_1 = xvals_1.T
xvals_0,_ = MPC_function.drone_control(trajectory_0,initial_state,0.1,0)
initial_state = xvals_0[-1,:]
xvals_0 = xvals_0.T

xvals = np.hstack((xvals_3,xvals_2,xvals_1,xvals_0))

# Initialize the figure and axes
fig = plt.figure(figsize=(12, 5))
ax = fig.add_subplot(111, projection='3d')

ax.set_xlabel('X Coordinate')
ax.set_ylabel('Y Coordinate')
ax.set_zlabel('Z Coordinate')
plt.title('Drone Delivery Full')

def plot_block(ax, x_start, x_end, y_start, y_end, z_start, z_end, color='black'):
    xx, yy = np.meshgrid([x_start, x_end], [y_start, y_end])
    zz_start = np.full_like(xx, z_start)
    zz_end = np.full_like(xx, z_end)

    ax.plot_surface(xx, yy, zz_start, color=color, alpha=0.5)
    ax.plot_surface(xx, yy, zz_end, color=color, alpha=0.5)

    yy, zz = np.meshgrid([y_start, y_end], [z_start, z_end])
    xx_start = np.full_like(yy, x_start)
    xx_end = np.full_like(yy, x_end)

    ax.plot_surface(xx_start, yy, zz, color=color, alpha=0.5)
    ax.plot_surface(xx_end, yy, zz, color=color, alpha=0.5)

    xx, zz = np.meshgrid([x_start, x_end], [z_start, z_end])
    yy_start = np.full_like(xx, y_start)
    yy_end = np.full_like(xx, y_end)

    ax.plot_surface(xx, yy_start, zz, color=color, alpha=0.5)
    ax.plot_surface(xx, yy_end, zz, color=color, alpha=0.5)




# Animation function
def drawframe(n):
    ax.clear()
    ax.plot3D([0, 1], [0, 0], [0, 0], color="red")
    ax.plot3D([0, 0], [0, 1], [0, 0], color="green")
    ax.plot3D([0, 0], [0, 0], [0, 1], color="blue")

    ax.set_xlim(-2, 20)
    ax.set_ylim(-2, 20)
    ax.set_zlim(-2, 5)

    # Plot the obstacle block
    plot_block(ax, 7, 13, 7, 13, 0, 2, color='black')

    ax.scatter3D(xvals[0, n], xvals[1, n], xvals[2, n])

    phi, psi, theta = float(xvals[3, n]), float(xvals[4, n]), float(xvals[5, n])
    R = MPC_function.get_R(phi, psi, theta)

    ax.plot3D([xvals[0, n], xvals[0, n] + R[0, 0]], [xvals[1, n], xvals[1, n] + R[1, 0]], [xvals[2, n], xvals[2, n] + R[2, 0]], color="red")
    ax.plot3D([xvals[0, n], xvals[0, n] + R[0, 1]], [xvals[1, n], xvals[1, n] + R[1, 1]], [xvals[2, n], xvals[2, n] + R[2, 1]], color="green")
    ax.plot3D([xvals[0, n], xvals[0, n] + R[0, 2]], [xvals[1, n], xvals[1, n] + R[1, 2]], [xvals[2, n], xvals[2, n] + R[2, 2]], color="blue")

# Create the animation
anim = animation.FuncAnimation(fig, drawframe, frames=np.size(xvals[0, :]), interval=20, blit=False)

# Save the animation as a video file
anim.save('drone_animation.mp4', writer='ffmpeg', fps=30)

# Show the plot (animation window)
plt.show()